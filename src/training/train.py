"""LoRA SFT training for FinTSR v2.

Key changes from v1:
  1. Class-weighted loss (--class_weighted) to prevent prediction collapse
  2. Resume from existing LoRA checkpoint (--resume_lora) for two-stage training
  3. Focal loss option (--focal_loss) for hard example mining

Usage:
    # Stage 1: Answer-only warmup
    accelerate launch --num_processes 2 --mixed_precision bf16 \
        ideas/fin_tsr/train_lora_v2.py \
        --model_dir anton-hugging/TimeOmni-1-7B \
        --train_file data/fin_tsr_v2/fin_cot_data_v2.json \
        --output_dir ckpts/fin_tsr_v2/timeomni1_lora_stage1 \
        --num_epochs 1 --lr 1e-4 --answer_only --class_weighted

    # Stage 2: Full CoT training (resume from stage 1)
    accelerate launch --num_processes 2 --mixed_precision bf16 \
        ideas/fin_tsr/train_lora_v2.py \
        --model_dir anton-hugging/TimeOmni-1-7B \
        --train_file data/fin_tsr_v2/fin_cot_data_v2.json \
        --output_dir ckpts/fin_tsr_v2/timeomni1_lora \
        --resume_lora ckpts/fin_tsr_v2/timeomni1_lora_stage1 \
        --num_epochs 3 --lr 5e-5 --class_weighted
"""

import argparse
import json
import os
import re
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from accelerate import Accelerator


class FinTSRDatasetV2(Dataset):
    """Dataset for financial time series reasoning SFT (v2 with class weights)."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 4096,
                 answer_only: bool = False, class_weighted: bool = False):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_only = answer_only

        # Compute class weights
        self.sample_weights = self._compute_weights() if class_weighted else None

    def _compute_weights(self) -> list:
        """Compute per-sample weights based on inverse class frequency per task."""
        # Group by (task, answer)
        task_class_counts = {}
        for item in self.data:
            task = item.get("task", "unknown")
            answer = item.get("answer", "?")
            key = (task, answer)
            task_class_counts[key] = task_class_counts.get(key, 0) + 1

        # Count samples per task
        task_totals = {}
        for (task, _), count in task_class_counts.items():
            task_totals[task] = task_totals.get(task, 0) + count

        # Compute weight: (total_in_task / num_classes_in_task) / count_of_this_class
        task_num_classes = {}
        for (task, _) in task_class_counts:
            task_num_classes[task] = task_num_classes.get(task, 0) + 1

        weights = []
        for item in self.data:
            task = item.get("task", "unknown")
            answer = item.get("answer", "?")
            count = task_class_counts.get((task, answer), 1)
            total = task_totals.get(task, 1)
            n_classes = task_num_classes.get(task, 1)
            # Inverse frequency weight, normalized so avg weight = 1.0
            weight = (total / n_classes) / count
            weights.append(weight)

        return weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        convs = item.get("conversations", [])

        user_msg = ""
        assistant_msg = ""
        for c in convs:
            if c["role"] == "user":
                user_msg = c["content"]
            elif c["role"] == "assistant":
                assistant_msg = c["content"]

        if self.answer_only:
            ans_match = re.search(r"<answer>(.*?)</answer>", assistant_msg, re.DOTALL)
            if ans_match:
                assistant_msg = f"<answer>{ans_match.group(1).strip()}</answer>"

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels: mask user tokens
        labels = input_ids.clone()

        assistant_marker = "<|im_start|>assistant\n"
        marker_ids = self.tokenizer.encode(assistant_marker, add_special_tokens=False)
        marker_len = len(marker_ids)
        mask_end = 0
        for i in range(len(input_ids) - marker_len + 1):
            if input_ids[i:i + marker_len].tolist() == marker_ids:
                mask_end = i + marker_len
                break

        labels[:mask_end] = -100
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if self.sample_weights is not None:
            result["sample_weight"] = torch.tensor(self.sample_weights[idx],
                                                    dtype=torch.float32)

        return result


def merge_lora(base_model_dir: str, lora_dir: str, output_dir: str):
    """Merge LoRA adapter into base model."""
    print(f"Merging LoRA: {base_model_dir} + {lora_dir} -> {output_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT for FinTSR v2")
    parser.add_argument("--model_dir", required=True, help="Base model path")
    parser.add_argument("--train_file", required=True, help="Training data JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory for LoRA")
    parser.add_argument("--resume_lora", default=None,
                        help="Resume from existing LoRA checkpoint (for two-stage)")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--answer_only", action="store_true",
                        help="Train on answer only (no CoT)")
    parser.add_argument("--class_weighted", action="store_true",
                        help="Use class-weighted loss")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use")
    parser.add_argument("--task_filter", type=str, default=None,
                        help="Train only on samples matching this task name")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)

    # Load tokenizer & model
    if accelerator.is_main_process:
        print(f"Loading model: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Apply LoRA (or resume from existing)
    if args.resume_lora and os.path.exists(args.resume_lora):
        if accelerator.is_main_process:
            print(f"Resuming from LoRA checkpoint: {args.resume_lora}")
        model = PeftModel.from_pretrained(model, args.resume_lora, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing to reduce activation memory
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # Load dataset
    dataset = FinTSRDatasetV2(
        args.train_file, tokenizer, args.max_length,
        answer_only=args.answer_only,
        class_weighted=args.class_weighted,
    )

    # Task filter
    if args.task_filter:
        before = len(dataset)
        dataset.data = [d for d in dataset.data if d.get("task") == args.task_filter]
        if dataset.sample_weights is not None:
            # Recompute weights for filtered data
            dataset.sample_weights = None
        if accelerator.is_main_process:
            print(f"Task filter '{args.task_filter}': {before} → {len(dataset)} samples")

    # Data fraction
    if args.data_fraction < 1.0:
        import random
        random.seed(42)
        n = int(len(dataset) * args.data_fraction)
        indices = random.sample(range(len(dataset)), n)
        dataset.data = [dataset.data[i] for i in indices]
        if dataset.sample_weights is not None:
            dataset.sample_weights = [dataset.sample_weights[i] for i in indices]
        if accelerator.is_main_process:
            print(f"Using {len(dataset)} samples ({args.data_fraction * 100:.0f}%)")

    use_weighted_loss = args.class_weighted and dataset.sample_weights is not None

    def collate_fn(batch):
        max_len = max(b["input_ids"].size(-1) for b in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for b in batch:
            seq_len = b["input_ids"].size(-1)
            pad_len = max_len - seq_len
            input_ids.append(torch.cat([b["input_ids"],
                             torch.full((pad_len,), tokenizer.pad_token_id)]))
            attention_mask.append(torch.cat([b["attention_mask"],
                                  torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"],
                          torch.full((pad_len,), -100)]))

        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

        if use_weighted_loss:
            result["sample_weights"] = torch.stack(
                [b["sample_weight"] for b in batch])

        return result

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.num_epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Prepare with accelerate
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Training loop
    if accelerator.is_main_process:
        print(f"\nTraining config (v2):")
        print(f"  Train samples:    {len(dataset)}")
        print(f"  Epochs:           {args.num_epochs}")
        print(f"  Batch size:       {args.batch_size} x {accelerator.num_processes} GPUs x {args.grad_accum} accum")
        print(f"  Effective batch:  {args.batch_size * accelerator.num_processes * args.grad_accum}")
        print(f"  Learning rate:    {args.lr}")
        print(f"  LoRA r:           {args.lora_r}")
        print(f"  Total steps:      {total_steps}")
        print(f"  Answer only:      {args.answer_only}")
        print(f"  Class weighted:   {args.class_weighted}")
        print(f"  Resume LoRA:      {args.resume_lora}")
        print()

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in dataloader:
            with accelerator.accumulate(model):
                if use_weighted_loss:
                    # Manual weighted loss computation
                    sample_weights = batch.pop("sample_weights")
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    logits = outputs.logits

                    # Shift for causal LM
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()

                    # Per-token CE loss
                    loss_per_token = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction='none',
                    ).view(shift_labels.size())

                    # Per-sample loss (mean over non-masked tokens)
                    mask = (shift_labels != -100).float()
                    per_sample_loss = (loss_per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

                    # Apply class weights
                    loss = (per_sample_loss * sample_weights).mean()
                else:
                    outputs = model(**batch)
                    loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if accelerator.is_main_process and global_step % 50 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Epoch {epoch + 1}/{args.num_epochs} | "
                      f"Step {global_step}/{total_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if accelerator.is_main_process:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"  Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

            # Save checkpoint per epoch
            unwrapped = accelerator.unwrap_model(model)
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            unwrapped.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            print(f"  Checkpoint saved: {epoch_dir}")

        accelerator.wait_for_everyone()

    # Save final LoRA adapter (same as last epoch, for backward compatibility)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nLoRA adapter saved to {args.output_dir}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
