"""Shared training utilities: LoRA setup, training loop, merge."""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel


def load_model_and_tokenizer(model_dir, use_lora=True, lora_r=16, lora_alpha=32,
                              lora_target_modules=None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if use_lora:
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def get_training_args(output_dir, num_epochs=3, per_device_batch_size=1,
                      gradient_accumulation_steps=16, learning_rate=1e-4,
                      max_steps=-1, warmup_ratio=0.1, logging_steps=10,
                      save_steps=200, bf16=True):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
    )


def train_model(model, tokenizer, train_dataset, training_args):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    return trainer


def merge_lora(base_model_dir, lora_dir, output_dir):
    """Merge LoRA weights back into base model for vLLM inference."""
    print(f"Merging LoRA from {lora_dir} into {base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    merged = model.merge_and_unload()
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")
