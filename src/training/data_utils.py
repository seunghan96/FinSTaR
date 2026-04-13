"""Shared data loading and formatting utilities for all ideas."""
import json
import copy
from torch.utils.data import Dataset


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chat_prompt(system_prompt: str, user_msg: str, assistant_msg: str = None):
    """Build ChatML-formatted prompt."""
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if assistant_msg is not None:
        prompt += assistant_msg + "<|im_end|>"
    return prompt


class SFTDataset(Dataset):
    """Generic SFT dataset. Each item must have 'problem', 'system', 'rationale'.
    If 'rationale' is absent, falls back to wrapping 'response' in <think>/<answer> tags.
    """
    def __init__(self, data, tokenizer, max_length=2048, transform_fn=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for item in data:
            if transform_fn is not None:
                item = transform_fn(copy.deepcopy(item))
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        rationale = item.get("rationale", None)
        if rationale is None:
            resp = item["response"]
            rationale = f"<think>The answer is {resp}.</think>\n<answer>{resp}</answer>"

        full_text = build_chat_prompt(item["system"], item["problem"], rationale)
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        # Mask everything before "assistant\n" so loss is only on the response
        assistant_token = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        seq = input_ids.tolist()
        ast_len = len(assistant_token)
        mask_end = 0
        for i in range(len(seq) - ast_len + 1):
            if seq[i:i + ast_len] == assistant_token:
                mask_end = i + ast_len
                break
        labels[:mask_end] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
