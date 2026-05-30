from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset

from src.tokenization import encode_text


def format_dialogue_text(example: dict[str, Any], eos_token: str = "<eos>") -> str:
    turns = example.get("dialog", [])
    formatted = []
    for index, turn in enumerate(turns):
        speaker = "User" if index % 2 == 0 else "Bot"
        formatted.append(f"{speaker}: {turn}")
    return "\n".join(formatted) + f"\n{eos_token}"


def build_dialogue_pairs(example: dict[str, Any]) -> list[dict[str, str]]:
    turns = example.get("dialog", [])
    pairs: list[dict[str, str]] = []
    for index in range(0, len(turns) - 1, 2):
        user_turn = turns[index].strip()
        bot_turn = turns[index + 1].strip()
        if not user_turn or not bot_turn:
            continue
        pairs.append({"prompt": f"User: {user_turn}\nBot:", "response": bot_turn})
    return pairs


def encode_dialogue(
    example: dict[str, Any],
    tokenizer: Any,
    max_length: int = 128,
) -> dict[str, torch.Tensor | str]:
    eos_token = tokenizer.eos_token or "<eos>"
    if "prompt" in example and "response" in example:
        prompt_text = example["prompt"].strip()
        response_text = example["response"].strip()
        text = f"{prompt_text} {response_text}{eos_token}"
        prompt_encoded = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
        )
        prompt_length = prompt_encoded["input_ids"].shape[1]
    else:
        text = format_dialogue_text(example, eos_token=eos_token)
        prompt_length = 0
    encoded = encode_text(tokenizer, text, max_length=max_length)
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    if prompt_length > 0:
        labels[: min(prompt_length, labels.shape[0])] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "text": text,
    }


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def has_valid_response_labels(encoded_example: dict[str, torch.Tensor | str]) -> bool:
    labels = encoded_example["labels"]
    if not isinstance(labels, torch.Tensor):
        return False
    return bool((labels != -100).any().item())


def load_dailydialog_subset(
    train_size: int = 512,
    validation_size: int = 128,
    seed: int = 42,
    dataset_sources: tuple[str, ...] = (
        "OpenRL/daily_dialog",
        "roskoN/dailydialog",
        "daily_dialog",
    ),
) -> dict[str, Any]:
    last_error: Exception | None = None
    dataset = None
    for source in dataset_sources:
        try:
            dataset = load_dataset(source)
            break
        except Exception as exc:
            last_error = exc
            continue
    if dataset is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not load any DailyDialog dataset source.")
    train_split = dataset["train"].shuffle(seed=seed).select(range(train_size))
    validation_split = dataset["validation"].shuffle(seed=seed).select(range(validation_size))
    return {"train": train_split, "validation": validation_split}
