from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import (
    build_dialogue_pairs,
    collate_batch,
    encode_dialogue,
    has_valid_response_labels,
    load_dailydialog_subset,
)
from src.tokenization import load_tokenizer


def freeze_lower_layers(model: Any, freeze_layers: int) -> None:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        return
    for index, block in enumerate(model.transformer.h):
        requires_grad = index >= freeze_layers
        for parameter in block.parameters():
            parameter.requires_grad = requires_grad


def build_dataloaders(
    tokenizer: Any,
    max_length: int = 128,
    train_size: int = 512,
    validation_size: int = 128,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    dataset = load_dailydialog_subset(train_size=train_size, validation_size=validation_size)
    train_pairs = [pair for example in dataset["train"] for pair in build_dialogue_pairs(example)]
    validation_pairs = [pair for example in dataset["validation"] for pair in build_dialogue_pairs(example)]
    train_items = [
        encoded
        for example in train_pairs
        for encoded in [encode_dialogue(example, tokenizer, max_length=max_length)]
        if has_valid_response_labels(encoded)
    ]
    validation_items = [
        encoded
        for example in validation_pairs
        for encoded in [encode_dialogue(example, tokenizer, max_length=max_length)]
        if has_valid_response_labels(encoded)
    ]
    train_loader = DataLoader(train_items, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    validation_loader = DataLoader(validation_items, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, validation_loader


def run_epoch(model: Any, dataloader: DataLoader, optimizer: AdamW | None, device: torch.device) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Expected loss from pretrained causal LM.")
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite loss during fine-tuning.")
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def train(args: argparse.Namespace) -> Path:
    from transformers import GPT2LMHeadModel

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = load_tokenizer(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    freeze_lower_layers(model, args.freeze_layers)
    model.to(device)
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate)
    train_loader, validation_loader = build_dataloaders(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_size=args.train_size,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        validation_loss = run_epoch(model, validation_loader, None, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={validation_loss:.4f}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained GPT-style model on DailyDialog.")
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--output-dir", default="artifacts/track_c_distilgpt2")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--freeze-layers", type=int, default=4)
    return parser


if __name__ == "__main__":
    model_dir = train(build_parser().parse_args())
    print(f"Saved fine-tuned Track C model to {model_dir}")
