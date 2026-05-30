from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import (
    build_dialogue_pairs,
    collate_batch,
    encode_dialogue,
    has_valid_response_labels,
    load_dailydialog_subset,
)
from src.model import MiniGPTConfig, MiniGPTLM
from src.tokenization import load_tokenizer


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


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels=labels)
        if loss is None:
            raise RuntimeError("Expected loss when labels are provided")
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite loss during training. Reduce batch size or inspect masked labels.")
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def save_checkpoint(
    checkpoint_path: str | Path,
    model: MiniGPTLM,
    optimizer: AdamW,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config.__dict__,
            "epoch": epoch,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[MiniGPTLM, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = MiniGPTConfig(**payload["config"])
    model = MiniGPTLM(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload


def train(args: argparse.Namespace) -> Path:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer_name)
    config = MiniGPTConfig(
        vocab_size=len(tokenizer),
        block_size=args.max_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPTLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    train_loader, validation_loader = build_dataloaders(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_size=args.train_size,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        validation_loss = run_epoch(model, validation_loader, None, device)
        metrics = {"train_loss": train_loss, "validation_loss": validation_loss}
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={validation_loss:.4f}")
        save_checkpoint(output_dir / "track_b_checkpoint.pt", model, optimizer, epoch=epoch, metrics=metrics)
    return output_dir / "track_b_checkpoint.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Track B MiniGPT on DailyDialog.")
    parser.add_argument("--tokenizer-name", default="distilgpt2")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    checkpoint = train(build_parser().parse_args())
    print(f"Saved checkpoint to {checkpoint}")
