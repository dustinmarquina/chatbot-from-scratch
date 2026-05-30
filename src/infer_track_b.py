from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch

from src.train_track_b import load_checkpoint
from src.tokenization import decode_tokens, load_tokenizer


@dataclass
class ChatSession:
    model: Any
    tokenizer: Any
    max_new_tokens: int = 32
    temperature: float = 0.5
    top_k: int = 30
    repetition_penalty: float = 1.2

    def build_prompt(self, history: list[str], user_message: str) -> str:
        turns = [self._normalize_turn(item) for item in history if item.strip()]
        turns.append(f"User: {user_message.strip()}")
        turns.append("Bot:")
        return "\n".join(turns)

    @staticmethod
    def _normalize_turn(turn: str) -> str:
        turn = turn.strip()
        if turn.startswith("User:") or turn.startswith("Bot:"):
            return turn
        return f"User: {turn}"

    def generate_reply(self, history: list[str], user_message: str) -> str:
        prompt = self.build_prompt(history, user_message)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "parameters"):
            model_device = next(self.model.parameters()).device
        else:
            model_device = getattr(self.model, "device", torch.device("cpu"))
        input_ids = encoded["input_ids"].clone().detach().to(device=model_device, dtype=torch.long)
        generated = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        reply_tokens = generated[0][input_ids.shape[1] :]
        decoded = decode_tokens(self.tokenizer, reply_tokens).strip()
        return self._clean_reply(decoded)

    @staticmethod
    def _clean_reply(text: str) -> str:
        cleaned = text.strip()
        for marker in ("\nUser:", "\nBot:", " User:", " Bot:"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        return cleaned.removeprefix("Bot:").strip()


def load_track_b(checkpoint_path: str, tokenizer_name: str = "distilgpt2") -> ChatSession:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_checkpoint(checkpoint_path, device=device)
    tokenizer = load_tokenizer(tokenizer_name)
    return ChatSession(model=model, tokenizer=tokenizer)


def interactive_chat(args: argparse.Namespace) -> None:
    session = load_track_b(args.checkpoint_path, args.tokenizer_name)
    history: list[str] = []
    print("Track B chat. Type 'quit' to exit.")
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        reply = session.generate_reply(history, user_message)
        history.extend([f"User: {user_message}", f"Bot: {reply}"])
        print(f"MiniGPT: {reply}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with the trained Track B model.")
    parser.add_argument("--checkpoint-path", default="artifacts/checkpoints/track_b_checkpoint.pt")
    parser.add_argument("--tokenizer-name", default="distilgpt2")
    return parser


if __name__ == "__main__":
    interactive_chat(build_parser().parse_args())
