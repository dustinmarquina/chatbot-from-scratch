from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DialogueGenerator:
    model: Any
    tokenizer: Any
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50

    def build_prompt(self, history: list[str], user_message: str) -> str:
        eos = self.tokenizer.eos_token or ""
        turns = [self._normalize_turn(item) for item in history if item.strip()]
        turns.append(user_message.strip())
        return "".join(f"{turn}{eos}" for turn in turns if turn)

    @staticmethod
    def _normalize_turn(turn: str) -> str:
        for prefix in ("Context:", "User:", "Bot:"):
            if turn.startswith(prefix):
                return turn[len(prefix) :].strip()
        return turn.strip()

    def generate_reply(self, history: list[str], user_message: str) -> str:
        prompt = self.build_prompt(history, user_message)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.as_tensor(encoded["attention_mask"], dtype=torch.long)
        generated = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        reply_tokens = generated[0][input_ids.shape[1] :]
        return self.tokenizer.decode(reply_tokens, skip_special_tokens=True).strip()


def load_track_a(model_name: str = "microsoft/DialoGPT-medium") -> DialogueGenerator:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return DialogueGenerator(model=model, tokenizer=tokenizer)


def interactive_chat() -> None:
    generator = load_track_a()
    history: list[str] = []
    print("Track A chat. Type 'quit' to exit.")
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        reply = generator.generate_reply(history, user_message)
        history.extend([user_message, reply])
        print(f"DialoGPT: {reply}")


if __name__ == "__main__":
    interactive_chat()
