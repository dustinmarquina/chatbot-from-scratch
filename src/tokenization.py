from __future__ import annotations

from typing import Any


DEFAULT_TOKENIZER_NAME = "distilgpt2"


def load_tokenizer(model_name: str = DEFAULT_TOKENIZER_NAME) -> Any:
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_text(tokenizer: Any, text: str, max_length: int) -> dict[str, Any]:
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


def decode_tokens(tokenizer: Any, token_ids: Any) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)
