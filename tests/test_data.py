from types import SimpleNamespace

import torch
import pytest

from src.data import (
    build_dialogue_pairs,
    collate_batch,
    encode_dialogue,
    format_dialogue_text,
    has_valid_response_labels,
    load_dailydialog_subset,
)


class DummyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 99
    pad_token = "<pad>"
    pad_token_id = 0

    def __call__(self, text, truncation, max_length, padding, return_tensors):
        token_ids = list(range(1, min(len(text.split()) + 1, max_length + 1)))
        token_ids = token_ids[:max_length]
        attention_mask = [1] * len(token_ids)
        if padding == "max_length":
            pad_len = max_length - len(token_ids)
            token_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor([token_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }


def test_format_dialogue_text_alternates_user_and_bot():
    example = {"dialog": ["hello", "hi there", "how are you?"]}

    text = format_dialogue_text(example)

    assert text.startswith("User: hello")
    assert "Bot: hi there" in text
    assert text.endswith("<eos>")


def test_build_dialogue_pairs_returns_user_bot_examples():
    example = {"dialog": ["hello", "hi there", "how are you?", "I am fine"]}

    pairs = build_dialogue_pairs(example)

    assert pairs == [
        {"prompt": "User: hello\nBot:", "response": "hi there"},
        {"prompt": "User: how are you?\nBot:", "response": "I am fine"},
    ]


def test_encode_dialogue_returns_fixed_length_tensors():
    tokenizer = DummyTokenizer()
    example = {"prompt": "User: hello\nBot:", "response": "hi there"}

    encoded = encode_dialogue(example, tokenizer=tokenizer, max_length=8)

    assert set(encoded.keys()) == {"input_ids", "attention_mask", "labels", "text"}
    assert encoded["input_ids"].shape == (8,)
    assert encoded["labels"].shape == (8,)
    assert encoded["labels"][0].item() == -100
    assert encoded["labels"][1].item() == -100
    assert encoded["labels"][3].item() != -100
    assert encoded["labels"][-1].item() == -100


def test_has_valid_response_labels_rejects_all_masked_examples():
    encoded = {
        "labels": torch.tensor([-100, -100, -100, -100]),
    }

    assert has_valid_response_labels(encoded) is False


def test_has_valid_response_labels_accepts_response_tokens():
    encoded = {
        "labels": torch.tensor([-100, -100, 42, -100]),
    }

    assert has_valid_response_labels(encoded) is True


def test_collate_batch_stacks_items():
    batch = [
        {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]), "labels": torch.tensor([1, 2])},
        {"input_ids": torch.tensor([3, 4]), "attention_mask": torch.tensor([1, 0]), "labels": torch.tensor([3, -100])},
    ]

    collated = collate_batch(batch)

    assert collated["input_ids"].shape == (2, 2)
    assert collated["labels"].shape == (2, 2)


def test_load_dailydialog_subset_falls_back_to_hub_native_source(monkeypatch):
    class FakeSplit:
        def shuffle(self, seed):
            return self

        def select(self, indices):
            return self

    calls = []

    def fake_load_dataset(name):
        calls.append(name)
        if name != "OpenRL/daily_dialog":
            raise RuntimeError("Dataset scripts are no longer supported, but found daily_dialog.py")
        return {"train": FakeSplit(), "validation": FakeSplit()}

    monkeypatch.setattr("src.data.load_dataset", fake_load_dataset)

    dataset = load_dailydialog_subset()

    assert dataset["train"] is not None
    assert calls[0] == "OpenRL/daily_dialog"


def test_load_dailydialog_subset_raises_last_error_when_all_sources_fail(monkeypatch):
    def fake_load_dataset(name):
        raise RuntimeError(f"failed: {name}")

    monkeypatch.setattr("src.data.load_dataset", fake_load_dataset)

    with pytest.raises(RuntimeError, match="failed: daily_dialog"):
        load_dailydialog_subset(
            dataset_sources=("OpenRL/daily_dialog", "roskoN/dailydialog", "daily_dialog")
        )
