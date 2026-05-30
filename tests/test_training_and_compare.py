import json

import torch

from src.compare import compare_models
from src.compare_track_c import compare_track_c
from src.model import MiniGPTConfig, MiniGPTLM
from src.train_track_b import save_checkpoint


class FakeTrackA:
    def generate_reply(self, history, user_message):
        return f"A:{user_message}"


class FakeTrackB:
    def generate_reply(self, history, user_message):
        return f"B:{user_message}"


class FakeTrackC:
    def generate_reply(self, history, user_message):
        return f"C:{user_message}"


def test_save_checkpoint_writes_model_state(tmp_path):
    model = MiniGPTLM(MiniGPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=1, n_embd=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(checkpoint_path, model, optimizer, epoch=1, metrics={"train_loss": 1.2})

    assert checkpoint_path.exists()
    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["epoch"] == 1
    assert payload["metrics"]["train_loss"] == 1.2


def test_compare_models_returns_non_empty_outputs(tmp_path):
    output_path = tmp_path / "compare.json"
    prompts = ["hello", "tell me a joke"]

    rows = compare_models(
        prompts=prompts,
        track_a=FakeTrackA(),
        track_b=FakeTrackB(),
        output_path=output_path,
    )

    assert len(rows) == 2
    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved[0]["track_a_reply"] == "A:hello"
    assert saved[0]["track_b_reply"] == "B:hello"


def test_compare_track_c_returns_non_empty_outputs(tmp_path):
    output_path = tmp_path / "compare_track_c.json"
    prompts = ["hello", "tell me a joke"]

    rows = compare_track_c(
        prompts=prompts,
        track_a=FakeTrackA(),
        track_c=FakeTrackC(),
        output_path=output_path,
    )

    assert len(rows) == 2
    saved = json.loads(output_path.read_text())
    assert saved[0]["track_a_reply"] == "A:hello"
    assert saved[0]["track_c_reply"] == "C:hello"
