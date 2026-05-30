from src.tokenization import load_tokenizer

class DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"


def test_load_tokenizer_prefers_local_cache_first(monkeypatch):
    calls = []
    tokenizer = DummyTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, local_files_only=False):
            calls.append((model_name, local_files_only))
            return tokenizer

    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)

    loaded = load_tokenizer("distilgpt2")

    assert loaded is tokenizer
    assert calls == [("distilgpt2", True)]
    assert loaded.pad_token == loaded.eos_token
