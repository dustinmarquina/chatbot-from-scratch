import torch

from src.infer_track_c import ChatSession
from src.train_track_c_finetune import freeze_lower_layers


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h = torch.nn.ModuleList([FakeBlock() for _ in range(6)])


class FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = FakeTransformer()


class FakeTokenizer:
    last_text = None
    decode_output = "track c reply"
    eos_token_id = 9

    def __call__(self, text, return_tensors=None):
        self.last_text = text
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, token_ids, skip_special_tokens=True):
        return self.decode_output


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.last_kwargs = None

    def generate(self, input_ids, attention_mask, max_new_tokens, temperature, top_k, pad_token_id):
        self.last_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "pad_token_id": pad_token_id,
        }
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_freeze_lower_layers_freezes_requested_prefix():
    model = FakeCausalLM()

    freeze_lower_layers(model, freeze_layers=4)

    for idx, block in enumerate(model.transformer.h):
        params = list(block.parameters())
        if idx < 4:
            assert all(not param.requires_grad for param in params)
        else:
            assert all(param.requires_grad for param in params)


def test_track_c_chat_session_cleans_follow_on_speaker_turns():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    tokenizer.decode_output = "I can help with that. User: more text"
    session = ChatSession(model=model, tokenizer=tokenizer)

    reply = session.generate_reply([], "hello")

    assert reply == "I can help with that."
    assert tokenizer.last_text == "User: hello\nBot:"
    assert torch.equal(model.last_kwargs["attention_mask"], torch.tensor([[1, 1, 1]]))
