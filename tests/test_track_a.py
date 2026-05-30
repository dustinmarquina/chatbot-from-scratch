import torch

from src.track_a import DialogueGenerator


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 7
    last_text = None

    def __call__(self, text, return_tensors=None):
        self.last_text = text
        return {
            "input_ids": [[1, 2, 3, self.eos_token_id]],
            "attention_mask": [[1, 1, 1, 1]],
        }

    def decode(self, tokens, skip_special_tokens=True):
        return "stub reply"


class FakeModel:
    last_kwargs = None

    def generate(self, input_ids, **kwargs):
        self.last_kwargs = kwargs
        suffix = torch.tensor([[4, 5, 7]])
        return torch.cat((input_ids, suffix), dim=1)


def test_generate_reply_uses_eos_delimited_history_and_attention_mask():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    generator = DialogueGenerator(model=model, tokenizer=tokenizer)

    reply = generator.generate_reply(["hello", "hi there"], "how are you?")

    assert reply == "stub reply"
    assert tokenizer.last_text == "hello<eos>hi there<eos>how are you?<eos>"
    assert torch.equal(model.last_kwargs["attention_mask"], torch.tensor([[1, 1, 1, 1]]))
    assert model.last_kwargs["do_sample"] is True
