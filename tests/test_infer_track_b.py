import torch

from src.infer_track_b import ChatSession


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 9
    last_text = None
    decode_output = "track b reply"

    def __call__(self, text, return_tensors=None):
        self.last_text = text
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, token_ids, skip_special_tokens=True):
        return self.decode_output


class FakeModel:
    last_kwargs = None
    device = torch.device("cpu")

    def generate(self, input_ids, max_new_tokens, temperature, top_k, repetition_penalty):
        self.last_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        return [[1, 2, 3, 4, 5]]


def test_chat_session_generate_reply_uses_training_style_prompt():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    session = ChatSession(model=model, tokenizer=tokenizer)

    reply = session.generate_reply(["User: hello", "Bot: hi there"], "how are you?")

    assert reply == "track b reply"
    assert tokenizer.last_text == "User: hello\nBot: hi there\nUser: how are you?\nBot:"
    assert torch.equal(model.last_kwargs["input_ids"], torch.tensor([[1, 2, 3]]))
    assert model.last_kwargs["input_ids"].device.type == "cpu"


def test_chat_session_strips_follow_on_speaker_turns():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    tokenizer.decode_output = "I can help with that. User: what else? Bot: more text"
    session = ChatSession(model=model, tokenizer=tokenizer)

    reply = session.generate_reply([], "hello")

    assert reply == "I can help with that."
