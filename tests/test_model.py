import torch

from src.model import (
    CausalSelfAttention,
    MiniGPTConfig,
    MiniGPTLM,
    ScaledDotProductAttention,
)


def test_scaled_dot_product_attention_applies_causal_mask():
    attention = ScaledDotProductAttention(dropout=0.0)
    q = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 1.0], [9.0, 8.0]]]])
    mask = torch.triu(torch.ones(2, 2), diagonal=1).bool().view(1, 1, 2, 2)

    output, weights = attention(q, k, v, mask)

    assert output.shape == (1, 1, 2, 2)
    assert weights.shape == (1, 1, 2, 2)
    assert torch.allclose(output[0, 0, 0], torch.tensor([2.0, 1.0]), atol=1e-5)
    assert weights[0, 0, 0, 1].item() == 0.0


def test_causal_self_attention_returns_expected_shape():
    config = MiniGPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
    module = CausalSelfAttention(config)
    x = torch.randn(3, 5, 16)

    output, weights = module(x, return_attention=True)

    assert output.shape == (3, 5, 16)
    assert weights.shape == (3, 2, 5, 5)


def test_minigpt_forward_returns_logits_and_loss():
    config = MiniGPTConfig(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = MiniGPTLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    logits, loss = model(input_ids, labels=input_ids)

    assert logits.shape == (2, 6, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)


def test_minigpt_generate_extends_sequence():
    config = MiniGPTConfig(vocab_size=15, block_size=10, n_layer=1, n_head=1, n_embd=8)
    model = MiniGPTLM(config)
    input_ids = torch.tensor([[1, 2, 3]])

    output_ids = model.generate(input_ids, max_new_tokens=4, temperature=1.0, top_k=5)

    assert output_ids.shape == (1, 7)
