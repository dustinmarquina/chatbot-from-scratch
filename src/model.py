from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class MiniGPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    dropout: float = 0.1


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, v)
        return output, weights


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attention = ScaledDotProductAttention(dropout=config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, return_attention: bool = False) -> tuple[Tensor, Tensor | None]:
        batch_size, seq_len, emb_dim = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = mask.view(1, 1, seq_len, seq_len)
        attended, weights = self.attention(q, k, v, mask=mask)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        output = self.resid_dropout(self.out_proj(attended))
        return output, weights if return_attention else None


class MLP(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, return_attention: bool = False) -> tuple[Tensor, Tensor | None]:
        attn_out, weights = self.attn(self.ln_1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, weights


class MiniGPTLM(nn.Module):
    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight
        self.apply(self.init_weights)


    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block size")
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.dropout(x)
        last_weights = None
        for block in self.blocks:
            x, last_weights = block(x, return_attention=return_attention)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if return_attention:
            return logits, loss, last_weights
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        self.eval()
        output = input_ids
        for _ in range(max_new_tokens):
            idx_cond = output[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if repetition_penalty != 1.0:
                for batch_idx in range(output.size(0)):
                    for token_id in set(output[batch_idx].tolist()):
                        if next_token_logits[batch_idx, token_id] < 0:
                            next_token_logits[batch_idx, token_id] *= repetition_penalty
                        else:
                            next_token_logits[batch_idx, token_id] /= repetition_penalty
            if top_k is not None:
                values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                cutoff = values[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            output = torch.cat((output, next_token), dim=1)
        return output
