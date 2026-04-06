import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from classes.KAN import KANLayer


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, H, T, head_dim)
    # cos, sin: (T, head_dim) -> broadcast over B and H
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)          # (max_seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)   # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float
    ffn_type: str
    kan_control_points: int
    kan_degree: int
    kan_range_min: float
    kan_range_max: float


class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
            .unsqueeze(0).unsqueeze(0),          # (1, 1, T, T)
        )

    def forward(self, x, cos, sin, return_attn=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)

        q, k = apply_rotary_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.resid_drop(self.out_proj(out))

        if return_attn:
            return out, attn
        return out


class MLPFeedForward(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class KANFeedForward(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.kan1 = KANLayer(
            cfg.d_model, cfg.d_ff,
            num_control_points=cfg.kan_control_points,
            degree=cfg.kan_degree,
            range_min=cfg.kan_range_min,
            range_max=cfg.kan_range_max,
        )
        self.kan2 = KANLayer(
            cfg.d_ff, cfg.d_model,
            num_control_points=cfg.kan_control_points,
            degree=cfg.kan_degree,
            range_min=cfg.kan_range_min,
            range_max=cfg.kan_range_max,
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        h = x.reshape(B * T, C)
        h = self.kan1(h)
        h = self.kan2(h)
        h = h.reshape(B, T, -1)
        return self.drop(h)


class TransformerBlock(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = (
            KANFeedForward(cfg) if cfg.ffn_type == "kan"
            else MLPFeedForward(cfg)
        )

    def forward(self, x, cos, sin, return_attn=False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.ln1(x), cos, sin, return_attn=True)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, attn_weights
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, cfg.max_seq_len)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self._print_param_summary()

    def _print_param_summary(self):
        def count(module):
            return sum(p.numel() for p in module.parameters())

        emb   = count(self.tok_emb)  # rope has no learned parameters
        attn  = sum(count(b.attn) for b in self.blocks)
        ffn   = sum(count(b.ffn)  for b in self.blocks)
        ln    = sum(count(b.ln1) + count(b.ln2) for b in self.blocks) + count(self.ln_f)
        total = count(self)

        w = len(f"{total:,}")
        print(f"\n{'─'*40}")
        print(f"  Model ({self.cfg.ffn_type.upper()})  parameter summary")
        print(f"{'─'*40}")
        print(f"  {'Embeddings':<18} {emb:{w},}")
        print(f"  {'Attention':<18} {attn:{w},}")
        print(f"  {'FFN':<18} {ffn:{w},}")
        print(f"  {'Layer norms':<18} {ln:{w},}")
        print(f"  {'─'*30}")
        print(f"  {'Total':<18} {total:{w},}")
        print(f"{'─'*40}\n")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attn=False):
        B, T = idx.shape

        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope(T)

        attn_maps = []
        for block in self.blocks:
            if return_attn:
                x, attn_w = block(x, cos, sin, return_attn=True)
                attn_maps.append(attn_w)
            else:
                x = block(x, cos, sin)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        if return_attn:
            return logits, loss, attn_maps
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx