"""
Inference hooks for the B-Spline / KAN visualizer.

HookedModel wraps GPTModel and installs forward hooks on every
CausalSelfAttention and KANLayer.  After a single forward pass the
captured data is available in self.store — a plain dict of Python
primitives that is safe to serialise as JSON.

To keep the store size bounded, the KAN section stores only the TOP_K
most-activated (input-dim, output-dim) spline pairs per layer.  Their
control points (4 floats each for degree-3 / 7-control-point KANs) are
stored inline so the B-spline detail view works without hitting the model
again.

Store schema
------------
{
  "tokens":        [int, ...],
  "token_labels":  [str, ...],
  "n_layers":      int,
  "ffn_type":      "kan" | "mlp",

  "block_residuals": {"0": float, "1": float, ...},   # mean-abs per block

  "attn_weights": {
      "0": [[[float]]],   # shape (H, T, T), list-of-lists
      ...
  },

  # Only when ffn_type == "kan":
  "kan": {
      "0": {                        # block index as string
          "kan1": {
              "splines": [          # up to TOP_K entries, sorted by mean_abs desc
                  {
                    "i": int,            # input-dim index
                    "j": int,            # output-dim index
                    "mean_abs": float,   # mean |spline contribution| over tokens
                    "mean_val": float,   # mean signed contribution (for colour)
                    "control_points": [float, ...],  # length num_basis
                    "x_scaled_mean": float,  # mean scaled-input value for dim i
                  }, ...
              ],
              "knot_vector":  [float, ...],
              "degree":       int,
              "range_min":    float,
              "range_max":    float,
              "in_features":  int,
              "out_features": int,
              "num_basis":    int,
              "num_ctrl":     int,
          },
          "kan2": { ... }
      },
      ...
  }
}
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm import GPTModel, GPTConfig, CausalSelfAttention
from classes.KAN import KANLayer

# Maximum number of (i, j) spline pairs stored per KAN layer.
# At d_model=256 × d_ff=1584 the full matrix has ~405K entries; we keep
# only the most-activated ones for the browser store.
TOP_K = 500


# ---------------------------------------------------------------------------
# B-spline basis evaluated in NumPy (mirrors KANLayer._compute_basis)
# ---------------------------------------------------------------------------

def _compute_basis_np(
    x_scaled: np.ndarray,    # (N, in_features) in [0, 1]
    knot_vector: np.ndarray, # 1-D, length num_ctrl + 1
    num_ctrl: int,           # == num_control_points
    degree: int,
) -> np.ndarray:
    """Return B-spline basis of shape (N, in_features, num_basis)."""
    eps = 1e-9
    x_exp = x_scaled[:, :, np.newaxis]  # (N, in, 1)

    lower = knot_vector[:num_ctrl][np.newaxis, np.newaxis, :]
    upper = knot_vector[1: num_ctrl + 1][np.newaxis, np.newaxis, :]

    N = ((x_exp >= lower) & (x_exp < upper)).astype(np.float32)
    boundary = (x_scaled == 1.0).astype(np.float32)[:, :, np.newaxis]
    N[:, :, -1:] += boundary

    current_num = num_ctrl
    for d in range(1, degree + 1):
        new_num = current_num - 1
        ki   = knot_vector[:new_num][np.newaxis, np.newaxis, :]
        kid  = knot_vector[d: d + new_num][np.newaxis, np.newaxis, :]
        d1   = np.maximum(kid - ki, eps)
        ki1  = knot_vector[1: new_num + 1][np.newaxis, np.newaxis, :]
        kid1 = knot_vector[d + 1: d + new_num + 1][np.newaxis, np.newaxis, :]
        d2   = np.maximum(kid1 - ki1, eps)
        t1   = ((x_exp - ki) / d1) * N[..., :new_num]
        t2   = ((kid1 - x_exp) / d2) * N[..., 1: new_num + 1]
        N    = t1 + t2
        current_num = new_num

    return N  # (N, in_features, num_basis)


# ---------------------------------------------------------------------------
# HookedModel
# ---------------------------------------------------------------------------

class HookedModel:
    """Wraps a GPTModel, installs hooks, runs one forward pass, fills store."""

    def __init__(self, model: GPTModel, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self._hooks: list = []
        self.store: dict  = {}

    def run(self, prompt: str) -> dict:
        """Run inference on *prompt* and return the populated store dict."""
        self.store = {}
        self._hooks.clear()

        ids    = self.tokenizer.encode(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        ids    = ids.to(device)

        tokens       = ids[0].tolist()
        token_labels = [self.tokenizer.decode([t]) for t in tokens]

        self.store.update({
            "tokens":       tokens,
            "token_labels": token_labels,
            "n_layers":     self.model.cfg.n_layers,
            "ffn_type":     self.model.cfg.ffn_type,
        })

        block_residuals: dict[str, float] = {}
        attn_weights:    dict[str, list]  = {}
        kan_data:        dict[str, dict]  = {}

        for block_idx, block in enumerate(self.model.blocks):
            bidx = str(block_idx)

            # ── Block residual ──────────────────────────────────────────────
            def _block_hook(module, input_, output, bidx=bidx):
                tensor = output[0] if isinstance(output, tuple) else output
                block_residuals[bidx] = float(tensor.detach().abs().mean().cpu())

            self._hooks.append(block.register_forward_hook(_block_hook))

            # ── Attention weights ────────────────────────────────────────────
            # We hook the *pre-softmax* computation by reusing the forward
            # with return_attn=True, but guard against re-entrant calls.
            _attn_busy: list[bool] = [False]  # mutable cell for closure

            def _attn_hook(
                module: CausalSelfAttention, input_, output,
                bidx=bidx, _busy=_attn_busy,
            ):
                if _busy[0]:
                    return
                _busy[0] = True
                try:
                    x_in, cos_in, sin_in = input_[0], input_[1], input_[2]
                    with torch.no_grad():
                        _, w = module(x_in, cos_in, sin_in, return_attn=True)
                    # w: (B, H, T, T) → mean over batch → (H, T, T)
                    attn_weights[bidx] = w.detach().mean(dim=0).cpu().numpy().tolist()
                finally:
                    _busy[0] = False

            self._hooks.append(block.attn.register_forward_hook(_attn_hook))

            # ── KAN layers ──────────────────────────────────────────────────
            if self.model.cfg.ffn_type == "kan":
                kan_data[bidx] = {}

                for layer_name in ("kan1", "kan2"):
                    kan_layer: KANLayer = getattr(block.ffn, layer_name)

                    def _kan_hook(
                        module: KANLayer, input_, output,
                        bidx=bidx, layer_name=layer_name,
                    ):
                        x_in = input_[0]  # (B*T, in_features)

                        # Scale input exactly as the layer does
                        x_sc = (x_in - module.range_min) / (
                            module.range_max - module.range_min
                        )
                        x_sc = torch.clamp(x_sc, 0.0, 1.0)

                        x_sc_np = x_sc.detach().cpu().numpy()         # (T, in)
                        cp_np   = module.control_points.detach().cpu().numpy()  # (in, out, nb)
                        kv_np   = module.knot_vector.cpu().numpy()

                        num_ctrl = module.num_control_points
                        degree   = module.degree
                        num_basis = num_ctrl - degree

                        # Basis: (T, in, num_basis)
                        basis_np = _compute_basis_np(x_sc_np, kv_np, num_ctrl, degree)

                        # Spline contribution per token per (i, j):
                        # contribs[t, i, j] = sum_k basis[t,i,k] * cp[i,j,k]
                        contribs = np.einsum("tik,iok->tio", basis_np, cp_np)  # (T, in, out)

                        mean_abs = np.abs(contribs).mean(axis=0)   # (in, out)
                        mean_val = contribs.mean(axis=0)            # (in, out)
                        x_sc_mean = x_sc_np.mean(axis=0)            # (in,)

                        # Top-K most activated (i, j) pairs
                        flat  = mean_abs.ravel()
                        top_k = min(TOP_K, flat.size)
                        top_idx = np.argpartition(flat, -top_k)[-top_k:]
                        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
                        ii, jj  = np.unravel_index(top_idx, mean_abs.shape)

                        splines = [
                            {
                                "i":             int(i),
                                "j":             int(j),
                                "mean_abs":      float(mean_abs[i, j]),
                                "mean_val":      float(mean_val[i, j]),
                                "control_points": cp_np[i, j, :].tolist(),
                                "x_scaled_mean": float(x_sc_mean[i]),
                            }
                            for i, j in zip(ii.tolist(), jj.tolist())
                        ]

                        kan_data[bidx][layer_name] = {
                            "splines":      splines,
                            "knot_vector":  kv_np.tolist(),
                            "degree":       degree,
                            "range_min":    module.range_min,
                            "range_max":    module.range_max,
                            "in_features":  module.in_features,
                            "out_features": module.out_features,
                            "num_basis":    num_basis,
                            "num_ctrl":     num_ctrl,
                        }

                    self._hooks.append(kan_layer.register_forward_hook(_kan_hook))

        # ── Forward pass ────────────────────────────────────────────────────
        self.model.eval()
        with torch.no_grad():
            self.model(ids)

        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        self.store["block_residuals"] = block_residuals
        self.store["attn_weights"]    = attn_weights
        if self.model.cfg.ffn_type == "kan":
            self.store["kan"] = kan_data

        return self.store


# ---------------------------------------------------------------------------
# Checkpoint loader (mirrors infer.py)
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Return (GPTModel, tokenizer, raw_config_dict)."""
    from transformers import AutoTokenizer

    ckpt      = torch.load(ckpt_path, map_location=device)
    cfg_dict  = ckpt["config"]
    model_cfg = cfg_dict.get("model", {})
    data_cfg  = cfg_dict.get("data",  {})
    kan_cfg   = cfg_dict.get("kan",   {})

    tokenizer = AutoTokenizer.from_pretrained(
        data_cfg.get("tokenizer_name", "gpt2")
    )
    vocab_size = tokenizer.vocab_size

    ffn_type = model_cfg.get("type", "kan")
    d_ff_key = "d_ff_kan" if ffn_type == "kan" else "d_ff_mlp"
    d_ff     = model_cfg.get(d_ff_key, model_cfg.get("d_ff", 2048))

    def _int(v):
        return int(float(v))

    def _flt(v):
        try:
            return float(v)
        except Exception:
            return v

    gpt_cfg = GPTConfig(
        vocab_size          = vocab_size,
        d_model             = _int(model_cfg.get("d_model", 256)),
        n_heads             = _int(model_cfg.get("n_heads", 8)),
        n_layers            = _int(model_cfg.get("n_layers", 6)),
        d_ff                = _int(d_ff),
        max_seq_len         = _int(data_cfg.get("seq_len", 512)),
        dropout             = 0.0,
        ffn_type            = ffn_type,
        kan_control_points  = _int(kan_cfg.get("control_points", 7)),
        kan_degree          = _int(kan_cfg.get("degree", 3)),
        kan_range_min       = _flt(kan_cfg.get("range_min", -3.0)),
        kan_range_max       = _flt(kan_cfg.get("range_max",  3.0)),
    )

    model = GPTModel(gpt_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, cfg_dict
