# B-Spline Neural Networks

A research project comparing standard Transformer language models against ones where the MLP feed-forward layers are replaced with **Kolmogorov–Arnold Network (KAN)** layers backed by B-spline basis functions.

Both variants are trained on the [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) dataset — a corpus of simple English children's stories — making it easy to evaluate generation quality without needing massive compute.

---

## How it works

### Standard model (MLP)

Each Transformer block uses a two-layer MLP feed-forward network:

```
x → Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, d_model) → Dropout
```

### KAN model

The MLP is replaced with two `KANLayer` modules. Each KAN layer learns a separate **B-spline activation function** for every input–output pair rather than using a fixed non-linearity:

```
x → KANLayer(d_model, d_ff) → KANLayer(d_ff, d_model) → Dropout
```

Each output is a sum of a spline term and a learnable residual:

```
φ(x)_j = Σ_i  w_residual_{i,j} · SiLU(x_i)  +  Σ_k  N_k(x_i) · w_spline_{i,j,k}
```

where `N_k` are the B-spline basis functions evaluated via the Cox–de Boor recursion and `w_spline` are the learned control-point weights. The SiLU residual stabilises training by providing a smooth gradient path even before the splines are fitted.

Both variants share the same attention mechanism: multi-head causal self-attention with **Rotary Positional Embeddings (RoPE)**.

---

## Project structure

```
.
├── classes/
│   ├── KAN.py           # B-spline KAN layer (Cox–de Boor basis, residual SiLU)
│   └── datapipeline.py  # Dataset download, tokenisation, and local caching
├── models/
│   └── llm.py           # GPT architecture (RoPE attention, MLP/KAN blocks, generation)
├── train.py             # Training loop, evaluation, TensorBoard logging, checkpointing
├── infer.py             # Text generation from a saved checkpoint
├── config.yaml          # Single source of truth for all hyperparameters
├── requirements.txt
└── .env                 # HuggingFace token (optional, for gated models)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need access to gated HuggingFace models, create a `.env` file:

```
HG_TOKEN=hf_...
```

---

## Configuration

**`config.yaml` is the single source of truth.** Edit it before training — nothing else needs to be touched.

```yaml
data:
  dataset_name: "karpathy/tinystories-gpt4-clean"  # HuggingFace dataset
  tokenizer_name: "gpt2"
  seq_len: 512          # Token sequence length per training example
  batch_size: 8
  val_ratio: 0.05       # Fraction of data held out for validation
  num_workers: 0        # DataLoader worker processes
  cache_dir: "data_cache"

model:
  type: "kan"           # "mlp" or "kan" — selects the feed-forward block
  d_model: 512          # Residual stream width
  n_heads: 8
  n_layers: 6
  d_ff_mlp: 512         # Hidden dimension for MLP blocks
  d_ff_kan: 512         # Hidden dimension for KAN blocks
  dropout: 0.1

kan:
  control_points: 7     # Must satisfy: control_points >= 2*degree + 1
  degree: 3             # B-spline polynomial degree
  range_min: -1.0       # Input range for spline evaluation
  range_max: 5.0

training:
  epochs: 10
  lr: 0.0003
  weight_decay: 0.00001
  grad_clip: 1.0
  log_every: 50               # Optimizer steps between scalar TensorBoard logs
  val_every: 1000             # Steps between validation runs; 0 = end of epoch only
  max_val_batches: 100        # Cap val set size per validation call; 0 = full val set
  log_attention: true         # Log causal attention heatmaps to TensorBoard Images
  log_attention_every: 500    # Steps between attention image dumps; 0 = epoch end only
  generate_every_epoch: true  # Sample text at the end of each epoch
  generate_tokens: 100
  checkpoint_dir: "checkpoints"
```

---

## Training

```bash
python train.py
```

To use a different config file:

```bash
python train.py --config my_experiment.yaml
```

**To switch between MLP and KAN**, set `model.type` in `config.yaml`:

```yaml
model:
  type: "kan"   # or "mlp"
```

---

## Monitoring

Training metrics are logged to TensorBoard:

| What | TensorBoard tab | Notes |
|------|-----------------|-------|
| Loss, perplexity, LR | **Scalars** | Train every `log_every` steps; val every `val_every` steps (and at epoch end if the last step isn't a multiple of `val_every`). Set `val_every: 0` for val only at epoch end. |
| **Attention patterns** | **Images** | Causal softmax attention per layer/head, tags `attention/layer*_head*`. Logged every `log_attention_every` steps and on each validation run. Open the **Images** tab, not Scalars. |
| Sample generation | **Text** | If `generate_every_epoch: true` |

Set `log_attention: false` to disable attention logging entirely. Use `log_attention_every: 0` to log only at epoch end.

```bash
tensorboard --logdir runs/
```

Then open [http://localhost:6006](http://localhost:6006).

---

## Data caching

The first run downloads and tokenises the dataset, then saves the token IDs to `data_cache/`. Subsequent runs load directly from disk — no re-download needed. The cache key encodes the dataset name, tokenizer, and val split ratio, so changing any of those will trigger a fresh cache automatically.

---

## Checkpoints

The best checkpoint (by validation loss) is saved to `checkpoints/<model_type>_best.pt` whenever a new validation loss minimum is reached. The full `config.yaml` dict is embedded in the checkpoint, so the exact architecture can always be reconstructed from the file alone.

---

## Inference

After training, use `infer.py` to generate text from any saved checkpoint. It reads the config embedded in the `.pt` file and rebuilds the model automatically.

```bash
# Default: MLP checkpoint, 200 tokens
python infer.py

# KAN checkpoint with custom sampling
python infer.py --checkpoint checkpoints/kan_best.pt \
                --prompt "Once upon a time" \
                --max-new-tokens 300 --temperature 1.0 --top-k 50
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `checkpoints/mlp_best.pt` | Path to the `.pt` checkpoint file |
| `--prompt` | `"Once upon a time"` | Text prompt to continue |
| `--max-new-tokens` | `200` | Number of tokens to generate beyond the prompt |
| `--temperature` | `0.8` | `< 1.0` → sharper / more focused; `> 1.0` → more random |
| `--top-k` | `40` | Restrict sampling to the top-k tokens at each step; `0` disables |
| `--device` | auto | Force a device: `cpu`, `cuda`, or `mps` |

### How generation works

Each new token is produced auto-regressively:

1. The context window is trimmed to `seq_len` tokens if it grows too long.
2. The model produces a logit vector over the full vocabulary for the **last** position.
3. Logits are divided by `temperature` — lower values make the distribution sharper.
4. **Top-k filtering** zeroes out all but the `k` highest-probability tokens.
5. A single token is sampled from the resulting softmax distribution.
6. That token is appended to the sequence and the loop repeats.

`infer.py` also prints tokens-per-second throughput after generation completes.

---

## Notes on laptop / low-memory usage

The KAN variant is more memory-intensive than MLP because the B-spline basis computation creates intermediate tensors of shape `(batch × seq_len, d_model, num_control_points)` which must all be retained for the backward pass. If training gets killed by the OS, reduce these values in `config.yaml`:

```yaml
data:
  seq_len: 64      # was 512
  batch_size: 2    # was 8

model:
  d_model: 256     # was 512
  n_layers: 4      # was 6
```
