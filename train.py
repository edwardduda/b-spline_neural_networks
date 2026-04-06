import argparse
import math
import os
from datetime import datetime

import matplotlib
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from classes.datapipeline import DataPipeline
from models.llm import GPTConfig, GPTModel


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # PyYAML (YAML 1.1) may parse values like 3e-4 as strings; coerce numerics.
    if "training" in cfg:
        t = cfg["training"]
        for key in ("lr", "weight_decay", "grad_clip"):
            if key in t and isinstance(t[key], str):
                t[key] = float(t[key])
        for int_key in ("log_attention_every", "val_every"):
            if int_key in t and t[int_key] is not None:
                v = t[int_key]
                if isinstance(v, str):
                    v = float(v)
                t[int_key] = int(v)
    return cfg


def build_model(cfg: dict, vocab_size: int, device: torch.device) -> GPTModel:
    model_cfg = cfg["model"]
    kan_cfg = cfg["kan"]
    d_ff = model_cfg["d_ff_kan"] if model_cfg["type"] == "kan" else model_cfg["d_ff_mlp"]

    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=d_ff,
        max_seq_len=cfg["data"]["seq_len"],
        dropout=model_cfg["dropout"],
        ffn_type=model_cfg["type"],
        kan_control_points=kan_cfg["control_points"],
        kan_degree=kan_cfg["degree"],
        kan_range_min=kan_cfg["range_min"],
        kan_range_max=kan_cfg["range_max"],
    )
    return GPTModel(gpt_cfg).to(device)


def _attention_heatmap_to_rgb(heatmap: torch.Tensor) -> torch.Tensor:
    """Normalize to [0,1] and apply a colormap → (3, H, W) float in [0, 1]."""
    h = heatmap.detach().float().cpu().numpy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    try:
        cmap = matplotlib.colormaps["viridis"]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm

        cmap = cm.get_cmap("viridis")
    rgba = cmap(h)  # (H, W, 4), float64
    rgb = torch.from_numpy(rgba[..., :3]).permute(2, 0, 1).to(torch.float32)
    return rgb


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 0):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()
        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break
    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, math.exp(avg_loss)


@torch.no_grad()
def run_validation(
    model,
    val_loader,
    device,
    writer: SummaryWriter,
    global_step: int,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    train_cfg: dict,
    best_val_loss: float,
    ckpt_path: str,
    cfg: dict,
    max_val_batches: int = 0,
) -> float:
    """Val pass (optionally capped), TensorBoard scalars, optional attention images, best checkpoint."""
    val_loss, val_ppl = evaluate(model, val_loader, device, max_batches=max_val_batches)
    writer.add_scalar("val/loss", val_loss, global_step)
    writer.add_scalar("val/perplexity", val_ppl, global_step)
    print(f"  [val @ step {global_step}] val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

    if train_cfg.get("log_attention", True):
        log_attention_maps(writer, model, val_loader, device, global_step)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
        }, ckpt_path)
        print(f"  Saved best checkpoint -> {ckpt_path}")

    return best_val_loss


@torch.no_grad()
def log_attention_maps(writer: SummaryWriter, model, val_loader, device,
                       global_step: int):
    model.eval()
    x, _ = next(iter(val_loader))
    x = x.to(device)
    _, _, attn_maps = model(x, return_attn=True)

    for layer_idx, attn in enumerate(attn_maps):
        n_heads = attn.shape[1]
        for head_idx in range(n_heads):
            heatmap = attn[0, head_idx]
            # RGB via viridis so TensorBoard shows color (1-channel = grayscale only)
            rgb = _attention_heatmap_to_rgb(heatmap)
            writer.add_image(
                f"attention/layer{layer_idx}_head{head_idx}",
                rgb,
                global_step,
                dataformats="CHW",
            )
    writer.flush()
    model.train()


@torch.no_grad()
def sample_generation(model, tokenizer, device, prompt="Once upon a time",
                      max_new_tokens=100):
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    model.train()
    return text


def train(cfg: dict):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    pipeline = DataPipeline(
        dataset_name=data_cfg["dataset_name"],
        tokenizer_name=data_cfg["tokenizer_name"],
        seq_len=data_cfg["seq_len"],
        batch_size=data_cfg["batch_size"],
        val_ratio=data_cfg["val_ratio"],
        num_workers=data_cfg["num_workers"],
        cache_dir=data_cfg["cache_dir"],
    )
    train_loader, val_loader = pipeline.prepare()
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    vocab_size = pipeline.tokenizer.vocab_size
    model = build_model(cfg, vocab_size, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = train_cfg["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps,
    )

    run_name = f"{model_cfg['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    best_val_loss = float("inf")

    # Attention images: also logged each epoch; mid-epoch cadence avoids waiting for huge epochs.
    att_every = train_cfg.get("log_attention_every", 500)
    if att_every is None:
        att_every = 500
    att_every = int(att_every)

    val_every = train_cfg.get("val_every", 1000)
    if val_every is None:
        val_every = 1000
    val_every = int(val_every)

    max_val_batches = train_cfg.get("max_val_batches", 100)
    if max_val_batches is None:
        max_val_batches = 100
    max_val_batches = int(max_val_batches)

    ckpt_path = os.path.join(
        train_cfg["checkpoint_dir"], f"{model_cfg['type']}_best.pt",
    )

    global_step = 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % train_cfg["log_every"] == 0:
                ppl = math.exp(loss.item())
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/perplexity", ppl, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            if (
                train_cfg.get("log_attention", True)
                and att_every > 0
                and global_step % att_every == 0
            ):
                log_attention_maps(writer, model, val_loader, device, global_step)

            if val_every > 0 and global_step % val_every == 0:
                best_val_loss = run_validation(
                    model,
                    val_loader,
                    device,
                    writer,
                    global_step,
                    epoch,
                    optimizer,
                    train_cfg,
                    best_val_loss,
                    ckpt_path,
                    cfg,
                    max_val_batches=max_val_batches,
                )

        # If val_every doesn't land on the last step of this epoch, validate once here.
        if val_every > 0 and global_step % val_every != 0:
            best_val_loss = run_validation(
                model,
                val_loader,
                device,
                writer,
                global_step,
                epoch,
                optimizer,
                train_cfg,
                best_val_loss,
                ckpt_path,
                cfg,
                max_val_batches=max_val_batches,
            )
        elif val_every <= 0:
            best_val_loss = run_validation(
                model,
                val_loader,
                device,
                writer,
                global_step,
                epoch,
                optimizer,
                train_cfg,
                best_val_loss,
                ckpt_path,
                cfg,
                max_val_batches=max_val_batches,
            )

        if train_cfg["generate_every_epoch"]:
            text = sample_generation(
                model, pipeline.tokenizer, device,
                max_new_tokens=train_cfg["generate_tokens"],
            )
            print(f"  [sample] {text[:300]}")
            writer.add_text("generation", text, global_step)

    writer.close()
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train MLP or KAN LLM")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file (the single source of truth for all settings)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
