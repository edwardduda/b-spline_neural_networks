import argparse
import time
import torch
from transformers import AutoTokenizer

from models.llm import GPTConfig, GPTModel


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPTModel, AutoTokenizer]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    model_cfg = cfg["model"]
    kan_cfg = cfg["kan"]
    d_ff = model_cfg["d_ff_kan"] if model_cfg["type"] == "kan" else model_cfg["d_ff_mlp"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer_name"])

    gpt_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=d_ff,
        max_seq_len=cfg["data"]["seq_len"],
        dropout=0.0,
        ffn_type=model_cfg["type"],
        kan_control_points=kan_cfg["control_points"],
        kan_degree=kan_cfg["degree"],
        kan_range_min=kan_cfg["range_min"],
        kan_range_max=kan_cfg["range_max"],
    )

    model = GPTModel(gpt_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer


def generate(
    model: GPTModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> tuple[str, float]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)
    elapsed = time.perf_counter() - t0

    tokens_generated = out.shape[1] - idx.shape[1]
    tok_per_sec = tokens_generated / elapsed if elapsed > 0 else float("inf")

    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    return text, tok_per_sec


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained MLP or KAN checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/mlp_best.pt",
        help="Path to checkpoint .pt file (default: checkpoints/mlp_best.pt)",
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time",
        help='Prompt text to continue (default: "Once upon a time")',
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=200,
        help="Number of tokens to generate beyond the prompt (default: 200)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature — lower is sharper, higher is more random (default: 0.8)",
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-k filtering; 0 disables it (default: 40)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: auto-detect)",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else auto_device()
    print(f"Device : {device}")

    print(f"Loading : {args.checkpoint}")
    model, tokenizer = load_model(args.checkpoint, device)

    top_k = args.top_k if args.top_k > 0 else None

    print(f"\nPrompt : {args.prompt!r}")
    print(f"Generating {args.max_new_tokens} tokens "
          f"(temperature={args.temperature}, top_k={top_k})...\n")
    print("─" * 60)

    text, tok_per_sec = generate(
        model, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        device=device,
    )
    print(text)
    print("─" * 60)
    print(f"{tok_per_sec:.1f} tok/s")


if __name__ == "__main__":
    main()
