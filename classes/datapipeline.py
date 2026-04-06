import hashlib
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

load_dotenv()

_hf_token = os.getenv("HG_TOKEN")
if _hf_token:
    login(token=_hf_token, add_to_git_credential=False)


class TokenChunkDataset(Dataset):
    """Pre-tokenised, fixed-length chunks for causal LM training."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.num_chunks = (len(self.token_ids) - 1) // seq_len

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.token_ids[start: start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


class DataPipeline:

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        seq_len: int,
        batch_size: int,
        val_ratio: float,
        num_workers: int,
        cache_dir: str,
    ):
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.cache_dir = Path(cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # We chunk the token stream ourselves, so the tokenizer's built-in
        # model_max_length limit is irrelevant and only produces noise.
        self.tokenizer.model_max_length = int(1e9)

    def _tokenize_split(self, split_data, desc: str = "Tokenizing") -> list[int]:
        all_ids: list[int] = []
        eos = self.tokenizer.eos_token_id
        for example in tqdm(split_data, desc=desc, unit="doc"):
            text = example.get("text") or example.get("story", "")
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(eos)
        return all_ids

    def _cache_key(self) -> str:
        raw = f"{self.dataset_name}|{self.tokenizer_name}|{self.val_ratio}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _cache_paths(self) -> tuple[Path, Path]:
        key = self._cache_key()
        return (
            self.cache_dir / f"{key}_train.pt",
            self.cache_dir / f"{key}_val.pt",
        )

    def prepare(self) -> tuple[DataLoader, DataLoader]:
        """Download, tokenize, chunk, and return (train_loader, val_loader)."""
        train_cache, val_cache = self._cache_paths()

        if train_cache.exists() and val_cache.exists():
            print(f"Loading cached tokens from {self.cache_dir}/")
            train_ids = torch.load(train_cache, weights_only=True).tolist()
            val_ids = torch.load(val_cache, weights_only=True).tolist()
        else:
            print("Downloading dataset...")
            raw = load_dataset(self.dataset_name, split="train")

            print("Splitting into train/val...")
            split = raw.train_test_split(test_size=self.val_ratio, seed=42)
            train_ids = self._tokenize_split(split["train"], desc="Tokenizing train")
            val_ids = self._tokenize_split(split["test"], desc="Tokenizing val")

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor(train_ids, dtype=torch.long), train_cache)
            torch.save(torch.tensor(val_ids, dtype=torch.long), val_cache)
            print(f"Cached tokens to {self.cache_dir}/")

        train_ds = TokenChunkDataset(train_ids, self.seq_len)
        val_ds = TokenChunkDataset(val_ids, self.seq_len)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader, val_loader
