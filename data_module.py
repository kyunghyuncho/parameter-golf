"""
data_module.py — FineWeb LightningDataModule for Parameter Golf
================================================================
Streams pre-tokenized binary shards produced by
`data/cached_challenge_fineweb.py` into PyTorch Lightning's training loop.

Binary shard format (written by the upstream pipeline):
  - 256 × int32 header  (magic=20240520, version=1, num_tokens, ...)
  - num_tokens × uint16  token IDs

Each worker is assigned a disjoint subset of shard files and reads them
sequentially. Training workers loop forever; validation workers make
exactly one pass and stop.
"""

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Shard reader — identical to the one in the upstream train_gpt.py
# ---------------------------------------------------------------------------

def load_data_shard(file: Path) -> torch.Tensor:
    """Read a single binary shard and return its tokens as a 1-D uint16 tensor."""
    header_bytes = 256 * np.dtype("<i4").itemsize   # 1024 bytes
    token_bytes = np.dtype("<u2").itemsize            # 2 bytes per token

    # Read and validate the 256-int header
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")

    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")

    # Memory-map the token payload directly, skipping the header
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")

    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


# ---------------------------------------------------------------------------
# IterableDataset — streams (x, y) pairs of length seq_len from shard files
# ---------------------------------------------------------------------------

class TokenStreamDataset(IterableDataset):
    """
    Yields (x, y) pairs where x = tokens[pos : pos+seq_len] and
    y = tokens[pos+1 : pos+seq_len+1], i.e. the standard causal LM
    shifted-by-one target layout.

    Multi-worker safe: each DataLoader worker is assigned a disjoint
    round-robin subset of the shard files.
    """

    def __init__(self, pattern: str, seq_len: int, is_train: bool):
        super().__init__()
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.seq_len = seq_len
        self.is_train = is_train

    def __iter__(self):
        # --- Multi-worker file partitioning ---
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # Each worker only touches its own round-robin slice of all shard files
        my_files = [f for i, f in enumerate(self.files) if i % num_workers == worker_id]
        if not my_files:
            return  # This worker has no files assigned

        file_idx = 0
        while True:
            # Load the entire shard into RAM as int64 (required by nn.Embedding)
            tokens = load_data_shard(my_files[file_idx]).to(torch.int64)
            total_tokens = tokens.numel()

            # Slide a window of (seq_len + 1) tokens across the shard,
            # advancing by seq_len each step (non-overlapping input windows)
            pos = 0
            while pos + self.seq_len + 1 <= total_tokens:
                chunk = tokens[pos : pos + self.seq_len + 1]
                x = chunk[:-1]   # input:  tokens[pos   : pos+seq_len]
                y = chunk[1:]    # target: tokens[pos+1 : pos+seq_len+1]
                yield x, y
                pos += self.seq_len

            # Advance to the next shard (wrapping around for training)
            file_idx = (file_idx + 1) % len(my_files)
            if not self.is_train and file_idx == 0:
                break  # Validation: exactly one pass through all shards


# ---------------------------------------------------------------------------
# LightningDataModule — wires the dataset into Lightning's train/val loops
# ---------------------------------------------------------------------------

class FineWebDataModule(pl.LightningDataModule):
    """
    Parameters
    ----------
    data_dir : str
        Path to the directory containing `fineweb_train_*.bin` / `fineweb_val_*.bin`.
    seq_len : int
        Number of tokens per sequence (context length fed to the model).
    batch_size : int
        Number of *sequences* per training mini-batch.
    val_batch_size : int or None
        Number of *sequences* per validation mini-batch.  Defaults to
        ``batch_size`` if not provided.  Since validation runs under
        ``torch.inference_mode`` (no gradient graph), this can be set
        much larger than the training batch size to saturate GPU compute.
    num_workers : int
        DataLoader worker count.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        batch_size: int,
        val_batch_size: int = None,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        # Default validation batch size to training batch size if not specified
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # No pre-processing required; shards are read on-the-fly by the dataset
        pass

    def train_dataloader(self):
        dataset = TokenStreamDataset(
            f"{self.data_dir}/fineweb_train_*.bin", self.seq_len, is_train=True
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,                            # Async H2D transfer
            persistent_workers=self.num_workers > 0,    # Keep workers alive between epochs
            drop_last=True,                             # Ensures uniform batch shapes
        )

    def val_dataloader(self):
        dataset = TokenStreamDataset(
            f"{self.data_dir}/fineweb_val_*.bin", self.seq_len, is_train=False
        )
        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,             # Can be much larger than train
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
