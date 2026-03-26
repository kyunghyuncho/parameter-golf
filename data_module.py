import glob
import math
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl

def load_data_shard(file: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStreamDataset(IterableDataset):
    def __init__(self, pattern: str, seq_len: int, is_train: bool):
        super().__init__()
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.seq_len = seq_len
        self.is_train = is_train

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # Deterministic file assignment for each worker
        my_files = [f for i, f in enumerate(self.files) if i % num_workers == worker_id]
        if not my_files:
            return

        file_idx = 0
        while True:
            # Load next file
            tokens = load_data_shard(my_files[file_idx]).to(torch.int64)
            
            # Serve chunks
            total_tokens = tokens.numel()
            pos = 0
            while pos + self.seq_len + 1 <= total_tokens:
                chunk = tokens[pos: pos + self.seq_len + 1]
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y
                pos += self.seq_len

            file_idx = (file_idx + 1) % len(my_files)
            if not self.is_train and file_idx == 0:
                break # only one pass for validation

class FineWebDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, seq_len: int, batch_size: int, val_batch_size: int = None, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        dataset = TokenStreamDataset(f"{self.data_dir}/fineweb_train_*.bin", self.seq_len, is_train=True)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )

    def val_dataloader(self):
        dataset = TokenStreamDataset(f"{self.data_dir}/fineweb_val_*.bin", self.seq_len, is_train=False)
        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )
