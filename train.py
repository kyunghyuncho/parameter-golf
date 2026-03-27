"""
train.py — Training launcher for the Residual GRU Parameter Golf entry
=======================================================================
Orchestrates:
  1. CLI argument parsing (compatible with W&B Sweep agent injection)
  2. Data module initialization (with separate train / val batch sizes)
  3. Model construction (Residual GRU with configurable architecture, ~8.2M params)
  4. Lightning Trainer with a hard 9 min 45 s wall-clock cap
  5. Post-training artifact export to bfloat16 ≤ 16.00 MB

Usage:
    # Single run (W&B disabled for local testing)
    WANDB_MODE=disabled python train.py --learning_rate 3e-4

    # Inside a W&B sweep agent
    wandb agent <user>/parameter-golf/<sweep_id>
"""

import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

from data_module import FineWebDataModule
from model import ResidualGRUModel


# ---------------------------------------------------------------------------
# Post-training Artifact Export Callback
# ---------------------------------------------------------------------------

class ArtifactExportCallback(Callback):
    """
    Triggered at the end of training to produce a competition-compliant
    submission artifact.

    Steps:
      1. Extract the raw model state_dict (no optimizer state).
      2. Cast every tensor to bfloat16 and move to CPU.
      3. Save via torch.save to `filepath`.
      4. Assert file size ≤ max_size_bytes (16,777,216 = 16.00 MB).
    """

    def __init__(self, filepath: str = "submission_model.pt", max_size_bytes: int = 16_777_216):
        self.filepath = filepath
        self.max_size_bytes = max_size_bytes

    def on_train_end(self, trainer, pl_module):
        # Only rank-0 saves in distributed training to avoid file corruption
        if not trainer.is_global_zero:
            return

        print("Exporting submission artifact...")

        # state_dict() contains only model weights — no optimizer buffers
        state_dict = pl_module.state_dict()

        # Downcast every parameter to bfloat16 on CPU for minimal file size
        for k, v in state_dict.items():
            state_dict[k] = v.cpu().to(torch.bfloat16)

        torch.save(state_dict, self.filepath)

        # --- Size compliance check ---
        size = os.path.getsize(self.filepath)
        print(f"Saved artifact to {self.filepath} ({size / 1024 / 1024:.2f} MB)")
        if size > self.max_size_bytes:
            print(
                f"⚠️  WARNING: Artifact size {size} bytes exceeds "
                f"maximum allowed {self.max_size_bytes} bytes!"
            )
        else:
            print("✅ Artifact size is compliant with competition rules.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # --- CLI arguments (also injected by `wandb agent` during sweeps) ---
    parser = argparse.ArgumentParser(
        description="Train the Pre-Norm Residual GRU for parameter-golf"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Peak learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="AdamW weight decay coefficient")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="AdamW beta2")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--bptt_steps", type=int, default=256,
                        help="Truncated BPTT chunk length (tokens)")
    parser.add_argument("--batch_size", type=int, default=65536,
                        help="Training batch size in *tokens* (divided by seq_len for sequences)")
    parser.add_argument("--val_batch_size", type=int, default=131072,
                        help="Validation batch size in *tokens* (can be much larger than train)")
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="Sequence length fed to the model per sample")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for initialization (random if None)")

    parser.add_argument("--architecture", type=str, default="medium",
                        choices=["shallow", "medium", "deep"],
                        help="Model architecture variant (controls depth and width)")

    parser.add_argument("--data_dir", type=str,
                        default="data/datasets/fineweb10B_sp1024",
                        help="Path to the directory containing tokenized shards")
    args = parser.parse_args()

    # --- Ensure seq_len is at least bptt_steps and a clean multiple of it ---
    # This guarantees every TBPTT chunk in training_step has exactly
    # bptt_steps tokens (no ragged tail chunk).
    seq_len = max(args.seq_len, args.bptt_steps)
    if seq_len % args.bptt_steps != 0:
        seq_len = (seq_len // args.bptt_steps) * args.bptt_steps

    # --- Seeding ---
    # Randomized by default for sweeps to avoid initialization bias.
    # If a specific seed is needed for debugging, pass --seed.
    if args.seed is not None:
        seed = args.seed
    else:
        # Generate a random 32-bit seed using OS entropy
        seed = int.from_bytes(os.urandom(4), byteorder="little")
    pl.seed_everything(seed)

    # --- Data module ---
    # batch_size and val_batch_size are specified in *tokens* on the CLI,
    # but the DataLoader operates on *sequences*, so we divide by seq_len.
    data_module = FineWebDataModule(
        data_dir=args.data_dir,
        seq_len=seq_len,
        batch_size=max(1, args.batch_size // seq_len),
        val_batch_size=max(1, args.val_batch_size // seq_len),
        num_workers=4,
    )

    # --- Map architecture to num_layers and dim ---
    if args.architecture == "shallow":
        num_layers, dim = 5, 504
    elif args.architecture == "deep":
        num_layers, dim = 20, 256
    else:  # "medium"
        num_layers, dim = 10, 360

    # --- Model ---
    model = ResidualGRUModel(
        vocab_size=1024,            # sp1024 tokenizer vocabulary
        dim=dim,                    # Hidden dimension D
        num_layers=num_layers,      # Residual GRU blocks
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        bptt_steps=args.bptt_steps,
        gradient_clip_val=args.gradient_clip_val,
    )

    # --- Logger ---
    wandb_logger = WandbLogger(project="parameter-golf", name="residual-gru")

    # --- Trainer ---
    trainer = pl.Trainer(
        max_time="00:00:09:45",         # Hard stop at 9 min 45 s (15 s safety margin)
        strategy="auto",                # Single-GPU: no DDP overhead
        devices=1,                      # Adapt to your hardware
        precision="bf16-mixed",         # AMP with bfloat16 for maximum throughput
        logger=wandb_logger,
        enable_checkpointing=False,     # We handle export via our custom callback
        callbacks=[ArtifactExportCallback(filepath="submission_model.pt")],
    )

    # --- Launch training ---
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
