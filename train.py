import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import torch
import argparse
import math

from data_module import FineWebDataModule
from model import ResidualGRUModel

class ArtifactExportCallback(Callback):
    def __init__(self, filepath="submission_model.pt", max_size_bytes=16777216):
        self.filepath = filepath
        self.max_size_bytes = max_size_bytes

    def on_train_end(self, trainer, pl_module):
        # We only want to save the state dict
        if trainer.is_global_zero:
            print("Exporting artifact...")
            state_dict = pl_module.state_dict()
            # Cast to bf16 and move to CPU
            for k, v in state_dict.items():
                state_dict[k] = v.cpu().to(torch.bfloat16)
            
            torch.save(state_dict, self.filepath)
            
            size = os.path.getsize(self.filepath)
            print(f"Saved artifact to {self.filepath} (Size: {size / 1024 / 1024:.2f} MB)")
            if size > self.max_size_bytes:
                print(f"WARNING: Artifact size {size} bytes exceeds maximum allowed {self.max_size_bytes} bytes!")
            else:
                print("Artifact size is compliant with rules.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--bptt_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--data_dir", type=str, default="data/datasets/fineweb10B_sp1024")
    args = parser.parse_args()

    # The rules mandate sequences of 2048 or 4096, but we slice them using BPTT.
    # We will use seq_len to determine data module sequence chunking length, making it a multiple of BPTT.
    # To keep the dataloader simple, seq_len can be e.g. 2048
    
    seq_len = max(args.seq_len, args.bptt_steps)
    if seq_len % args.bptt_steps != 0:
        seq_len = (seq_len // args.bptt_steps) * args.bptt_steps

    pl.seed_everything(42)

    data_module = FineWebDataModule(
        data_dir=args.data_dir,
        seq_len=seq_len,
        batch_size=max(1, args.batch_size // seq_len), 
        num_workers=4
    )

    model = ResidualGRUModel(
        vocab_size=1024,
        dim=256,
        num_layers=20,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bptt_steps=args.bptt_steps
    )

    wandb_logger = WandbLogger(project="parameter-golf", name="residual-gru")

    trainer = pl.Trainer(
        max_time="00:00:09:45",
        strategy="auto",
        devices=1,
        precision="bf16-mixed",
        logger=wandb_logger,
        enable_checkpointing=False,
        callbacks=[ArtifactExportCallback(filepath="submission_model.pt")]
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
