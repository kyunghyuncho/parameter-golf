"""
model.py — Pre-Norm Residual Deep GRU for Parameter Golf
=========================================================
Implements a 10-layer residual GRU language model that fits within the
strict 16.00 MB (8,388,608 bfloat16 parameters) artifact limit.

Architecture per layer:
    x̄ = LayerNorm(x)
    h_out, h_next = GRU(x̄, h_prev)   # cuDNN-accelerated sequence GRU
    x_next = x + h_out                 # residual skip connection

Key design decisions:
  - Uses nn.GRU (not GRUCell) so the entire sequence is processed by a
    single fused cuDNN kernel per layer, maximizing GPU utilization.
  - Orthogonal init on W_hh keeps transition eigenvalues near 1 for
    stable gradients across the time dimension.
  - Tied embedding/output head saves V×D parameters.
  - Manual TBPTT: hidden states are detached every `bptt_steps` tokens
    to bound the backward graph while preserving forward recurrent state.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Utility Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Normalizes the variance of the input features and applies a learned scale.
    Cheaper and often faster than LayerNorm as it doesn't mean-center.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ---------------------------------------------------------------------------
# Residual GRU Block — the core repeating unit of the architecture
# ---------------------------------------------------------------------------

class ResidualGRUBlock(nn.Module):
    """
    Residual GRU block.

    Forward path:
        1. Run the input through a 1-layer nn.GRU across the full sequence
        2. Add the GRU output back to the original input (residual skip)

    No normalization layer is used — the GRU's sigmoid/tanh gates
    inherently bound the hidden state, and gradient clipping handles
    the residual growth across depth.
    """

    def __init__(self, dim: int):
        super().__init__()
        # nn.GRU processes the entire (B, T, D) tensor in one cuDNN call,
        # which is orders of magnitude faster than looping with GRUCell.
        self.gru = nn.GRU(dim, dim, num_layers=1, batch_first=True)
        # Orthogonal init on hidden-to-hidden weights keeps eigenvalues
        # close to 1, preventing vanishing/exploding gradients over time.
        nn.init.orthogonal_(self.gru.weight_hh_l0)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (B, T, D)
            Input features for this layer.
        h : Tensor of shape (1, B, D)
            Previous hidden state for this layer's GRU.

        Returns
        -------
        x_out : Tensor (B, T, D) — residual-updated features.
        h_next : Tensor (1, B, D) — final hidden state for TBPTT carry.
        """
        h_out, h_next = self.gru(x, h)          # h_out: (B, T, D), h_next: (1, B, D)
        x_out = x + h_out                       # Residual addition
        return x_out, h_next


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------

class ResidualGRUModel(pl.LightningModule):
    """
    10-layer Residual GRU language model.

    Parameter budget (D=360, V=1024, L=10):
      - Tied Embedding:     V × D         = 368,640
      - Per-block GRU:      L × (6D²+6D)  = 7,797,600
      - RMSNorm:            D             = 360
      - Total:              ≈ 8.16M params → 15.6 MB in bf16 ✓
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 360,
        num_layers: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        bptt_steps: int = 256,
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()  # Logs all __init__ args to W&B / checkpoints

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.bptt_steps = bptt_steps
        self.gradient_clip_val = gradient_clip_val

        # --- Model layers ---

        # Embedding table — also reused as the output projection (tied weights)
        self.embedding = nn.Embedding(vocab_size, dim)

        # Stack of Residual GRU blocks
        self.blocks = nn.ModuleList(
            [ResidualGRUBlock(dim) for _ in range(num_layers)]
        )

        # Final normalization before the tied output head to stabilize logits
        # (since the residual stream variance grows across layers)
        self.norm = RMSNorm(dim)

        # We use manual optimization to implement Truncated BPTT correctly.
        # Lightning's automatic optimization doesn't support the per-chunk
        # backward / step pattern that TBPTT requires.
        self.automatic_optimization = False

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Parameters
        ----------
        x : LongTensor of shape (B, T)
            Token IDs for input sequence.
        h : list of L tensors, each (1, B, D), or None
            Per-layer hidden states carried from the previous TBPTT chunk.
            If None, all hidden states are initialized to zeros.

        Returns
        -------
        logits : Tensor (B, T, V)
            Raw (unnormalized) token predictions.
        h_next : list of L tensors, each (1, B, D)
            Updated per-layer hidden states to pass to the next chunk.
        """
        batch_size, seq_len = x.shape

        # Token embedding: (B, T) → (B, T, D)
        x_t = self.embedding(x)

        # Initialize hidden states to zeros if this is the first chunk
        if h is None:
            # nn.GRU expects hidden shape (num_layers_in_gru=1, B, D)
            h = [
                torch.zeros(1, batch_size, self.dim, device=x.device, dtype=x_t.dtype)
                for _ in range(self.num_layers)
            ]

        h_next = []

        # Process through all layers — each layer sees the FULL sequence
        # (the cuDNN GRU kernel parallelizes across the time dimension)
        for l in range(self.num_layers):
            x_t, h_l_next = self.blocks[l](x_t, h[l])
            h_next.append(h_l_next)

        # Tied output projection
        x_t = self.norm(x_t)
        logits = F.linear(x_t, self.embedding.weight)       # (B, T, V) — tied head

        return logits, h_next

    # -------------------------------------------------------------------
    # Training with manual Truncated BPTT
    # -------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        Implements Truncated Backpropagation Through Time (TBPTT)
        with gradient accumulation across TBPTT chunks.

        Key change vs. naive TBPTT: instead of doing a separate
        optimizer step per chunk (which multiplies the effective LR
        by the number of chunks), we ACCUMULATE gradients across all
        chunks and do a SINGLE optimizer step at the end. This gives
        the model a stable, well-averaged gradient signal.

        Steps:
        1. Zero gradients once at the start.
        2. For each TBPTT chunk:
           a. Detach hidden states (truncate backward graph).
           b. Forward → loss → backward (gradients accumulate).
        3. Clip accumulated gradients.
        4. Single optimizer step + LR scheduler step.
        """
        opt = self.optimizers()
        x, y = batch                     # x, y: (B, seq_len)
        batch_size, seq_len = x.shape

        # Count how many TBPTT chunks we'll process (for loss averaging)
        num_chunks = max(1, (seq_len + self.bptt_steps - 1) // self.bptt_steps)

        hidden_state = None
        losses = []

        # Zero gradients ONCE before accumulating across all TBPTT chunks
        opt.zero_grad()

        for i in range(0, seq_len, self.bptt_steps):
            # Slice out the current TBPTT chunk
            x_chunk = x[:, i : i + self.bptt_steps]
            y_chunk = y[:, i : i + self.bptt_steps]

            # Detach hidden states: keeps the values but severs the
            # computational graph, bounding backprop to this chunk only
            if hidden_state is not None:
                hidden_state = [h.detach() for h in hidden_state]

            # Forward pass for this chunk
            logits, hidden_state = self(x_chunk, hidden_state)

            # Cross-entropy loss, scaled by 1/num_chunks so the accumulated
            # gradient has the same magnitude as a single full-sequence pass
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),  # (B*T_chunk, V)
                y_chunk.reshape(-1),                   # (B*T_chunk,)
            ) / num_chunks

            # Backward — gradients accumulate (no zero_grad here)
            self.manual_backward(loss)

            losses.append(loss.detach() * num_chunks)  # Undo scaling for logging

            # Early exit on NaN — no point continuing a diverged run
            if torch.isnan(loss):
                self.log("train_loss", float("inf"), prog_bar=True)
                self.trainer.should_stop = True
                return torch.tensor(float("inf"))

        # Single clip + step after all chunks have been accumulated
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
        opt.step()

        # Log the average loss across all TBPTT chunks in this batch
        avg_loss = torch.stack(losses).mean()
        self.log("train_loss", avg_loss, prog_bar=True)

        # Log current learning rate for monitoring
        current_lr = opt.param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True)

        return avg_loss

    # -------------------------------------------------------------------
    # Validation (no TBPTT needed — runs under inference_mode)
    # -------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        """
        Standard single-pass validation.  No TBPTT splitting is needed
        because there is no backward pass, so memory is not a concern.
        """
        x, y = batch
        logits, _ = self(x)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            y.reshape(-1),
        )

        # Convert nats → bits: bpb = loss_nats / ln(2)
        val_bpb = loss / math.log(2)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_bpb", val_bpb, sync_dist=True, prog_bar=True)
        return val_bpb

    # -------------------------------------------------------------------
    # Optimizer + LR Schedule
    # -------------------------------------------------------------------

    def configure_optimizers(self):
        """
        AdamW optimizer with a fixed learning rate.

        GRU's sigmoid/tanh gates and orthogonal W_hh init provide inherent
        stability, so warmup is unnecessary. Fixed LR maximizes use of the
        limited ~900-step training budget under the 10-min wall-clock cap.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            # Fused AdamW runs the entire update in a single CUDA kernel
            fused=torch.cuda.is_available(),
        )

        return optimizer

