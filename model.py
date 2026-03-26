"""
model.py — Pre-Norm Residual Deep GRU for Parameter Golf
=========================================================
Implements a 20-layer residual GRU language model that fits within the
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
# Residual GRU Block — the core repeating unit of the architecture
# ---------------------------------------------------------------------------

class ResidualGRUBlock(nn.Module):
    """
    Pre-LayerNorm Residual GRU block.

    Forward path:
        1. LayerNorm the input tensor  (stabilizes inputs to the GRU)
        2. Run through a 1-layer nn.GRU across the full sequence length
        3. Add the GRU output back to the original input (residual skip)

    The residual connection ensures pristine gradient flow through depth,
    allowing the network to scale to 20+ layers without vanishing gradients.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
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
        x_norm = self.norm(x)                   # Pre-norm before recurrence
        h_out, h_next = self.gru(x_norm, h)     # h_out: (B, T, D), h_next: (1, B, D)
        x_out = x + h_out                       # Residual addition
        return x_out, h_next


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------

class ResidualGRUModel(pl.LightningModule):
    """
    20-layer Pre-Norm Residual GRU language model.

    Parameter budget (D=256, V=1024, L=20):
      - Tied Embedding:     V × D         = 262,144
      - Per-block LN:       L × 2D        =  10,240
      - Per-block GRU:      L × (6D²+6D)  = 7,895,040
      - Final LN:           2D            =     512
      - Total:              ≈ 8.17M params → 15.6 MB in bf16 ✓
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 256,
        num_layers: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
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

        # Stack of 20 Pre-Norm Residual GRU blocks
        self.blocks = nn.ModuleList(
            [ResidualGRUBlock(dim) for _ in range(num_layers)]
        )

        # Final LayerNorm before projecting to vocabulary logits
        self.final_norm = nn.LayerNorm(dim)

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

        # Process through all 20 layers — each layer sees the FULL sequence
        # (the cuDNN GRU kernel parallelizes across the time dimension)
        for l in range(self.num_layers):
            x_t, h_l_next = self.blocks[l](x_t, h[l])
            h_next.append(h_l_next)

        # Final normalization + tied output projection
        out = self.final_norm(x_t)                          # (B, T, D)
        logits = F.linear(out, self.embedding.weight)       # (B, T, V) — tied head

        return logits, h_next

    # -------------------------------------------------------------------
    # Training with manual Truncated BPTT
    # -------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        Implements Truncated Backpropagation Through Time (TBPTT):

        1. Split the full sequence into chunks of `bptt_steps` tokens.
        2. For each chunk:
           a. Detach hidden states from the previous chunk's graph
              (truncates the backward pass to bptt_steps tokens).
           b. Forward through the model to get logits.
           c. Compute cross-entropy loss against shifted targets.
           d. Backward + clip gradients + optimizer step.
        3. Hidden state *values* are preserved across chunks (only the
           gradient graph is truncated), so the GRU has full recurrent
           context up to the current position.
        """
        opt = self.optimizers()
        x, y = batch                     # x, y: (B, seq_len)
        batch_size, seq_len = x.shape

        hidden_state = None
        losses = []

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

            # Cross-entropy loss (flatten batch & time dimensions)
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),  # (B*T_chunk, V)
                y_chunk.reshape(-1),                   # (B*T_chunk,)
            )

            # Manual optimization: zero → backward → clip → step
            opt.zero_grad()
            self.manual_backward(loss)
            # Direct PyTorch grad clipping (Lightning's clip_gradients is
            # incompatible with fused AdamW under AMP)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
            opt.step()

            losses.append(loss.detach())

        # Log the average loss across all TBPTT chunks in this batch
        avg_loss = torch.stack(losses).mean()
        self.log("train_loss", avg_loss, prog_bar=True)
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
    # Optimizer
    # -------------------------------------------------------------------

    def configure_optimizers(self):
        """AdamW with optional fused CUDA implementation for speed."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            # Fused AdamW runs the entire update in a single CUDA kernel,
            # avoiding per-parameter Python overhead.
            fused=torch.cuda.is_available(),
        )
        return optimizer
