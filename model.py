"""
model.py — Pre-Norm Residual Deep GRU for Parameter Golf
=========================================================
Implements a configurable residual GRU language model that fits within the
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

import numpy as np
import sentencepiece as spm
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
# Token-to-Byte Lookup Tables (from train_gpt.py)
# ---------------------------------------------------------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return base_bytes_np, has_leading_space_np, is_boundary_token_np


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

    def __init__(self, dim: int, use_post_rmsnorm: bool = False):
        super().__init__()
        self.use_post_rmsnorm = use_post_rmsnorm
        # nn.GRU processes the entire (B, T, D) tensor in one cuDNN call,
        # which is orders of magnitude faster than looping with GRUCell.
        self.gru = nn.GRU(dim, dim, num_layers=1, batch_first=True)
        # Orthogonal init on hidden-to-hidden weights keeps eigenvalues
        # close to 1, preventing vanishing/exploding gradients over time.
        nn.init.orthogonal_(self.gru.weight_hh_l0)

        # Post-norm layer applied to the residual stream 'x'
        if self.use_post_rmsnorm:
            self.norm = RMSNorm(dim)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, use_qat: bool = False
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
        if use_qat:
            from torch.func import functional_call
            params = dict(self.gru.named_parameters())
            qat_params = {}
            for k, v in params.items():
                q_v = v.to(torch.float8_e5m2).to(v.dtype)
                qat_params[k] = v + (q_v - v).detach()
            h_out, h_next = functional_call(self.gru, qat_params, (x, h))
        else:
            h_out, h_next = self.gru(x, h)          # h_out: (B, T, D), h_next: (1, B, D)
        
        x_bar = self.norm(x) if self.use_post_rmsnorm else x
        x_out = x_bar + h_out
        return x_out, h_next


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------

class ResidualGRUModel(pl.LightningModule):
    """
    Residual GRU language model.

    Parameter budget example for medium architecture (D=360, V=1024, L=10):
      - Tied Embedding:     V × D         = 368,640
      - Per-block GRU:      L × (6D²+6D)  = 7,797,600
      - Per-block norm:     L × D         = 3,600
      - Final RMSNorm:      D             = 360
      - Total:              ≈ 8.16M params → 15.6 MB in bf16 ✓
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 360,
        num_layers: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        bptt_steps: int = 256,
        gradient_clip_val: float = 1.0,
        use_qat: bool = False,
        use_post_rmsnorm: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()  # Logs all __init__ args to W&B / checkpoints

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.bptt_steps = bptt_steps
        self.gradient_clip_val = gradient_clip_val
        self.use_qat = use_qat
        self.use_post_rmsnorm = use_post_rmsnorm

        # --- Tokenizer & Metrics Computations ---
        # Build the exact token-to-byte lookup tables used for official val_bpb
        sp = spm.SentencePieceProcessor("./data/tokenizers/fineweb_1024_bpe.model")
        base_bytes, has_leading_space, is_boundary_token = build_sentencepiece_luts(
            sp, vocab_size
        )
        # Register as buffers so model.to(device) moves them automatically
        self.register_buffer("base_bytes_lut", torch.tensor(base_bytes, dtype=torch.int16))
        self.register_buffer("has_leading_space_lut", torch.tensor(has_leading_space, dtype=torch.bool))
        self.register_buffer("is_boundary_token_lut", torch.tensor(is_boundary_token, dtype=torch.bool))

        # --- Model layers ---

        # Embedding table — also reused as the output projection (tied weights)
        self.embedding = nn.Embedding(vocab_size, dim)

        # Stack of Residual GRU blocks
        self.blocks = nn.ModuleList(
            [ResidualGRUBlock(dim, use_post_rmsnorm) for _ in range(num_layers)]
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

        embed_weight = self.embedding.weight
        if self.use_qat:
            q_w = embed_weight.to(torch.float8_e5m2).to(embed_weight.dtype)
            embed_weight = embed_weight + (q_w - embed_weight).detach()

        # Token embedding: (B, T) → (B, T, D)
        x_t = F.embedding(x, embed_weight)

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
            x_t, h_l_next = self.blocks[l](x_t, h[l], self.use_qat)
            h_next.append(h_l_next)

        # Tied output projection
        x_t = self.norm(x_t)
        logits = F.linear(x_t, embed_weight)       # (B, T, V) — tied head

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

    def on_validation_epoch_start(self):
        self.val_loss_sum = 0.0
        self.val_token_count = 0.0
        self.val_byte_count = 0.0

    def validation_step(self, batch, batch_idx):
        """
        Standard single-pass validation. Computes exact bits-per-byte exactly
        as the official train_gpt.py script does.
        """
        x, y = batch
        logits, _ = self(x)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            y.reshape(-1),
            reduction="mean",
        )

        batch_token_count = float(y.numel())
        self.val_loss_sum += float(loss.item()) * batch_token_count
        self.val_token_count += batch_token_count

        # Exact token-to-byte count logic from `train_gpt.py`
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        token_bytes = self.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        token_bytes += (self.has_leading_space_lut[tgt_ids] & ~self.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
        self.val_byte_count += float(token_bytes.to(torch.float64).sum().item())

    def on_validation_epoch_end(self):
        # Sync metrics across GPUs if using DDP (single GPU in param golf normally)
        if self.trainer.world_size > 1:
            metrics = torch.tensor(
                [self.val_loss_sum, self.val_token_count, self.val_byte_count],
                device=self.device,
            )
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            val_loss_sum, val_token_count, val_byte_count = metrics.tolist()
        else:
            val_loss_sum = self.val_loss_sum
            val_token_count = self.val_token_count
            val_byte_count = self.val_byte_count

        val_loss = val_loss_sum / val_token_count
        bits_per_token = val_loss / math.log(2.0)
        tokens_per_byte = val_token_count / val_byte_count
        val_bpb = bits_per_token * tokens_per_byte

        self.log("val_loss", val_loss, sync_dist=True)
        self.log("val_bpb", val_bpb, sync_dist=True, prog_bar=True)

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
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            # Fused AdamW runs the entire update in a single CUDA kernel
            fused=torch.cuda.is_available(),
        )

        return optimizer

