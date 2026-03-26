import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ResidualGRUBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # We use a 1-layer sequence GRU instead of GRUCell to utilize highly optimized cuDNN kernels over the entire sequence at once.
        self.gru = nn.GRU(dim, dim, num_layers=1, batch_first=True)
        nn.init.orthogonal_(self.gru.weight_hh_l0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)
        h_out, h_next = self.gru(x_norm, h)
        x_out = x + h_out
        return x_out, h_next

class ResidualGRUModel(pl.LightningModule):
    def __init__(self, vocab_size: int = 1024, dim: int = 256, num_layers: int = 20, learning_rate: float = 1e-3, weight_decay: float = 0.0, bptt_steps: int = 256, gradient_clip_val: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.bptt_steps = bptt_steps
        self.gradient_clip_val = gradient_clip_val
        
        # Tie embeddings
        self.embedding = nn.Embedding(vocab_size, dim)
        
        self.blocks = nn.ModuleList([
            ResidualGRUBlock(dim) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(dim)

        self.automatic_optimization = False # We manage TBPTT manually if using modern lightning

    def forward(self, x: torch.Tensor, h: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        x_t = self.embedding(x) # (B, T, D)
        
        if h is None:
            # Sequence GRU expects hidden states of shape (num_layers=1, batch, dim)
            h = [torch.zeros(1, batch_size, self.dim, device=x.device, dtype=x_t.dtype) for _ in range(self.num_layers)]

        h_next = []
        
        # Iterate over layers (processing the complete sequence dimension simultaneously per block)
        for l in range(self.num_layers):
            x_t, h_l_next = self.blocks[l](x_t, h[l])
            h_next.append(h_l_next)

        out = self.final_norm(x_t) # (B, T, D)
        
        # Tied lm_head
        logits = F.linear(out, self.embedding.weight)
        
        return logits, h_next

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        batch_size, seq_len = x.shape
        
        # TBPTT Implementation
        hidden_state = None
        losses = []
        
        for i in range(0, seq_len, self.bptt_steps):
            x_chunk = x[:, i:i+self.bptt_steps]
            y_chunk = y[:, i:i+self.bptt_steps]
            
            # Keep gradient across sequence chunks but detach the input hidden state
            if hidden_state is not None:
                hidden_state = [h.detach() for h in hidden_state]
            
            logits, hidden_state = self(x_chunk, hidden_state)
            
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), y_chunk.reshape(-1))
            
            opt.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
            opt.step()
            
            losses.append(loss.detach())
        
        avg_loss = torch.stack(losses).mean()
        self.log('train_loss', avg_loss, prog_bar=True)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1))
        
        # log val_bpb
        val_bpb = loss / math.log(2)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_bpb', val_bpb, sync_dist=True, prog_bar=True)
        return val_bpb

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=True if torch.cuda.is_available() else False
        )
        return optimizer
