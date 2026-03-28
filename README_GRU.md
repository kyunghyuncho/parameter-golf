# Pre-Norm Residual Deep GRU for Parameter Golf

This directory implements an ultra-deep Post-RMSNorm Residual GRU (with configurable shallow, medium, and deep variants) trained within the strict limits of the [openai/parameter-golf](https://github.com/openai/parameter-golf) competition. 

## Architectural Philosophy

Standard Transformers pay an $O(N^2)$ quadratic attention tax on context length and require heavy MLP blocks to stabilize feature representations. In contrast, this implementation relies entirely on a Recurrent Neural Network (RNN) structure, composed of up to 20 dense Gated Recurrent Unit (GRU) layers. 

To overcome the traditional vanishing gradient problems that limit standard RNN depth, we implement a **Pre-LayerNorm Residual Design**, heavily inspired by modern Transformer setups. 

### The Residual GRU Block
For any layer $l$ and timestep $t$, the input features $x_t^{(l)}$ are processed via:
1. **Recurrence:** $h_t^{(l)} = \text{GRUCell}^{(l)}(x_t^{(l)}, h_{t-1}^{(l)})$
2. **Post-Normalization (Optional):** $\bar{x}_t^{(l)} = \text{RMSNorm}(x_t^{(l)})$ (controlled via `--use_post_rmsnorm`)
3. **Residual Addition:** $x_t^{(l+1)} = \bar{x}_t^{(l)} + h_t^{(l)}$

By retaining a pristine identity shortcut (before the RMSNorm rescale), the network depth can be scaled up to 20 layers without shattering gradients. 
We also apply **Orthogonal Initialization** to the GRU hidden-to-hidden weights to force internal transition eigenvalues close to 1, acting as a secondary gradient stabilization measure.

### Parameter Budget Constraint ($16.00$ MB)
The competition strictly enforces a 16.00 MB size limit on the final `submission_model.pt`. To fully saturate this limit in `bfloat16` precision (2 bytes per parameter), we are allowed an absolute maximum of $8,388,608$ parameters.

We configure the depth ($L$) and width ($D$) to fully map the parameter space depending on the chosen architecture:
- `shallow`: 5 layers, $D = 504$ ($\approx 8.15$M parameters)
- `medium`: 10 layers, $D = 360$ ($\approx 8.17$M parameters)
- `deep`: 20 layers, $D = 256$ ($\approx 8.16$M parameters)

**Parameter Breakdown Example (`medium` variant):**
- Tied Embeddings and Output Head: $1024 \times 360 = 368,640$
- 10 GRU Cells ($6 \times D^2 + 6 \times D$): $10 \times 779,760 = 7,797,600$
- 10 Post-RMSNorms: $10 \times 360 = 3,600$
- Final Output RMSNorm: $360$
- **Total:** $\approx 8.16 \text{M parameters}$ (weighing exactly $15.6 \text{ MB}$, perfectly compliant!)


## Implementation Details

### Data Loading (`data_module.py`)
To ensure zero CPU I/O bottleneck during the frantic 10-minute training window, we utilize a PyTorch `IterableDataset` feeding a standard PyTorch-Lightning `LightningDataModule`. 
- The tokenizer maps texts to sequences of native `uint16` tokens natively exported by the competition data preprocessing pipeline.
- We utilize `numpy.fromfile` with zero abstractions to memmap the `.bin` shards. We skip the 256-integer challenge header array dynamically on load.

### Continuous Truncated BPTT (`model.py`)
Training an RNN across 4096-length context windows naively overflows GPU VRAM. Standard Lightning paradigms don't support passing recurrent state automatically anymore. To achieve seamless recurrence, we:
1. Disable `automatic_optimization=False`.
2. Process the 2048/4096 context window in successive **BPTT chunks** of length `bptt_steps` (e.g., 256).
3. Between chunks, detach the resulting hidden states from the computational graph, but seamlessly pass their values into the next temporal chunk.
4. Manually trigger backwards computation and `torch.nn.utils.clip_grad_norm_` explicitly at every chunk segment! 

### Custom Artifact Export Protocol (`train.py`)
Because our artifact must be compliant at the moment of evaluation, we use a custom Lightning Callback (`ArtifactExportCallback`). On `on_train_end`:
1. Grabs `model.state_dict()`.
2. Strips away the `AdamW` momentum caches (Lightning naturally omits optimizer payloads from the raw neural network dictionary).
3. Casts every tensor explicitly into pure CPU `torch.bfloat16`. 
4. Asserts that the final serialized object `< 16,777,216` bytes on disk!

## Launching a Training Sweep

Since training lasts precisely 9 minutes and 45 seconds, evaluating hyperparameters is fast. 
We leverage W&B's Bayesian Optimization sweeps defined in `sweep.yaml` to search for optimal learning rates, batch sizes, and gradient clip boundaries. We also tune boolean flags like `use_post_rmsnorm` and `use_qat` (the latter exclusively for FP8 models).

1. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```
2. **Authenticate with Weights & Biases:**
   ```bash
   wandb login
   ```
3. **Initialize the Sweep Controller:**
   ```bash
   wandb sweep sweep.yaml
   ```
   *(This prints a `<sweep_id>` to your console).*
4. **Deploy a local L4 GPU Sweep Agent:**
   ```bash
   wandb agent <username>/parameter-golf/<sweep_id>
   ```

*(Wait 10 minutes, and the orchestrator outputs your best Loss and Artifact to WandB!)*

## Attribution & License

This project is a fork of the OpenAI Parameter Golf challenge. 
Original Parameter Golf codebase (data processing, evaluation loops, `train_gpt.py`) is Copyright (c) 2026 OpenAI. 

The Custom Residual GRU model architecture (`model.py`), continuous TBPTT training loop (`train.py`), hyperparameter sweeps, and `README_GRU.md` are Copyright (c) 2026 Kyunghyun Cho, licensed under the MIT License.
