# H100 GPU Optimization Plan - BeastyBar AlphaZero Training

## Current State: SEVERE UNDERUTILIZATION

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| GPU Utilization (avg) | **10-15%** | 80-90% | 6-8x |
| Memory Used | 11.8 GB / 80 GB (**15%**) | 45-60 GB | 4-5x |
| Power Draw | 160W / 700W (**23%**) | 500-600W | 3-4x |
| Time per iteration | ~2 min | 10-20 sec | 6-12x |
| ETA (3000 iterations) | ~100 hours | ~10-15 hours | 7-10x |

**Root Cause**: CPU-bound MCTS self-play. GPU sits idle 70% of the time waiting for game generation.

---

## OPTIMIZATION TASKS

### TIER 1: Config Changes (No Code Required)

#### Task 1.1: Create Optimized Config File
**File**: `configs/h100_maxout.yaml`

```yaml
# H100 MAXIMUM UTILIZATION CONFIG
# Target: 80%+ GPU utilization, 45%+ memory usage

network_config:
  hidden_dim: 256
  num_heads: 8
  queue_layers: 6
  bar_layers: 2
  hand_layers: 2
  fusion_layers: 4
  dropout: 0.1
  species_embedding_dim: 64
  use_dueling: true
  use_auxiliary_heads: true
  auxiliary_weight: 0.1

network_version: "v2"

# MCTS configuration - INCREASED for quality
num_simulations: 200          # Was 100 - better policy improvement
c_puct: 1.5
dirichlet_alpha: 0.3
dirichlet_epsilon: 0.25
temperature: 1.0
temperature_drop_move: 12
final_temperature: 0.1

# Parallel self-play - MASSIVELY INCREASED
games_per_iteration: 2048     # Was 512 - 4x more data per iteration
parallel_games: 512           # Was 128 - 4x more concurrent games

# Training schedule - ADJUSTED for larger batches
total_iterations: 1500        # Was 3000 - fewer iters, more data each
batch_size: 16384             # Was 4096 - 4x larger batches
epochs_per_iteration: 4

# Optimization
learning_rate: 0.0001
weight_decay: 0.0001
max_grad_norm: 0.5
value_loss_weight: 1.0
auxiliary_loss_weight: 0.1

# Checkpointing and evaluation
checkpoint_frequency: 50      # More frequent due to fewer iterations
eval_frequency: 25
eval_games_per_opponent: 200

# Learning rate schedule
lr_warmup_iterations: 15      # Adjusted for fewer iterations
lr_decay: "cosine"

# Replay buffer - INCREASED
buffer_size: 3000000          # Was 1500000 - 2x for more data
min_buffer_size: 30000        # Was 15000

# Performance - CRITICAL SETTINGS
torch_compile: true
torch_compile_mode: "max-autotune-no-cudagraphs"  # Was reduce-overhead - avoids CUDA graph issues

# Tablebase
tablebase_path: "data/endgame_4card_final.tb"
use_tablebase_values: true
use_tablebase_play: true

# Misc
seed: 42
device: "cuda"
experiment_name: "h100_maxout"
checkpoint_dir: "checkpoints/h100_maxout"

# Evaluation opponents
eval_opponents:
  - random
  - heuristic
  - aggressive
  - defensive
  - mcts-100
  - mcts-500
```

---

### TIER 2: Code Fixes (High Impact)

#### Task 2.1: Enable TF32 Precision
**File**: `scripts/train_alphazero.py`
**Location**: Add at top of main() function, before any model creation

```python
# Add these lines for H100 optimization
import torch
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

**Impact**: ~7x speedup for float32 matrix operations

---

#### Task 2.2: Fix torch.compile Dynamic Masking
**File**: `_02_agents/neural/network_v2.py`
**Location**: Lines 296-335, `AsymmetricTransformerEncoder.forward()`

**Problem**: Dynamic boolean indexing `x[non_empty_mask]` breaks CUDA graphs

**Current Code (lines 310-323)**:
```python
if padding_mask is not None and all_padded.any():
    non_empty_mask = ~all_padded
    x_non_empty = x[non_empty_mask]  # BREAKS CUDA GRAPHS
    padding_non_empty = padding_mask[non_empty_mask]
    x_encoded = self.transformer(x_non_empty, src_key_padding_mask=padding_non_empty)
    # ... scatter back with result[non_empty_mask] = pooled_non_empty
```

**Fixed Code**:
```python
def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if self.use_positional_encoding and self.positional_encoding is not None:
        x = self.positional_encoding(x)

    # Always process full batch - transformer handles masking internally
    x = self.transformer(x, src_key_padding_mask=padding_mask)

    # Vectorized pooling (no conditional indexing - CUDA graph safe)
    if padding_mask is not None:
        inv_mask = ~padding_mask
        inv_mask_expanded = inv_mask.unsqueeze(-1).float()
        # Masked positions contribute 0, then we normalize
        pooled = (x * inv_mask_expanded).sum(dim=1) / inv_mask_expanded.sum(dim=1).clamp(min=1)
    else:
        pooled = x.mean(dim=1)

    return pooled
```

**Impact**: Enables `max-autotune` mode with CUDA graphs (30-40% additional speedup)

---

#### Task 2.3: Fix DuelingHead Dynamic Indexing
**File**: `_02_agents/neural/network_v2.py`
**Location**: Lines 387-393, `DuelingHead.forward()`

**Problem**: `masked_advantage[action_mask == 0] = float("-inf")` breaks CUDA graphs

**Current Code**:
```python
if action_mask is not None:
    masked_advantage = advantage.clone()
    masked_advantage[action_mask == 0] = float("-inf")  # BREAKS CUDA GRAPHS
```

**Fixed Code**:
```python
if action_mask is not None:
    # Use torch.where instead of boolean indexing (CUDA graph safe)
    masked_advantage = torch.where(
        action_mask > 0,
        advantage,
        torch.full_like(advantage, float("-inf"))
    )
```

**Impact**: Part of enabling full CUDA graph capture

---

#### Task 2.4: Unify P0/P1 MCTS Batching
**File**: `_03_training/alphazero_trainer.py`
**Location**: Lines 860-885, `generate_training_data()`

**Problem**: P0 and P1 positions processed in separate forward passes

**Current Code**:
```python
# Process P0 games
if p0_indices:
    p0_results = batch_mcts.search_batch(states=p0_states, perspective=0, ...)

# Process P1 games SEPARATELY
if p1_indices:
    p1_results = batch_mcts.search_batch(states=p1_states, perspective=1, ...)
```

**Fixed Code**:
```python
# Combine all states into single batch
all_states = []
all_perspectives = []
all_indices = []

for idx in p0_indices:
    all_states.append(states_batch[idx])
    all_perspectives.append(0)
    all_indices.append(idx)

for idx in p1_indices:
    all_states.append(states_batch[idx])
    all_perspectives.append(1)
    all_indices.append(idx)

# Single unified forward pass
if all_states:
    all_results = batch_mcts.search_batch_mixed_perspective(
        states=all_states,
        perspectives=all_perspectives,
        ...
    )
    # Distribute results back
    for result, idx in zip(all_results, all_indices):
        # ... handle result for game at idx
```

**Note**: This requires adding `search_batch_mixed_perspective()` to BatchMCTS

**Impact**: 2x speedup (single forward pass instead of two)

---

#### Task 2.5: Increase MCTS Leaf Batch Size
**File**: `_02_agents/mcts/batch_mcts.py`
**Location**: Line 96 (or config)

**Change**: `batch_size: int = 16` → `batch_size: int = 64`

**Impact**: 20-30% faster neural inference during MCTS

---

#### Task 2.6: Add Pinned Memory for Transfers
**File**: `_03_training/alphazero_trainer.py`
**Location**: Lines 1013-1021, tensor creation

**Current Code**:
```python
obs_tensor = torch.from_numpy(observations).to(self._device)
```

**Fixed Code**:
```python
# Use pinned memory for faster CPU->GPU transfer
obs_tensor = torch.from_numpy(observations).pin_memory().to(self._device, non_blocking=True)
```

**Also add at training start**:
```python
torch.cuda.memory._set_allocator_settings(
    "pinned_use_cuda_host_register:True,"
    "pinned_num_register_threads:8"
)
```

**Impact**: 15-25% faster data transfer

---

### TIER 3: Architecture Changes (Maximum Impact)

#### Task 3.1: True Async Game Generation
**File**: `_03_training/alphazero_trainer.py`
**Location**: Lines 1197-1225, main training loop

**Problem**: `future.result()` blocks - no true overlap

**Solution**: Use double-buffering with separate generation/training threads

```python
# Pseudo-code for true async overlap
generation_queue = Queue(maxsize=2)
training_queue = Queue(maxsize=2)

def generation_worker():
    while training:
        examples = self.generate_training_data(...)
        generation_queue.put(examples)

def training_worker():
    while training:
        examples = generation_queue.get()
        self.train_on_buffer()

# Run both concurrently
Thread(target=generation_worker).start()
Thread(target=training_worker).start()
```

**Impact**: GPU never waits for game generation

---

#### Task 3.2: Implement Mixed-Perspective Batch MCTS
**File**: `_02_agents/mcts/batch_mcts.py`
**New Method**: `search_batch_mixed_perspective()`

This allows batching P0 and P1 positions together with perspective-aware value handling.

---

## EXPECTED RESULTS

| Optimization | GPU Util | Memory | Speedup |
|--------------|----------|--------|---------|
| Baseline | 10-15% | 15% | 1x |
| + Config changes (Tier 1) | 40-50% | 31% | 2-3x |
| + TF32 + torch.compile fix (Tier 2.1-2.3) | 55-65% | 35% | 4-5x |
| + Unified batching (Tier 2.4) | 65-75% | 35% | 6-7x |
| + All Tier 2 | 70-80% | 40% | 7-8x |
| + Tier 3 (async) | 80-90% | 45% | 8-10x |

**Final ETA**: ~100 hours → **~10-12 hours**

---

## RUNPOD CONNECTION INFO

```bash
ssh root@213.181.105.224 -p 16728 -i ~/.ssh/id_ed25519
```

**Current training running**: PID may vary, check with `ps aux | grep train`

**Wandb**: https://wandb.ai/diegoships101-none/beastybar

---

## FILES TO MODIFY

1. `configs/h100_maxout.yaml` - CREATE NEW
2. `scripts/train_alphazero.py` - Add TF32 settings
3. `_02_agents/neural/network_v2.py` - Fix dynamic masking (lines 296-335, 387-393)
4. `_03_training/alphazero_trainer.py` - Unify P0/P1 batching, pinned memory
5. `_02_agents/mcts/batch_mcts.py` - Increase batch_size, add mixed-perspective method

---

## VERIFICATION

After changes, monitor with:
```bash
# GPU utilization (should be 70-90%)
nvidia-smi dmon -s pucvmet -d 1

# Power draw (should be 400-600W)
nvidia-smi --query-gpu=power.draw --format=csv -l 1

# Training logs
tail -f /workspace/beastybar/training.log
```
