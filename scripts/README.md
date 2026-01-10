# Beasty Bar CLI Scripts

Command-line tools for training, evaluating, and playing with Beasty Bar AI agents.

## Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | PPO training |
| `train_mcts.py` | AlphaZero-style MCTS training |
| `train_h200.py` | H200 GPU optimized training |
| `evaluate.py` | Model evaluation vs baselines |
| `play.py` | Interactive play against AI |
| `benchmark_cython.py` | Cython performance testing |
| `build_cython.sh` | Build Cython extensions |

## Installation

```bash
# Install dependencies
uv sync

# Optional: Build Cython for 200x speedup
bash scripts/build_cython.sh
```

## Training

### PPO Training (Fast)

```bash
# Basic training
uv run scripts/train.py --config configs/default.yaml

# With W&B tracking
uv run scripts/train.py --config configs/default.yaml --tracker wandb

# Override parameters
uv run scripts/train.py --config configs/default.yaml --lr 0.0001 --iterations 500

# Resume from checkpoint
uv run scripts/train.py --resume checkpoints/iter_000500.pt
```

### MCTS Training (Recommended)

```bash
# Basic MCTS training
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml

# With W&B tracking
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml --wandb

# Resume from checkpoint
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml --checkpoint checkpoints/iter_000100.pt
```

### Quick Test

```bash
# PPO quick test
uv run scripts/train.py --iterations 10 --games-per-iter 64 --device cpu

# MCTS quick test (create minimal config)
cat > /tmp/test_mcts.yaml << EOF
network_config:
  hidden_dim: 64
  num_layers: 1
mcts_simulations: 10
total_iterations: 5
games_per_iteration: 8
device: "cpu"
EOF
uv run scripts/train_mcts.py --config /tmp/test_mcts.yaml
```

## Evaluation

```bash
# Evaluate against single opponent
uv run scripts/evaluate.py --model checkpoints/model.pt --opponent heuristic

# Evaluate against multiple opponents
uv run scripts/evaluate.py --model checkpoints/model.pt \
    --opponents random,heuristic,mcts-500 \
    --games 200

# Play as both sides for fairness
uv run scripts/evaluate.py --model checkpoints/model.pt \
    --opponent mcts-500 --games 100 --both-sides

# Save results
uv run scripts/evaluate.py --model checkpoints/model.pt \
    --opponents random,heuristic \
    --output results.json
```

### Available Opponents

- `random` - Uniform random baseline
- `heuristic` - Rule-based with material evaluation
- `mcts-100` - MCTS with 100 simulations
- `mcts-500` - MCTS with 500 simulations
- `mcts-1000` - MCTS with 1000 simulations

## Interactive Play

```bash
# Play against neural agent
uv run scripts/play.py --model checkpoints/model.pt

# You go first
uv run scripts/play.py --model checkpoints/model.pt --you-start

# Watch AI vs AI
uv run scripts/play.py --model checkpoints/model.pt --opponent heuristic

# Multiple games
uv run scripts/play.py --model checkpoints/model.pt --games 5
```

## Cython Benchmarking

```bash
# Build Cython extensions
bash scripts/build_cython.sh

# Run benchmarks
uv run scripts/benchmark_cython.py
```

## Configuration Files

Located in `configs/`:

| Config | Use Case |
|--------|----------|
| `default.yaml` | Standard PPO training |
| `fast.yaml` | Quick testing (small network) |
| `h100_mcts.yaml` | H100 GPU MCTS training |
| `h200_optimized_v2.yaml` | H200 GPU optimization |
| `ppo_warmstart.yaml` | PPO with MCTS warmstart |

### Configuration Format

```yaml
# Network architecture
network_config:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.1
  species_embedding_dim: 64

# MCTS settings (for MCTS training)
mcts_config:
  num_simulations: 200
  c_puct: 2.0
  batch_size: 32
  virtual_loss: 3.0

# PPO settings (for PPO training)
ppo_config:
  learning_rate: 0.0003
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  gamma: 0.99
  gae_lambda: 0.95

# Training schedule
total_iterations: 500
games_per_iteration: 128
checkpoint_frequency: 50
eval_frequency: 10

# Misc
seed: 42
device: "auto"
experiment_name: "beastybar_training"
checkpoint_dir: "checkpoints"
```

## Checkpoints

Saved in `checkpoints/` as PyTorch `.pt` files containing:
- Model weights
- Optimizer state
- Training iteration
- Configuration
- Training metrics

Naming convention:
- `iter_NNNNNN.pt` - Checkpoint at iteration NNNNNN
- `best_model.pt` - Best by evaluation metric
- `final.pt` - Final checkpoint

## Experiment Tracking

### Console (default)
```bash
uv run scripts/train.py --tracker console
```

### Weights & Biases
```bash
uv run scripts/train.py --tracker wandb --wandb-project beastybar
```

### TensorBoard
```bash
uv run scripts/train.py --tracker tensorboard
# View: tensorboard --logdir runs
```

## Troubleshooting

### PyTorch not found
```bash
pip install torch
```

### CUDA out of memory
Reduce `minibatch_size` or `games_per_iteration` in config.

### wandb authentication
```bash
wandb login
```

### Import errors
Run from project root directory.

## See Also

- `configs/` - Training configuration files
- `_03_training/MCTS_TRAINING.md` - MCTS training details
- `_02_agents/mcts/README.md` - MCTS implementation
- `_02_agents/mcts/BATCH_MCTS.md` - BatchMCTS details
