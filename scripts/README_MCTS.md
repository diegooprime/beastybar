# MCTS Training Script

## Overview

`train_mcts.py` provides a command-line interface for training Beasty Bar agents using AlphaZero-style MCTS self-play. This approach differs from PPO training:

- **MCTS Search**: Uses Monte Carlo Tree Search to improve policies during self-play
- **Policy Targets**: Network learns to match MCTS visit distributions (not single actions)
- **Direct Outcomes**: Uses game outcomes directly (no GAE or advantage estimation)
- **Stability**: More stable and sample-efficient for two-player zero-sum games

## Quick Start

### Basic Training

```bash
# Train with H100 configuration
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml

# Enable Weights & Biases tracking
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml --wandb

# Custom W&B project
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml --wandb --wandb-project my-project
```

### Resume Training

```bash
# Resume from checkpoint
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --checkpoint checkpoints/h100_mcts_v1/iter_000100.pt
```

### Override Configuration

```bash
# Override experiment name and seed
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --experiment-name mcts_test_run \
  --seed 999
```

## Command-Line Arguments

### Required
- `--config PATH`: Path to YAML configuration file

### Optional
- `--checkpoint PATH`: Resume training from checkpoint (.pt file)
- `--wandb`: Enable Weights & Biases tracking (default: console)
- `--wandb-project NAME`: W&B project name (default: beastybar)
- `--experiment-name NAME`: Override experiment name from config
- `--seed INT`: Override random seed from config
- `--verbose`: Enable verbose logging (DEBUG level)

## Configuration Files

### YAML Structure

```yaml
# Network architecture
network_config:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.1
  species_embedding_dim: 64

# MCTS self-play parameters
mcts_simulations: 200      # Simulations per move
c_puct: 1.5                # Exploration constant
temperature: 1.0           # Action selection temperature

# Training schedule
total_iterations: 200
games_per_iteration: 256
batch_size: 256
epochs_per_iteration: 10

# Optimization
learning_rate: 0.001
weight_decay: 0.0001
value_loss_weight: 1.0
policy_loss_weight: 1.0
entropy_bonus_weight: 0.01

# Checkpointing
checkpoint_frequency: 10
eval_frequency: 5

# Learning rate schedule
lr_warmup_iterations: 10
lr_decay: "cosine"         # "linear", "cosine", or "none"

# Misc
seed: 42
device: "cuda"             # "cpu", "cuda", "mps", or "auto"
experiment_name: "h100_mcts_v1"
checkpoint_dir: "checkpoints"
log_frequency: 1
```

### Available Configs

- `configs/h100_mcts.yaml`: Production config for H100 GPU (80GB VRAM)

## MCTS Parameters

### Key Parameters

- **`mcts_simulations`**: Number of MCTS simulations per move
  - Higher = stronger play, slower generation
  - Typical: 200-800 for strong play
  - Test: 10-50 for fast iteration

- **`c_puct`**: PUCT exploration constant
  - Higher = more exploration during search
  - Typical: 1.0-2.0
  - AlphaZero uses 1.0-1.5

- **`temperature`**: Action selection temperature
  - 1.0 = stochastic (proportional to visit counts)
  - 0.0 = deterministic (argmax)
  - Typical: Start at 1.0, decrease over time

### Loss Weights

- **`policy_loss_weight`**: Weight for policy cross-entropy loss
- **`value_loss_weight`**: Weight for value MSE loss
- **`entropy_bonus_weight`**: Weight for entropy bonus (exploration)

Balance these to control training dynamics:
- Policy loss: Teaches network to match MCTS policies
- Value loss: Teaches network to predict game outcomes
- Entropy bonus: Prevents premature convergence

## Training Output

### Console Logs

```
2026-01-07 16:08:35,878 - __main__ - INFO - MCTS Training Configuration:
2026-01-07 16:08:35,878 - __main__ - INFO -   Total iterations: 200
2026-01-07 16:08:35,878 - __main__ - INFO -   Games per iteration: 256
2026-01-07 16:08:35,878 - __main__ - INFO -   MCTS simulations: 200
2026-01-07 16:08:35,878 - __main__ - INFO -   Learning rate: 0.001
2026-01-07 16:08:35,878 - __main__ - INFO -   Device: cuda
...
2026-01-07 16:08:38,309 - _03_training.mcts_trainer - INFO - Iteration 0/200 | Loss: 2.3456 | Policy: 1.2345 | Value: 0.5678 | LR: 1.00e-04 | ETA: 45.3min
```

### Checkpoints

Saved to `{checkpoint_dir}/{experiment_name}/`:
- `iter_000000.pt`, `iter_000010.pt`, ... - Periodic checkpoints
- `final.pt` - Final checkpoint after training completes
- `iter_000000.json`, ... - Configuration snapshots

### Evaluation

Automatic evaluation against baseline opponents:
- Random agent
- Heuristic agent

Metrics logged:
- Win rate
- Average point margin
- Games played
- Confidence intervals

## W&B Integration

When using `--wandb`, logs include:
- Training metrics (loss, learning rate, entropy)
- Self-play statistics (game length, win rates)
- Evaluation results (win rate vs baselines)
- Checkpoints as artifacts
- Hyperparameters

View at: https://wandb.ai/{username}/{project}/{run}

## Example Workflows

### Quick Test Run

```bash
# Create minimal test config
cat > /tmp/test_mcts.yaml << EOF
network_config:
  hidden_dim: 64
  num_layers: 1
mcts_simulations: 10
total_iterations: 5
games_per_iteration: 8
batch_size: 16
device: "cpu"
experiment_name: "test_run"
EOF

# Run test
uv run scripts/train_mcts.py --config /tmp/test_mcts.yaml --verbose
```

### Production Training on H100

```bash
# Launch training with W&B
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --wandb \
  --wandb-project beastybar-h100 \
  --experiment-name h100_mcts_run_001

# Monitor with W&B dashboard
# Checkpoints saved to: checkpoints/h100_mcts_run_001/
```

### Resume After Interruption

```bash
# Find latest checkpoint
ls -lt checkpoints/h100_mcts_v1/

# Resume training
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --checkpoint checkpoints/h100_mcts_v1/iter_000150.pt \
  --wandb
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `games_per_iteration`
- Reduce network size (`hidden_dim`, `num_layers`)

### Training Too Slow
- Reduce `mcts_simulations` (faster, weaker policies)
- Reduce `games_per_iteration`
- Increase parallelism (if using multiprocessing)

### Loss is NaN
- Check learning rate (try reducing)
- Check gradient clipping
- Verify data quality (inspect trajectories)

### Poor Performance
- Increase `mcts_simulations` (stronger policy targets)
- Train for more iterations
- Tune `c_puct` exploration constant
- Check evaluation metrics for overfitting

## See Also

- `_03_training/mcts_trainer.py` - MCTSTrainer implementation
- `_03_training/mcts_self_play.py` - MCTS self-play generation
- `_03_training/MCTS_TRAINING.md` - MCTS training methodology
- `scripts/train.py` - PPO training script (alternative approach)
