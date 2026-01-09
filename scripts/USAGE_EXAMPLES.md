# Training Script Usage Examples

This document provides practical examples for both PPO and MCTS training scripts.

## Table of Contents

1. [PPO Training (`train.py`)](#ppo-training)
2. [MCTS Training (`train_mcts.py`)](#mcts-training)
3. [Comparing Approaches](#comparing-approaches)
4. [Production Workflows](#production-workflows)

---

## PPO Training

### Basic Usage

```bash
# Train with default settings
uv run scripts/train.py

# Train with custom config
uv run scripts/train.py --config configs/h100_scaled.yaml

# Enable W&B tracking
uv run scripts/train.py --config configs/h100_scaled.yaml --tracker wandb
```

### Override Parameters

```bash
# Override learning rate and iterations
uv run scripts/train.py \
  --config configs/h100_scaled.yaml \
  --lr 0.0001 \
  --iterations 500

# Override network architecture
uv run scripts/train.py \
  --config configs/h100_scaled.yaml \
  --hidden-dim 256 \
  --num-layers 4
```

### Resume Training

```bash
# Resume from checkpoint
uv run scripts/train.py \
  --resume checkpoints/beastybar_neural/iter_000250.pt \
  --tracker wandb
```

### Quick Test

```bash
# Small test run (10 iterations)
uv run scripts/train.py \
  --iterations 10 \
  --games-per-iter 64 \
  --lr 0.001 \
  --device cpu \
  --experiment-name test_ppo
```

---

## MCTS Training

### Basic Usage

```bash
# Train with H100 config (required --config flag)
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml

# Enable W&B tracking
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml --wandb

# Custom W&B project
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --wandb \
  --wandb-project beastybar-experiments
```

### Override Parameters

```bash
# Override experiment name and seed
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --experiment-name mcts_exp_001 \
  --seed 999
```

### Resume Training

```bash
# Resume from checkpoint
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --checkpoint checkpoints/h100_mcts_v1/iter_000100.pt \
  --wandb
```

### Quick Test

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
checkpoint_frequency: 2
eval_frequency: 2
device: "cpu"
experiment_name: "mcts_test"
checkpoint_dir: "/tmp/checkpoints"
EOF

# Run quick test
uv run scripts/train_mcts.py --config /tmp/test_mcts.yaml --verbose
```

---

## Comparing Approaches

### PPO (Proximal Policy Optimization)

**Best for:**
- Fast iteration and experimentation
- CPU-friendly training
- Standard RL benchmarks
- Single-player or cooperative games

**Characteristics:**
- Faster game generation (no MCTS overhead)
- Uses GAE for advantage estimation
- Clipped surrogate objective for stability
- More hyperparameters to tune

**Example:**
```bash
uv run scripts/train.py \
  --config configs/h100_scaled.yaml \
  --iterations 500 \
  --games-per-iter 2048 \
  --tracker wandb
```

### MCTS (AlphaZero-style)

**Best for:**
- Two-player zero-sum games
- High-quality policy targets
- Sample efficiency
- Competitive play strength

**Characteristics:**
- Slower game generation (MCTS search)
- More stable training
- Better final performance
- Simpler loss function

**Example:**
```bash
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --wandb \
  --experiment-name alphazero_v1
```

### When to Use Which?

| Scenario | Recommended Approach |
|----------|---------------------|
| Quick prototyping | PPO |
| Production training | MCTS |
| Limited compute | PPO |
| GPU cluster available | MCTS |
| Need interpretability | PPO |
| Maximum performance | MCTS |
| CPU-only training | PPO |

---

## Production Workflows

### H100 GPU Training (MCTS)

```bash
# SSH into H100 instance
ssh user@h100-instance

# Clone repository
git clone https://github.com/user/beastybar.git
cd beastybar

# Install dependencies
uv sync

# Start training with W&B
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --wandb \
  --wandb-project beastybar-production \
  --experiment-name h100_mcts_$(date +%Y%m%d_%H%M%S)

# Monitor training
# - W&B dashboard: https://wandb.ai/user/beastybar-production
# - Checkpoints: checkpoints/h100_mcts_*/
```

### Multi-Run Experimentation

```bash
# Experiment 1: Low learning rate
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --wandb \
  --experiment-name mcts_lr_low \
  --seed 42 &

# Experiment 2: High c_puct (more exploration)
# Edit config to increase c_puct to 2.0
uv run scripts/train_mcts.py \
  --config configs/h100_mcts_high_explore.yaml \
  --wandb \
  --experiment-name mcts_explore_high \
  --seed 42 &

# Experiment 3: Baseline PPO
uv run scripts/train.py \
  --config configs/h100_scaled.yaml \
  --tracker wandb \
  --experiment-name ppo_baseline \
  --seed 42 &

# Wait for all to complete
wait
```

### Checkpoint Management

```bash
# List checkpoints
ls -lh checkpoints/h100_mcts_v1/

# Resume interrupted training
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --checkpoint checkpoints/h100_mcts_v1/iter_000150.pt \
  --wandb

# Extract best checkpoint (by evaluation)
# Check W&B for best iteration, then:
cp checkpoints/h100_mcts_v1/iter_000180.pt models/best_mcts_model.pt
```

### Continuous Training Pipeline

```bash
#!/bin/bash
# train_pipeline.sh - Continuous training with checkpointing

set -e

CONFIG="configs/h100_mcts.yaml"
CHECKPOINT_DIR="checkpoints/production_run"
EXPERIMENT="mcts_production"

# Find latest checkpoint
LATEST=$(ls -t $CHECKPOINT_DIR/iter_*.pt 2>/dev/null | head -1 || echo "")

if [ -n "$LATEST" ]; then
  echo "Resuming from: $LATEST"
  uv run scripts/train_mcts.py \
    --config $CONFIG \
    --checkpoint "$LATEST" \
    --wandb \
    --experiment-name $EXPERIMENT
else
  echo "Starting fresh training"
  uv run scripts/train_mcts.py \
    --config $CONFIG \
    --wandb \
    --experiment-name $EXPERIMENT
fi
```

### Evaluation Workflow

```bash
# Train model
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --wandb \
  --experiment-name mcts_eval_test

# Evaluation happens automatically during training
# Check logs or W&B for:
# - eval/random/win_rate
# - eval/heuristic/win_rate

# For manual evaluation of saved checkpoint:
# (Create evaluation script if needed)
python scripts/evaluate.py \
  --checkpoint checkpoints/mcts_eval_test/final.pt \
  --opponents random,heuristic,mcts \
  --games 100
```

### Debugging and Monitoring

```bash
# Verbose logging
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --verbose

# Monitor GPU usage (separate terminal)
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/training.log

# Real-time metrics (if using console tracker)
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml 2>&1 | tee training_output.log
```

---

## Configuration Tips

### For Fast Iteration
```yaml
total_iterations: 10
games_per_iteration: 16
mcts_simulations: 20
checkpoint_frequency: 5
eval_frequency: 5
```

### For Production Quality
```yaml
total_iterations: 500
games_per_iteration: 256
mcts_simulations: 400
checkpoint_frequency: 25
eval_frequency: 10
```

### For Maximum Performance
```yaml
total_iterations: 1000
games_per_iteration: 512
mcts_simulations: 800
checkpoint_frequency: 50
eval_frequency: 20
network_config:
  hidden_dim: 512
  num_layers: 6
```

---

## Common Issues

### Out of Memory

**Problem:** CUDA out of memory error

**Solutions:**
```bash
# Reduce batch size in config
# Reduce games_per_iteration
# Reduce network size (hidden_dim, num_layers)
# Use gradient accumulation
```

### Training Too Slow

**Problem:** Training takes too long

**Solutions for MCTS:**
```bash
# Reduce mcts_simulations (faster but weaker policies)
# Reduce games_per_iteration
# Use fewer evaluation games
# Increase checkpoint_frequency (checkpoint less often)
```

**Solutions for PPO:**
```bash
# Reduce games_per_iter
# Reduce ppo_epochs
# Use smaller network
```

### Loss is NaN

**Problem:** Training loss becomes NaN

**Solutions:**
```bash
# Reduce learning rate (e.g., 0.0001 -> 0.00001)
# Add gradient clipping (check max_grad_norm in config)
# Check data quality (reduce temperature for more stable policies)
# Use smaller network (reduce hidden_dim)
```

### Poor Win Rate

**Problem:** Agent performs poorly against baselines

**Solutions for MCTS:**
```bash
# Increase mcts_simulations (stronger policy targets)
# Train for more iterations
# Tune c_puct (try 1.0-2.0 range)
# Increase network capacity
```

**Solutions for PPO:**
```bash
# Increase games_per_iter (more diverse data)
# Tune entropy coefficient (exploration)
# Check GAE parameters (gamma, gae_lambda)
# Train longer (more iterations)
```

---

## See Also

- `scripts/train.py` - PPO training script
- `scripts/train_mcts.py` - MCTS training script
- `scripts/README_MCTS.md` - MCTS training details
- `configs/` - Configuration files
- `_03_training/` - Training implementation modules
