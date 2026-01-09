# Beasty Bar CLI Tools

Command-line tools for training, evaluating, and playing with the Beasty Bar neural network agent.

## Overview

This directory contains three main CLI scripts:

1. **train.py** - Train the neural network agent using PPO self-play
2. **evaluate.py** - Evaluate a trained agent against baseline opponents
3. **play.py** - Interactive play against the neural agent

## Installation

Ensure all dependencies are installed:

```bash
pip install torch pyyaml
```

For experiment tracking support:
```bash
pip install wandb  # For Weights & Biases
pip install tensorboard  # For TensorBoard
```

## Usage

### Training (train.py)

Train a new agent from scratch:

```bash
python scripts/train.py
```

Train with a configuration file:

```bash
python scripts/train.py --config configs/default.yaml
```

Override specific parameters:

```bash
python scripts/train.py --config configs/default.yaml --lr 0.0001 --iterations 2000
```

Resume from checkpoint:

```bash
python scripts/train.py --resume checkpoints/experiment_1/iter_000500.pt
```

Use Weights & Biases tracking:

```bash
python scripts/train.py --tracker wandb --wandb-project beastybar-runs
```

#### Training Arguments

- `--config PATH` - Path to YAML configuration file
- `--resume PATH` - Resume from checkpoint file
- `--iterations N` - Total training iterations (default: 1000)
- `--games-per-iter N` - Games per iteration (default: 256)
- `--lr FLOAT` - Learning rate (default: 0.0003)
- `--device {cpu,cuda,mps,auto}` - Training device
- `--seed INT` - Random seed
- `--tracker {console,wandb,tensorboard}` - Experiment tracker
- `--checkpoint-dir DIR` - Checkpoint directory
- `--experiment-name NAME` - Experiment name
- `--eval-frequency N` - Evaluate every N iterations
- `--hidden-dim N` - Network hidden dimension
- `--num-layers N` - Number of transformer layers
- `--verbose` - Enable verbose logging

### Evaluation (evaluate.py)

Evaluate against a single opponent:

```bash
python scripts/evaluate.py --model checkpoints/iter_001000.pt --opponent mcts-500
```

Evaluate against multiple opponents:

```bash
python scripts/evaluate.py --model checkpoints/final.pt \
    --opponents random,heuristic,mcts-500,mcts-1000 \
    --games 200
```

Save results to JSON:

```bash
python scripts/evaluate.py --model checkpoints/best.pt \
    --opponents random,heuristic,mcts-500 \
    --output results.json
```

Play as both sides for fairness:

```bash
python scripts/evaluate.py --model checkpoints/iter_001000.pt \
    --opponent mcts-500 --games 100 --both-sides
```

#### Evaluation Arguments

- `--model PATH` - Path to model checkpoint (required)
- `--opponent NAME` - Single opponent name
- `--opponents LIST` - Comma-separated list of opponents
- `--games N` - Games per opponent (default: 100)
- `--both-sides` - Play as both player 0 and player 1
- `--device {cpu,cuda,mps,auto}` - Inference device
- `--mode {greedy,stochastic,temperature}` - Inference mode
- `--temperature FLOAT` - Temperature for sampling
- `--output PATH` - Save results to JSON file
- `--verbose` - Print detailed results

#### Available Opponents

- `random` - Random agent (baseline)
- `heuristic` - Heuristic agent (intermediate)
- `mcts-100` - MCTS with 100 iterations
- `mcts-500` - MCTS with 500 iterations
- `mcts-1000` - MCTS with 1000 iterations
- `mcts-2000` - MCTS with 2000 iterations

### Interactive Play (play.py)

Play against the neural agent:

```bash
python scripts/play.py --model checkpoints/final.pt
```

You go first:

```bash
python scripts/play.py --model checkpoints/final.pt --you-start
```

Watch neural agent vs heuristic:

```bash
python scripts/play.py --model checkpoints/final.pt --opponent heuristic
```

Play multiple games:

```bash
python scripts/play.py --model checkpoints/final.pt --games 5
```

#### Play Arguments

- `--model PATH` - Path to model checkpoint (required)
- `--device {cpu,cuda,mps,auto}` - Inference device
- `--mode {greedy,stochastic,temperature}` - Inference mode
- `--temperature FLOAT` - Temperature for sampling
- `--you-start` - You play as player 0 (go first)
- `--opponent NAME` - Watch agent vs another agent
- `--seed INT` - Random seed (default: 42)
- `--games N` - Number of games to play (default: 1)
- `--verbose` - Print detailed information

## Configuration Files

Three example configuration files are provided in the `configs/` directory:

### configs/default.yaml

Balanced configuration for standard training:
- 1000 iterations
- 256 games per iteration
- Hidden dim: 128
- Learning rate: 0.0003

### configs/fast.yaml

Quick iteration configuration for testing:
- 100 iterations
- 64 games per iteration
- Smaller network (hidden dim: 64)
- Good for debugging and quick experiments

### configs/production.yaml

High-quality training for best performance:
- 5000 iterations
- 512 games per iteration
- Larger network (hidden dim: 256, 2 layers)
- Cosine learning rate decay

## Configuration File Format

Configuration files are in YAML format with three main sections:

```yaml
# Network architecture
network_config:
  hidden_dim: 128
  num_heads: 4
  num_layers: 1
  dropout: 0.1
  species_embedding_dim: 32

# PPO algorithm parameters
ppo_config:
  learning_rate: 0.0003
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  gamma: 0.99
  gae_lambda: 0.95
  ppo_epochs: 4
  minibatch_size: 64
  max_grad_norm: 0.5
  normalize_advantages: true
  clip_value: false

# Training schedule
total_iterations: 1000
games_per_iteration: 256
checkpoint_frequency: 50
eval_frequency: 10
self_play_temperature: 1.0

# Learning rate schedule
lr_warmup_iterations: 10
lr_decay: "linear"  # Options: "linear", "cosine", "none"

# Misc settings
seed: 42
device: "auto"
experiment_name: "beastybar_neural"
checkpoint_dir: "checkpoints"
```

## Checkpoints

Checkpoints are saved as PyTorch `.pt` files containing:
- Model weights
- Optimizer state
- Training step number
- Configuration
- Training metrics

Checkpoint naming convention:
- `iter_NNNNNN.pt` - Checkpoint at iteration NNNNNN
- `final.pt` - Final checkpoint after training completes

## Experiment Tracking

Three tracking backends are supported:

### Console (default)
Simple console output, no external dependencies:
```bash
python scripts/train.py --tracker console
```

### Weights & Biases
Cloud-based tracking with rich visualizations:
```bash
python scripts/train.py --tracker wandb --wandb-project my-project
```

### TensorBoard
Local tracking with TensorBoard:
```bash
python scripts/train.py --tracker tensorboard
# View with: tensorboard --logdir runs
```

## Examples

### Quick Test Run

```bash
# Train for 100 iterations with small network
python scripts/train.py --config configs/fast.yaml

# Evaluate the result
python scripts/evaluate.py \
    --model checkpoints/beastybar_fast_test/final.pt \
    --opponents random,heuristic \
    --games 50
```

### Production Training

```bash
# Start long training run with W&B tracking
python scripts/train.py \
    --config configs/production.yaml \
    --tracker wandb \
    --wandb-project beastybar-production

# Comprehensive evaluation
python scripts/evaluate.py \
    --model checkpoints/beastybar_production/final.pt \
    --opponents random,heuristic,mcts-500,mcts-1000 \
    --games 200 \
    --both-sides \
    --output production_results.json
```

### Resume Training

```bash
# Resume from checkpoint with different learning rate
python scripts/train.py \
    --resume checkpoints/experiment_1/iter_000500.pt \
    --lr 0.0001 \
    --iterations 2000
```

## Troubleshooting

### PyTorch not found
Install PyTorch: `pip install torch`

### CUDA out of memory
Reduce batch size or use gradient accumulation:
```yaml
ppo_config:
  minibatch_size: 32  # Reduce from 64
gradient_accumulation_steps: 2  # Accumulate 2 batches
```

### Wandb authentication
First time using wandb:
```bash
wandb login
```

### Import errors
Ensure you're running from the project root directory.

## Performance Tips

1. **GPU Training**: Use `--device cuda` for 10-100x speedup on GPU
2. **Parallel Games**: Larger `games_per_iteration` improves sample efficiency
3. **Learning Rate**: Start with 0.0003, reduce if training unstable
4. **Network Size**: Larger networks learn better but train slower
5. **Checkpointing**: Save frequently to avoid losing progress

## See Also

- `NEURAL_TRAINING_PLAN.md` - Full training plan and architecture details
- `_03_training/` - Training implementation modules
- `_02_agents/neural/` - Neural network architecture
