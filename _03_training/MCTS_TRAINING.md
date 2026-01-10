# AlphaZero-Style MCTS Training

Complete training infrastructure for Beasty Bar using the AlphaZero methodology.

## Overview

1. **Self-Play with MCTS**: Generate games using Monte Carlo Tree Search guided by neural network
2. **Policy Improvement**: MCTS search improves upon raw network policy through lookahead
3. **Supervised Learning**: Train network to match improved MCTS policies and game outcomes

### Why MCTS over PPO?

| Aspect | MCTS (AlphaZero) | PPO |
|--------|------------------|-----|
| **Policy Target** | MCTS visit distribution (full) | Single action taken |
| **Training Signal** | Stronger (rich distribution) | Weaker (sparse reward) |
| **Stability** | More stable for zero-sum games | Requires careful tuning |
| **Speed** | Slower (tree search overhead) | Faster (vectorized) |

## Data Flow

```
1. Self-Play Generation (per move):
   ┌─────────────────────────────────────┐
   │ Run MCTS Search (200 simulations)   │
   │  - Neural network guides selection  │
   │  - Build search tree with PUCT      │
   │  - Add Dirichlet noise at root      │
   └─────────────────────────────────────┘
              ↓
   ┌─────────────────────────────────────┐
   │ Convert Visit Counts to Policy      │
   │  policy[action] = visits / total    │
   └─────────────────────────────────────┘
              ↓
   ┌─────────────────────────────────────┐
   │ Sample Action from MCTS Policy      │
   │  - Temperature controls exploration │
   │  - Store MCTS policy as target      │
   └─────────────────────────────────────┘

2. Training (per iteration):
   ┌─────────────────────────────────────┐
   │ Generate N Games (e.g., 128 games)  │
   │  - Use opponent pool for diversity  │
   │  - Store MCTS policies at each move │
   └─────────────────────────────────────┘
              ↓
   ┌─────────────────────────────────────┐
   │ Assign Terminal Values              │
   │  - Winner: +1, Loser: -1, Draw: 0   │
   │  - All moves get game outcome       │
   └─────────────────────────────────────┘
              ↓
   ┌─────────────────────────────────────┐
   │ Train Network (4 epochs)            │
   │  Loss = policy_CE + value_MSE       │
   │         - entropy_bonus             │
   └─────────────────────────────────────┘
```

## Loss Function

```python
# Policy Loss: Match network policy to MCTS visit distribution
policy_loss = CrossEntropy(network_policy, mcts_policy)

# Value Loss: Predict game outcome
value_loss = MSE(network_value, game_outcome)

# Entropy Bonus: Encourage exploration
entropy = -sum(p * log(p))

# Total Loss
total_loss = policy_loss + value_loss - 0.05 * entropy
```

## Usage

### Quick Start

```bash
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml
```

### With Weights & Biases

```bash
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml --wandb
```

### Resume from Checkpoint

```bash
uv run scripts/train_mcts.py \
  --config configs/h100_mcts.yaml \
  --checkpoint checkpoints/iter_000100.pt
```

## Configuration

### Key Parameters

```yaml
# MCTS Search
mcts_simulations: 200      # Search depth per move
c_puct: 2.0               # Exploration constant
temperature: 1.0          # Action sampling temperature
batch_size: 32            # Batched leaf evaluation
virtual_loss: 3.0         # Parallel MCTS exploration

# Training
learning_rate: 0.00005
entropy_bonus_weight: 0.05
games_per_iteration: 128
epochs_per_iteration: 4
total_iterations: 200

# Opponent Diversity (Critical!)
opponent_pool:
  current: 0.6      # 60% self-play
  checkpoint: 0.2   # 20% past checkpoints
  random: 0.1       # 10% random baseline
  heuristic: 0.1    # 10% heuristic anchor
```

### Hyperparameter Guide

| Parameter | Low | Medium | High | Effect |
|-----------|-----|--------|------|--------|
| `mcts_simulations` | 50 | 200 | 800 | Stronger policy targets |
| `c_puct` | 1.0 | 2.0 | 4.0 | More exploration |
| `temperature` | 0.1 | 1.0 | 2.0 | More stochastic |
| `entropy_bonus` | 0.01 | 0.05 | 0.1 | Prevents collapse |
| `learning_rate` | 1e-5 | 5e-5 | 1e-4 | Faster learning |

## PUCT Selection

```
PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

Where:
- Q(s, a): Average value of taking action a in state s
- P(s, a): Prior probability from neural network
- N(s): Visit count of parent node
- N(s, a): Visit count of child node
```

### Dirichlet Noise

At root node, add noise for exploration:

```
P'(s, a) = (1 - ε) * P(s, a) + ε * η_a

Where:
- ε = 0.4 (noise weight)
- η ~ Dirichlet(α) with α = 0.5
```

## Opponent Diversity

**Critical for training stability.** Pure self-play causes collapse where model exploits own weaknesses.

The opponent pool mixes:
- **60% current network**: Main learning signal
- **20% past checkpoints**: Prevents catastrophic forgetting
- **10% random agent**: Baseline calibration
- **10% heuristic agent**: Quality anchor

Checkpoints added to pool every 20 iterations (max 10 kept).

## Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Sequential MCTS | ~0.03 games/sec | Network bottleneck |
| BatchMCTS | ~0.5 games/sec | Batched evaluation |
| PPO (comparison) | ~100 games/sec | Vectorized |

## Key Files

```
_03_training/
├── mcts_trainer.py      # MCTSTrainer - main training loop
├── mcts_self_play.py    # MCTS game generation
├── game_generator.py    # Self-play with opponent pool
├── opponent_pool.py     # Diverse opponent sampling
└── evaluation.py        # Win rate evaluation
```

## Metrics to Watch

| Metric | Target | Meaning |
|--------|--------|---------|
| Policy loss | Decreasing | Network matching MCTS |
| Value loss | Decreasing | Predicting outcomes |
| Entropy | 0.5-2.0 | Healthy exploration |
| Win vs random | >50% | Basic strategy |
| Win vs heuristic | >40% | Real understanding |

## Troubleshooting

### Training Too Slow
- Reduce `mcts_simulations` (faster, weaker)
- Increase `batch_size` for BatchMCTS
- Reduce `games_per_iteration`

### Loss is NaN
- Reduce `learning_rate`
- Check gradient clipping
- Increase entropy bonus

### Win Rate Drops
- Check opponent pool is enabled
- Increase opponent diversity
- Reduce learning rate

## References

- [AlphaZero Paper (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero Paper (Silver et al., 2017)](https://www.nature.com/articles/nature24270)
