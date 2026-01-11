# AlphaZero-Style Training

MCTS search generates improved policy targets for supervised learning.

## How It Works

1. **Self-play with MCTS**: Generate games using tree search guided by neural network
2. **Policy improvement**: MCTS visit distribution is stronger than raw network policy
3. **Supervised learning**: Train network to match MCTS policies and game outcomes

## Training Loop

```
For each iteration:
1. Generate N games with MCTS (stores policy distributions per move)
2. Assign terminal values (+1 win, -1 loss, 0 draw)
3. Train network: Loss = CrossEntropy(policy) + MSE(value) - entropy_bonus
```

## Usage

```bash
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml --wandb
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml --checkpoint checkpoints/iter_000100.pt
```

## Configuration

```yaml
mcts_simulations: 200      # Search depth per move
c_puct: 2.0                # Exploration constant
temperature: 1.0           # Action sampling

learning_rate: 0.00005
entropy_bonus_weight: 0.05
games_per_iteration: 128
epochs_per_iteration: 4
total_iterations: 200

opponent_pool:
  current: 0.6
  checkpoint: 0.2
  random: 0.1
  heuristic: 0.1
```

## MCTS vs PPO

| Aspect | MCTS | PPO |
|--------|------|-----|
| Policy target | MCTS visit distribution | Single action |
| Training signal | Stronger (rich distribution) | Weaker (sparse reward) |
| Stability | More stable | Requires tuning |
| Speed | Slower (search overhead) | Faster |

## Metrics

| Metric | Target |
|--------|--------|
| Policy loss | Decreasing |
| Value loss | Decreasing |
| Entropy | 0.5-2.0 |
| Win vs random | >50% |
| Win vs heuristic | >40% |

## Troubleshooting

**Too slow:** Reduce `mcts_simulations` or `games_per_iteration`

**NaN loss:** Reduce `learning_rate`, increase `entropy_bonus_weight`

**Win rate drops:** Check opponent pool is enabled, increase diversity

## References

- [AlphaZero (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)
