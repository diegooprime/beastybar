# AlphaZero-Style MCTS Training for Beasty Bar

This directory contains a complete AlphaZero-style MCTS training infrastructure for the Beasty Bar neural network agent.

## Overview

The MCTS training approach follows the AlphaZero methodology:

1. **Self-Play with MCTS**: Generate games using Monte Carlo Tree Search guided by the neural network
2. **Policy Improvement**: MCTS search improves upon the raw network policy through lookahead
3. **Supervised Learning**: Train network to match improved MCTS policies and game outcomes

### Key Advantages over PPO

- **Stronger Training Signal**: Policy target is the improved MCTS visit distribution, not just the action taken
- **Simpler Loss Function**: No clipped surrogate objective, GAE, or advantage estimation
- **More Stable**: Better suited for two-player zero-sum games
- **Higher Quality Play**: Expected to achieve higher Elo with sufficient training

### Trade-offs

- **Slower**: MCTS search is compute-intensive (800+ simulations per move)
- **Not GPU-Optimized**: Tree search doesn't benefit from vectorization like PPO self-play

## Architecture

### Core Components

**`mcts_self_play.py`**
- `MCTSNode`: MCTS search tree node with PUCT selection
- `MCTSTransition`: Training sample with MCTS policy distribution
- `MCTSTrajectory`: Complete game trajectory with both players' data
- `mcts_search()`: Neural MCTS with Dirichlet noise at root
- `play_mcts_game()`: Self-play game generation
- `generate_mcts_games()`: Batch game generation

**`mcts_trainer.py`**
- `MCTSTrainerConfig`: Complete training configuration
- `MCTSTrainer`: Main training orchestrator
- `policy_loss()`: Cross-entropy between network and MCTS policies
- `value_loss()`: MSE between predicted values and game outcomes
- `entropy_bonus()`: Exploration regularization

**`example_mcts_training.py`**
- Example configurations and usage patterns
- Comparison with PPO training
- Quick test and production configs

## Data Flow

```
1. Self-Play Generation (per move):
   ┌─────────────────────────────────────┐
   │ Run MCTS Search (800 simulations)   │
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
   │ Generate N Games (e.g., 512 games)  │
   │  - Both players use same network    │
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
   │ Train Network (10 epochs)           │
   │  Loss = policy_CE + value_MSE       │
   │         - entropy_bonus             │
   └─────────────────────────────────────┘
```

## Training Data Structure

### MCTSTransition
Each training sample contains:
```python
@dataclass
class MCTSTransition:
    state: State                    # Game state (for debugging)
    observation: NDArray            # State tensor (988,)
    action_mask: NDArray            # Legal actions (124,)
    mcts_policy: dict[int, float]   # MCTS visit distribution ← KEY DIFFERENCE
    action_taken: int               # Action actually played
    value: float                    # Game outcome (+1/-1/0)
    player: int                     # Player who moved (0 or 1)
```

**Key Difference from PPO**:
- PPO stores only `action_taken` and `action_prob` (single action)
- MCTS stores full `mcts_policy` distribution over all legal actions
- This provides a richer training signal from MCTS search

## Loss Function

```python
# Policy Loss: Match network policy to MCTS visit distribution
policy_loss = CrossEntropy(network_policy, mcts_policy)

# Value Loss: Predict game outcome
value_loss = MSE(network_value, game_outcome)

# Entropy Bonus: Encourage exploration
entropy = -sum(p * log(p))

# Total Loss
total_loss = policy_loss + value_loss - 0.01 * entropy
```

### Comparison with PPO

| Component | MCTS (AlphaZero) | PPO |
|-----------|------------------|-----|
| **Policy Target** | MCTS visit distribution | Single action taken |
| **Policy Loss** | Cross-entropy | Clipped surrogate objective |
| **Value Target** | Game outcome | GAE-computed returns |
| **Value Loss** | MSE | Clipped MSE |
| **Advantage** | Not used | GAE (λ=0.95) |
| **Training Signal** | Stronger (full distribution) | Weaker (single action) |
| **Stability** | More stable | Requires careful tuning |

## Usage Examples

### Quick Test (5 iterations)

```python
from _03_training.mcts_trainer import MCTSTrainer, MCTSTrainerConfig

config = MCTSTrainerConfig(
    games_per_iteration=10,
    mcts_simulations=100,
    total_iterations=5,
    experiment_name="mcts_test",
)

trainer = MCTSTrainer(config)
trainer.train()
```

### Production Training

```python
from _03_training.mcts_trainer import MCTSTrainer, MCTSTrainerConfig
from _02_agents.neural.utils import NetworkConfig

config = MCTSTrainerConfig(
    # Network
    network_config=NetworkConfig(
        hidden_dim=512,
        num_layers=6,
    ),

    # Self-play (AlphaZero defaults)
    games_per_iteration=512,
    mcts_simulations=800,
    temperature=1.0,
    c_puct=1.0,

    # Training
    total_iterations=1000,
    batch_size=256,
    epochs_per_iteration=10,
    learning_rate=1e-3,

    # Schedule
    lr_decay="cosine",
    checkpoint_frequency=50,

    experiment_name="beastybar_alphazero",
)

trainer = MCTSTrainer(config)
trainer.train()
```

### Resume from Checkpoint

```python
from _03_training.mcts_trainer import create_trainer_from_checkpoint

trainer = create_trainer_from_checkpoint(
    path="checkpoints/beastybar_alphazero/iter_000500.pt",
    config_overrides={"total_iterations": 1500},  # Train more
)
trainer.train()
```

## MCTS Search Details

### PUCT Selection Formula

The PUCT (Predictor + UCT) formula guides node selection during MCTS:

```
PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

Where:
- Q(s, a): Average value of taking action a in state s
- P(s, a): Prior probability from neural network
- N(s): Visit count of parent node
- N(s, a): Visit count of child node
- c_puct: Exploration constant (default: 1.0)
```

### Dirichlet Noise

At the root node, add Dirichlet noise to priors for exploration:

```python
P'(s, a) = (1 - ε) * P(s, a) + ε * η_a

Where:
- ε = 0.25 (noise weight)
- η ~ Dirichlet(α) with α = 0.3
```

This encourages exploration of diverse moves during self-play.

## Hyperparameter Guidance

### MCTS Simulations
- **100-200**: Quick testing, weak play
- **400-800**: Good quality, reasonable speed
- **1600+**: Highest quality, very slow

### Temperature
- **1.0**: Sample proportionally to visits (exploration)
- **0.5**: Sharper distribution (less exploration)
- **0.0**: Greedy (select most visited action)

Recommendation: Use 1.0 during self-play, 0.0 for evaluation.

### Games per Iteration
- **10-50**: Quick prototyping
- **256-512**: Production training
- **1024+**: Large-scale training

More games = more diverse data per iteration.

### Learning Rate Schedule
- **Warmup**: Linear ramp from 0 to base_lr (10 iterations)
- **Decay**: Cosine annealing to 0 over training
- **Base LR**: 1e-3 is a good starting point

### Batch Size
- **64-128**: Small networks, limited memory
- **256**: Good default
- **512+**: Large networks, lots of memory

## Performance Expectations

### Training Speed
- **PPO**: ~100 games/minute (vectorized)
- **MCTS (800 sims)**: ~5 games/minute (sequential search)

MCTS is ~20x slower due to tree search overhead.

### Quality (Expected)
- **PPO (well-tuned)**: ~1400 Elo
- **MCTS (800 sims)**: ~1600+ Elo

MCTS expected to achieve higher strength with sufficient training.

### Convergence
- **PPO**: Fast early learning, plateaus around 500 iterations
- **MCTS**: Slower early learning, continues improving longer

MCTS benefits more from extended training (1000+ iterations).

## Comparison with Existing Trainer

### trainer.py (PPO-based)
```python
# PPO self-play: fast, vectorized
trajectories = generate_games(network, num_games=256)

# Train with GAE advantages
advantages, returns = compute_gae(...)
loss = clipped_policy_loss + value_loss - entropy
```

### mcts_trainer.py (AlphaZero-based)
```python
# MCTS self-play: slower, higher quality
trajectories = generate_mcts_games(network, num_games=100, mcts_simulations=800)

# Train with MCTS policies
loss = policy_cross_entropy + value_loss - entropy
```

## File Organization

```
_03_training/
├── mcts_self_play.py          # MCTS search and game generation
├── mcts_trainer.py            # Training orchestration
├── example_mcts_training.py   # Usage examples
├── MCTS_TRAINING.md          # This file
│
# Existing PPO infrastructure (untouched)
├── trainer.py                 # PPO-based trainer
├── self_play.py              # PPO-based self-play
├── ppo.py                    # PPO algorithm
├── replay_buffer.py          # Experience storage
└── ...
```

Both training methods coexist and can be used interchangeably.

## Tips and Best Practices

1. **Start Small**: Test with 10 games and 100 simulations first
2. **Monitor Metrics**: Watch policy loss, value loss, and win rates
3. **Checkpoint Often**: Save every 50 iterations for long training
4. **Evaluate Regularly**: Test against baselines every 10 iterations
5. **Compare Approaches**: Try both MCTS and PPO to see which works better
6. **GPU Utilization**: MCTS doesn't saturate GPU like PPO (tree search is CPU-bound)
7. **Patience**: MCTS training takes time but achieves higher quality

## Future Improvements

Possible enhancements:
- [ ] Parallel MCTS (virtual loss)
- [ ] Batched MCTS inference (combine leaf evaluations)
- [ ] Temperature annealing schedule
- [ ] Resignation threshold
- [ ] Policy distillation from strong MCTS to fast network
- [ ] Muzero-style model-based planning

## References

- [AlphaZero Paper (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero Paper (Silver et al., 2017)](https://www.nature.com/articles/nature24270)
- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)

## License

Same as parent project (Beasty Bar).

---

## Known Issue: MCTS Cold-Start Collapse (Jan 2026)

**Status**: MCTS training collapses even with PPO warmstart. Deprioritized until debugged.

**Symptoms**:
- Win rate vs random drops from ~50% to ~20% over 20-30 iterations
- Value loss decreases but policy degrades
- Happens both cold-start and with warmstart

**Workaround**: Use PPO training instead (achieves 84%+ vs random, 32%+ vs heuristic).

**TODO**: Debug MCTS search/backup or training loop before next attempt.
