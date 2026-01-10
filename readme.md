# Beasty Bar AI

**Goal: Create the best Beasty Bar player ever, for any opponent, all the time.**

A complete AI training platform for the board game [Beasty Bar](https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf). Includes a deterministic game engine, multiple AI agents (random, heuristic, MCTS, neural), and training infrastructure with AlphaZero-style self-play.

## Project Structure

```
beastybar/
├── _01_simulator/     # Game engine (deterministic, immutable state)
│   ├── engine.py      # Core rules: legal_actions(), step(), is_terminal()
│   ├── state.py       # Immutable State, Card, Zones dataclasses
│   ├── cards.py       # Species-specific handlers (12 animals)
│   ├── action_space.py# Fixed 124-action catalog for neural networks
│   ├── observations.py# 988-dim state encoding for neural networks
│   └── _cython/       # Optional Cython acceleration (200x speedup)
│
├── _02_agents/        # AI players
│   ├── random_agent.py# Baseline: uniform random from legal actions
│   ├── heuristic.py   # Rule-based with material evaluation
│   ├── mcts/          # Monte Carlo Tree Search
│   │   ├── search.py  # PUCT-based tree search with neural guidance
│   │   ├── batch_mcts.py # Batched MCTS with virtual loss (10x speedup)
│   │   └── agent.py   # MCTSAgent wrapper
│   └── neural/        # Neural network agent
│       ├── network.py # Transformer policy-value network (17M params)
│       └── agent.py   # NeuralAgent with batch inference
│
├── _03_training/      # Training infrastructure
│   ├── trainer.py     # PPO training orchestrator
│   ├── mcts_trainer.py# AlphaZero-style MCTS training
│   ├── mcts_self_play.py # MCTS game generation
│   ├── game_generator.py # Self-play with opponent diversity
│   ├── opponent_pool.py  # Mixed opponent sampling (critical for stability)
│   ├── evaluation.py  # Win rate evaluation with confidence intervals
│   ├── ppo.py         # Proximal Policy Optimization
│   └── checkpoint_manager.py # Model persistence
│
├── _04_ui/            # Web interface
│   ├── app.py         # FastAPI server
│   └── static/        # HTML/JS frontend
│
├── _05_other/         # Tests and documentation
│   ├── tests/         # Comprehensive test suite
│   └── rules.md       # Official Beasty Bar rules
│
├── scripts/           # CLI tools
│   ├── train.py       # PPO training script
│   ├── train_mcts.py  # MCTS training script
│   ├── evaluate.py    # Model evaluation
│   └── play.py        # Interactive play
│
└── configs/           # Training configurations
    ├── h100_mcts.yaml # H100 GPU MCTS config
    ├── default.yaml   # Standard PPO config
    └── fast.yaml      # Quick testing config
```

## Quick Start

### 1. Setup Environment

```bash
# Using uv (recommended)
uv sync

# Or pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run Tests

```bash
pytest _05_other/tests -ra
```

### 3. Run the Web UI

```bash
uvicorn _04_ui.app:create_app --reload
# Visit http://localhost:8000
```

### 4. Train an Agent

```bash
# PPO training (fast, good baseline)
uv run scripts/train.py --config configs/default.yaml

# MCTS training (slower, higher quality)
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml
```

### 5. Evaluate a Model

```bash
uv run scripts/evaluate.py \
  --model checkpoints/model.pt \
  --opponents random,heuristic \
  --games 100
```

## Neural Network Architecture

**Transformer-based policy-value network** (17M parameters):

| Component | Details |
|-----------|---------|
| Input | 988-dim observation (game state encoding) |
| Output | 124-dim policy logits + scalar value |
| Hidden | 256-dim, 8 attention heads, 4 transformer layers |
| Species embeddings | 64-dim learned embeddings for 12 animals |

The observation encodes:
- Queue cards (5 slots x 17 features)
- Beasty Bar cards (24 slots)
- That's It cards (24 slots)
- Own hand (4 cards, full visibility)
- Opponent hand (4 slots, masked)
- Scalars (turn, active player, hand counts)

## Training Approaches

### PPO (Proximal Policy Optimization)

Standard RL with clipped surrogate objective.

**Pros:** Fast game generation (~2048 games in 17s)
**Cons:** Hit NaN losses at iteration 80 in some runs

```bash
uv run scripts/train.py --config configs/default.yaml
```

### AlphaZero-style MCTS (Recommended)

MCTS search generates improved policy targets for supervised learning.

**Pros:** More stable, stronger final performance
**Cons:** Slower (MCTS search overhead)

```bash
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml
```

## Opponent Diversity (Critical)

Pure self-play causes training collapse—the model exploits its own weaknesses and develops blind spots against different playstyles. Research on hidden-information games (Libratus, Pluribus) shows diverse opponents are essential. The opponent pool mixes multiple agent types during training:

| Opponent Type | Default Weight | Purpose |
|--------------|----------------|---------|
| Current network | 60% | Main learning signal (self-play) |
| Past checkpoints | 20% | Prevent catastrophic forgetting |
| Random agent | 10% | Baseline calibration |
| Heuristic agent | 10% | Quality anchor |
| MCTS agents | 0% | Strong search-based opponents (optional) |

### Opponent Types in Detail

**Current Network** — The agent plays against itself, learning from the resulting win/loss signal. This is the core self-play loop that drives improvement but can stagnate without diversity.

**Past Checkpoints** — Historical snapshots of the network saved every 20 iterations (max 10 kept). Training against older versions prevents catastrophic forgetting where the model "forgets" how to beat strategies it previously mastered.

**Random Agent** — Plays uniformly random legal moves. Ensures the network doesn't lose to trivial strategies and provides a floor for basic competence. If win rate vs random drops, something is fundamentally broken.

**Heuristic Agent** — Rule-based agent using material evaluation (card points weighted by zone: bar > queue front > hand > queue back > that's it). Configurable parameters include:
- `bar_weight`: Value multiplier for cards in the Beasty Bar (default 2.0)
- `aggression`: Bias toward offensive vs defensive plays (0-1 scale)
- `noise_epsilon`: Random noise for bounded rationality (simulates human mistakes)
- `species_weights`: Per-animal multipliers for specialized strategies

Six pre-built heuristic variants are available via `create_heuristic_variants()`:
1. **Aggressive** — High aggression (0.9), prioritizes attacking opponent position
2. **Defensive** — Low aggression (0.1), focuses on protecting own cards
3. **Bar Focused** — Triple weight on bar cards, plays for final scoring
4. **Queue Controller** — Emphasizes queue front positioning for board control
5. **Skunk Specialist** — Higher value on Skunk plays, aims for double-elimination
6. **OnlineStrategies** — Reactive counter-play agent that tracks played cards to infer opponent's remaining hand and holds counter cards (Parrot, Skunk, Seal) for punishing overcommitments

**MCTS Agents** — Monte Carlo Tree Search opponents using the current network for position evaluation. Six default configurations with varying characteristics:
- `mcts_exploit` — Low exploration (c_puct=0.5), greedy play
- `mcts_balanced` — Standard parameters (c_puct=1.5, 200 sims)
- `mcts_explore` — High exploration (c_puct=3.0), considers unusual moves
- `mcts_fast` — Shallow search (50 sims), quick but weaker
- `mcts_deep` — Deep search (500 sims), strongest but slowest
- `mcts_noisy` — High Dirichlet noise (ε=0.5), diverse move selection

MCTS opponents are disabled by default (`mcts_weight=0`) because they use pure Python and are 5x slower than Cython-accelerated generation. Enable them for stronger training signal at the cost of throughput.

## Performance Optimizations

### Cython Acceleration

The simulator has optional Cython bindings for 200x speedup in batch game generation.

```bash
# Build Cython extensions
bash scripts/build_cython.sh

# Benchmark
uv run scripts/benchmark_cython.py
```

Auto-detected at import; falls back to pure Python if not available.

### BatchMCTS

Batched neural network evaluation across parallel search trees with virtual loss for diverse exploration.

| Feature | Sequential MCTS | Batch MCTS |
|---------|----------------|------------|
| Network calls | N x simulations | ~simulations (batched) |
| GPU utilization | Low | High |
| Throughput | Baseline | 5-10x faster |

## MCTS Configuration

```yaml
mcts_config:
  num_simulations: 200      # Search depth per move
  c_puct: 2.0               # Exploration constant
  temperature: 1.0          # Action sampling temperature
  temperature_drop_move: 30 # Stay stochastic longer
  final_temperature: 0.25   # Less deterministic late game
  dirichlet_alpha: 0.5      # Root noise concentration
  dirichlet_epsilon: 0.4    # Root noise mixing
  batch_size: 32            # Batched leaf evaluation
  virtual_loss: 3.0         # Parallel MCTS penalty
```

## Training Hyperparameters

```yaml
learning_rate: 0.00005      # Slow learning for stability
entropy_bonus_weight: 0.05  # Encourage exploration
games_per_iteration: 128    # Self-play games per step
batch_size: 512             # Training batch size
epochs_per_iteration: 4     # Network updates per iteration
total_iterations: 200       # Training length
```

## Key Metrics to Watch

| Metric | Target | Meaning |
|--------|--------|---------|
| Policy loss | Decreasing | Network matching MCTS policies |
| Value loss | Decreasing | Predicting game outcomes |
| Entropy | 0.5-2.0 | Healthy exploration (not collapsed) |
| Win rate vs random | >50% | Learning basic strategy |
| Win rate vs heuristic | >40% | Real strategic understanding |

## Game Rules Summary

Beasty Bar is a 2-player card game where animals jostle to enter Heaven's Gate.

**Turn structure:**
1. Play one card at the back of the queue
2. Execute the card's animal action
3. Process recurring actions (Hippo, Crocodile, Giraffe)
4. Five-animal check: front 2 enter bar, last 1 bounced
5. Draw a card

**12 Animals** (strength in parentheses):
- **Lion (12)**: Scares monkeys, moves to front
- **Hippo (11)**: Recurring - passes weaker animals toward gate
- **Crocodile (10)**: Recurring - eats weaker animals ahead
- **Snake (9)**: Sorts queue by strength descending
- **Giraffe (8)**: Recurring - jumps over one weaker animal
- **Zebra (7)**: Permanent blocker for hippo/croc
- **Seal (6)**: Reverses entire queue
- **Chameleon (5)**: Imitates another species
- **Monkey (4)**: If 2+, bounces hippos/crocs, moves to front
- **Kangaroo (3)**: Jumps over last 1-2 animals
- **Parrot (2)**: Bounces any one animal
- **Skunk (1)**: Expels highest and second-highest strength animals

**Scoring:** Points per animal in the Beasty Bar. Winner has most points.

## Development

```bash
# Run linting
uv run ruff check .

# Run type checking
uv run mypy _01_simulator _02_agents _03_training

# Run all tests
uv run pytest _05_other/tests -ra
```

## References

- [Beasty Bar Rules PDF](https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
