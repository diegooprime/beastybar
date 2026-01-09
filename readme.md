Beasty Bar is a board game I like a lot. It's fun, simple, easy to learn and requires strategy. 
I played a ton with friends and wanted the “best” strategy. I searched online. Nothing.
So I’m building a simulator to test strategies until I find the best one. Then I’ll use the engine to sharpen my intuition and win more games.
First project of this type for me.
Purpose: Have the best Beasty Bar strategy in the world.

### Project structure:
- **_01_simulator**: Full game rules and state management (deterministic, side-effect free)
- **_02_agents**: AI players (Random, Heuristic, MCTS)
- **_03_training**: Tournament runner and Elo ratings
- **_04_ui**: FastAPI interface and static viewer for human vs. human play
- **_05_other**: Tests, utilities, docs, and references

The code for each section is independent so we can modify one without fucking up the other sections. 

### Quick Start

#### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### 2. Run the UI (Human vs Human)
```bash
uvicorn _04_ui.app:create_app --reload
# Visit http://localhost:8000
```

#### 3. Run Tests
```bash
pytest _05_other/tests -ra
```

#### 4. Run Training (on NUC)
All training runs happen on `primenuc@prime-nuc`. Use the remote script:

```bash
# Sync code and run benchmark
./scripts/remote.sh sync
./scripts/remote.sh run --games 100

# Long training in background (tmux)
./scripts/remote.sh train --games 500 --include-slow

# Monitor
./scripts/remote.sh status   # check if running
./scripts/remote.sh attach   # view live (Ctrl+B, D to detach)
./scripts/remote.sh logs     # tail output
```

### How It Works

1. **Simulator** (`_01_simulator/`) provides deterministic game engine with seed-threaded randomness
2. **Agents** (`_02_agents/`) play against each other to find optimal strategies
3. **Training** (`_03_training/`) runs tournaments and tracks Elo ratings
4. **UI** (`_04_ui/`) lets humans play both sides of a match, inspect turn history, and replay deterministic seeds

---

## Neural Network Training

### Architecture

**Transformer-based policy-value network** (17M parameters):
- Input: 988-dim observation (game state encoding)
- Output: 124-dim policy logits + scalar value
- Hidden: 256-dim, 8 attention heads, 4 transformer layers
- Species embeddings: 64-dim learned embeddings for 12 animal types

### Training Approaches

We tried two approaches in parallel:

#### 1. PPO (Proximal Policy Optimization)
- Standard RL approach with clipped surrogate objective
- Fast game generation (~2048 games in 17s)
- **Problem**: Exploded with NaN losses at iteration 80

#### 2. AlphaZero-style MCTS (Current)
- MCTS search (200 simulations) generates improved policy targets
- Network trained via cross-entropy against MCTS visit distributions
- Value target = game outcome (+1/-1/0)
- **Status**: Active, showing healthy learning

### Key Problems & Solutions

#### Problem 1: MCTS Too Slow
- **Symptom**: ~35 seconds per game (sequential neural network calls)
- **Solution**: BatchMCTS - batch leaf evaluations across parallel games
- **Result**: ~0.17s per game (200x speedup)

#### Problem 2: Self-Play Collapse
- **Symptom**: Model performance degraded from 56% to 16% vs random
- **Cause**: Pure self-play leads to co-adaptation; model exploits own weaknesses
- **Solution**: Opponent diversity pool

### Opponent Diversity (Critical Fix)

Instead of 100% self-play, training now uses mixed opponents:

| Opponent Type | Weight | Purpose |
|--------------|--------|---------|
| Current network | 60% | Main learning signal |
| Past checkpoints | 20% | Prevent catastrophic forgetting |
| Random agent | 10% | Baseline calibration |
| Heuristic agent | 10% | Quality anchor |

Checkpoints added to pool every 20 iterations (max 10 kept).

### MCTS Configuration (Tuned for Exploration)

```yaml
mcts_config:
  num_simulations: 200      # Search depth
  c_puct: 2.0               # Exploration constant (was 1.5)
  temperature_drop_move: 30 # Stay stochastic longer (was 15)
  final_temperature: 0.25   # Less deterministic (was 0.1)
  dirichlet_alpha: 0.5      # More uniform noise (was 0.3)
  dirichlet_epsilon: 0.4    # More noise mixing (was 0.25)
  batch_size: 32            # Batched leaf evaluation
  virtual_loss: 3.0         # Parallel MCTS exploration
```

### Training Hyperparameters

```yaml
learning_rate: 0.00005      # Slow learning (was 0.0001)
entropy_bonus_weight: 0.05  # Encourage exploration (was 0.01)
games_per_iteration: 128
batch_size: 512
epochs_per_iteration: 4
total_iterations: 200
```

### Running Training (RunPod H100)

```bash
# SSH to RunPod
ssh runpod

# Start MCTS training with opponent diversity
cd ~/beastybar
nohup python -m _03_training.run_mcts_training \
  --config configs/h100_mcts.yaml \
  > logs/mcts_training.log 2>&1 &

# Monitor
tail -f logs/mcts_training.log
```

### Training Metrics to Watch

- **Policy loss**: Should decrease (network matching MCTS policies)
- **Value loss**: Should decrease (predicting game outcomes)
- **Entropy**: Should stay moderate (0.5-2.0), not collapse to 0
- **Win rate vs random**: Should increase above 50%
- **Win rate vs heuristic**: Target >40% indicates real learning

### File Structure

```
_03_training/
├── mcts_trainer.py      # AlphaZero training loop
├── mcts_self_play.py    # BatchMCTS game generation
├── opponent_pool.py     # Diverse opponent sampling
├── trainer.py           # PPO training (deprecated)
├── evaluation.py        # Win rate evaluation
└── tracking.py          # Metrics logging

configs/
├── h100_mcts.yaml       # MCTS training config
└── h100_scaled.yaml     # PPO config (deprecated)
```

---

### Misc
- Built in Python. Fast to build. I want to get better at it.
- FastAPI + static html. Fast, simple and zero build.
- Beasty Bar intro: https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf
- Beady Bar more rules: https://www.ultraboardgames.com/beasty-bar/game-rules.php
