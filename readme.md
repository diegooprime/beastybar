# Beasty Bar AI

**Goal: Create the best Beasty Bar player ever, for any opponent.**

This is my first machine learning project. I've read a lot of theory, but this is the first time I'm actually building something from scratch. I'm using [Claude Code](https://github.com/anthropics/claude-code) as my main collaborator for thinking through problems, evaluating approaches, and writing the code. I'm doing this for fun because it's interesting and maximizes my learning.

**Fair warning:** This is a messy project. I'm learning as I go, trying lots of experiments, and figuring things out. I've spent ~$300 on RunPod A100s so far, mostly running experiments to see what works. I'm using W&B for tracking runs, but honestly it's been a mess—lots of abandoned runs, inconsistent naming, and experiments that went nowhere. That's part of learning.

---

[Beasty Bar](https://www.zoch-verlag.com/zoch_en/brands/beasty-bar/) is a card game designed by Stefan Kloß and published by Zoch Verlag in 2014. Players compete to get their animal cards into "Heaven" (the bar) while avoiding "Hell". Each of the 12 animal species has unique abilities that trigger when played—lions scare away weaker animals, skunks repel neighbors, crocodiles eat the weakest, and so on. The game combines hand management, timing, and player interaction.

Watch this video for an explanation: [Beasty Bar Rules](https://youtu.be/UTweio-pcro?si=FdxYL1L6yX5mU5SC)

---

## What This Codebase Is

This project is three things:

1. **A Game Simulator** — A complete implementation of Beasty Bar rules, state management, and action space. Currently has both Python (for development) and Cython (for training speed) implementations.

2. **A PPO Training Pipeline** — Self-play reinforcement learning with opponent diversity to train neural networks that play Beasty Bar. Runs on GPU (I'm using RunPod A100s), with Cython acceleration for 200x faster game simulation.

3. **A Web UI** — Play against AI opponents, watch AIs battle each other, or play with friends. Includes real-time neural network visualization showing what the AI is "thinking".

---

## Demo

*Video coming soon*

---

# Game Simulator

The simulator (`_01_simulator/`) is a complete implementation of Beasty Bar (2-player only).

### What It Does

- **State Management**: Immutable game states with zones (hands, queue, heaven, hell)
- **Action Space**: 124 discrete actions covering all possible plays and ability choices
- **Card Abilities**: All 12 species implemented with their unique effects
- **Observation Encoding**: 988-dimensional vector representation for neural networks
- **Rewards**: Configurable reward shaping for RL training

### Why It Matters

Most board game AIs start with a simulator. This one is designed for RL from the ground up:
- Immutable states enable easy tree search and parallel simulation
- The observation encoding captures everything the AI needs to make decisions
- Action masking ensures the AI only considers legal moves

### Design Choices

- **Immutable states** over mutable: Safer for parallel self-play, easier to debug
- **Flat action space** over hierarchical: Simpler policy network, faster training
- **Dense observations** over sparse: Transformers handle redundancy well
- **2-player only**: The real game supports 2-4 players, but we're keeping it simple

### Structure

```
_01_simulator/
├── state.py          # Game state representation
├── engine.py         # Game loop and turn management
├── cards.py          # Species-specific ability handlers
├── action_space.py   # Action encoding/decoding
├── observations.py   # State → tensor conversion
├── rewards.py        # Reward calculation
├── simulate.py       # High-level game simulation
└── _cython/          # Cython acceleration (200x speedup)
```

### Cython Acceleration

The `_cython/` directory contains a complete reimplementation of the game engine in Cython for ~200x faster simulation. Production training uses `force_cython: true`—the Cython extension must be compiled or training will fail (no Python fallback).

**Current state:** Both Python and Cython implementations exist. The Python modules provide type definitions and single-game interfaces used throughout the codebase, while the Cython module provides batch acceleration required for training.

```bash
# Build Cython extension (required for training)
bash scripts/build_cython.sh
```

---

# Training Pipeline

The training code (`_03_training/`) implements PPO self-play with opponent diversity.

### What It Does

- **PPO (Proximal Policy Optimization)**: Stable policy gradient updates with clipping
- **GAE (Generalized Advantage Estimation)**: Variance reduction for value learning
- **Opponent Pool**: Mix of self-play, past checkpoints, random, and heuristic opponents
- **Adaptive Weighting**: Automatically adjusts opponent mix based on win rates

### Why Opponent Diversity?

I believe pure self-play causes policy collapse—the AI learns to exploit itself rather than play well generally. Mixing in diverse opponents (random, heuristic, past versions) should force the AI to stay robust. **Note:** I haven't actually tested pure self-play to confirm this causes collapse; this is a design choice based on reading about RL training. Testing pure self-play could be a good experiment.

### Design Choices

- **PPO over DQN/A2C**: Better sample efficiency, stable training
- **Opponent pool over pure self-play**: Prevents collapse (theoretically), maintains diversity
- **Cython acceleration**: 200x faster simulation enables more games per iteration
- **Async game generation**: Parallel workers to maximize GPU utilization

### Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Hidden dim | 256 | Balance between capacity and speed |
| Transformer layers | 4 | Enough depth for strategic reasoning |
| Games per iteration | 8,192 | Sufficient batch diversity |
| Learning rate | 0.0001 | Stable convergence |
| Entropy coefficient | 0.04→0.01 | Encourage exploration early, exploit later |

### Structure

```
_03_training/
├── trainer.py           # Main training loop
├── ppo.py               # PPO algorithm implementation
├── self_play.py         # Game generation and trajectory collection
├── opponent_pool.py     # Diverse opponent sampling
├── evaluation.py        # Model evaluation against baselines
├── checkpoint_manager.py # Save/load checkpoints
└── tracking.py          # W&B and console logging
```

### Usage

```bash
# Train from scratch
uv run scripts/train.py --config configs/iter600_to_1000.yaml

# Resume training
uv run scripts/train.py --resume checkpoints/iter_949.pt

# With W&B tracking
uv run scripts/train.py --config configs/iter600_to_1000.yaml --tracker wandb
```

### Known Issues

**GPU Utilization**: One of my main challenges is not fully utilizing compute resources. Some training runs only consume ~20% of the GPU, and I haven't figured out how to balance game generation speed with network forward passes to maximize throughput. If you have experience with this, I'd love to hear suggestions.

---

# Web UI

The UI (`_04_ui/`) lets you play and watch Beasty Bar games in the browser.

### What It Does

- **Play vs AI**: Challenge any of the trained agents
- **Watch AI vs AI**: See neural networks battle with real-time visualization
- **Multiplayer**: Play against friends locally
- **Visualization**: Watch neural network activations as the AI thinks

### Why It Matters

Training metrics only tell part of the story. Watching games reveals:
- Does the AI understand card abilities?
- Does it plan ahead or react randomly?
- Where does it make mistakes?

### Features

- **Multiple AI opponents**: Random, heuristic variants, neural network, tablebase-enhanced
- **Real-time updates**: WebSocket-based game state streaming
- **Activation visualization**: See which cards the AI is attending to
- **Battle mode**: Pit any two agents against each other

### Structure

```
_04_ui/
├── app.py              # FastAPI application
├── static/             # HTML/CSS/JS frontend
│   ├── index.html      # Main game interface
│   ├── visualizer.html # Neural network visualization
│   └── cards/          # Card images
└── visualization/      # Real-time activation capture
```

*Screenshots coming soon*

---

# Current Model

### Training History

| Checkpoint | Iterations | Games | Notes |
|------------|------------|-------|-------|
| `iter_600_final.pt` | 600 | ~5M | Stable baseline |
| `iter_949.pt` | 949 | ~15M | Best model, continued training |

Training done on RunPod A100 GPUs (~$300 spent on experiments so far, ~10 hours of actual training).

### Performance

Evaluated with 500 games per opponent (both sides), greedy action selection.

| Opponent | Win Rate | 95% CI | Margin |
|----------|----------|--------|--------|
| Random | 93.4% | [0.91, 0.95] | +7.58 |
| Heuristic | 76.8% | [0.73, 0.80] | +3.81 |
| Aggressive | 75.0% | [0.71, 0.79] | +3.64 |
| Defensive | 81.0% | [0.77, 0.84] | +4.46 |
| Queue | 75.6% | [0.72, 0.79] | +3.80 |
| Skunk | 75.6% | [0.72, 0.79] | +3.99 |
| Noisy | 75.6% | [0.72, 0.79] | +3.80 |
| Online | 70.2% | [0.66, 0.74] | +3.21 |
| Outcome Heuristic | 66.0% | [0.62, 0.70] | +2.54 |
| Distilled Outcome | 67.4% | [0.63, 0.71] | +2.75 |

**Overall**: 5000 games, 75.7% win rate, ~1379 ELO

### Architecture

- **Type**: Transformer policy-value network
- **Parameters**: ~1.3M
- **Input**: 988-dim observation vector
- **Output**: 124-dim action logits + scalar value

### Download

- Hugging Face (model): https://huggingface.co/shiptoday101/beastybar-ppo
- Hugging Face (tablebase): https://huggingface.co/datasets/shiptoday101/beastybar-tablebase

### Loading

```python
from _03_training.checkpoint_manager import load_for_inference
from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig

state_dict, config = load_for_inference("model_inference.pt")
network = BeastyBarNetwork(NetworkConfig.from_dict(config))
network.load_state_dict(state_dict)
```

---

# Next Steps

Ordered by priority. Each task is written to be delegated to an AI agent with clear scope.

### Immediate

- [x] **Migrate to Cython-only**: Removed fallback logic from `_01_simulator/_cython/__init__.py`. Cython extension is now required. Note: The Python modules (`engine.py`, `state.py`, etc.) remain because they provide type definitions and single-game interfaces used throughout the codebase. The Cython module provides batch acceleration for training. All 43 Cython tests pass.

- [x] **Evaluate latest model**: Evaluated `iter_949.pt` against all 10 opponents with 500 games each (both sides). Results: 75.7% overall win rate, ~1379 ELO. Full results saved to `evaluation_results_iter949.json` and Performance table updated above.

- [x] **Clean up W&B**: Deleted 19 crashed runs, kept 11 finished runs. Naming convention: `{experiment_type}_{hardware}_{description}` (e.g., `alphazero_v2_100sim`, `h200_iter600_to_1000`).

### Before Next Training Run

- [ ] **Maximize GPU/CPU utilization**: Currently training runs only use ~20% GPU. Investigate the bottleneck—is it game generation (CPU), batch transfer, or network inference? Profile the training loop, identify where time is spent, and optimize. Goal: get GPU utilization above 80% before starting the next training run. Document findings.

### Soon

- [x] **Refactor web UI**: Refactored `_04_ui/app.py` (1,456 lines) into proper FastAPI modules. New structure: `core/` (config, session, rate_limiter), `models/` (Pydantic requests), `services/` (ai, game, serializer), `api/` (game, actions, agents, stats, claude, visualization). All 31 endpoints preserved, app factory pattern implemented.

- [x] **Fix Hugging Face repo**: Updated model card with performance stats, code examples, and architecture details. Deleted test file `dev/tiny.pt`. Created README for tablebase dataset repo.

- [ ] **Continue training**: Push past iteration 1000 with lower learning rate schedule. Requires GPU utilization fix first.

### Later

- [ ] **Expand tablebase**: Generate 8-card tablebase using AWS high-CPU instance (see `docs/TABLEBASE_AWS_PLAN.md`). This will cover more endgame positions for stronger play.

- [ ] **Strategy guide**: After final training run, analyze what the AI learned—which cards it values, common patterns, mistakes it makes. Write up findings as documentation.

---

# Testing

```bash
uv run pytest tests/
```

# Evaluation

```bash
uv run scripts/evaluate.py --model checkpoints/iter_949.pt --opponents random,heuristic --games 200
```

---

# Links

- [Technical docs](docs/TECHNICAL.md)
- [API docs](docs/API.md)
- [Tablebase AWS plan](docs/TABLEBASE_AWS_PLAN.md)
- [Beasty Bar rules (PDF)](https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf)
- [Beasty Bar on Zoch Verlag](https://www.zoch-verlag.com/zoch_en/brands/beasty-bar/)

---

# License

MIT
