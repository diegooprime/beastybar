# Technical Reference

Complete technical documentation for Beasty Bar AI.

## Environment

- **Python**: >=3.10
- **Package manager**: uv (recommended) or pip
- **Core deps**: fastapi, uvicorn, numpy, torch
- **Optional**: cython (200x game simulation speedup)
- **Dev**: pytest, ruff, mypy

---

## Module Architecture

### _01_simulator — Game Engine

Deterministic, side-effect-free game simulator with complete Beasty Bar rules.

| File | Purpose | Key Exports |
|------|---------|-------------|
| `state.py` | Immutable game state | `Card`, `State`, `initial_state()` |
| `engine.py` | Game execution | `legal_actions()`, `step()`, `is_terminal()`, `score()` |
| `cards.py` | Species mechanics | `resolve_play()`, `process_recurring()` |
| `action_space.py` | 124-action catalog | `action_index()`, `legal_action_mask()` |
| `observations.py` | 988-dim encoding | `state_to_tensor()` |
| `_cython/` | Optional acceleration | ~200x speedup |

**Turn resolution order:**
1. Play card at back of queue
2. Execute on-play effect
3. Process recurring effects (Hippo, Crocodile, Giraffe)
4. Five-animal check (front 2 enter bar, last 1 bounced)
5. Draw replacement card

**Species mechanics:**
| Species | Strength | Effect |
|---------|----------|--------|
| Lion | 12 | Scares monkeys, moves to front (or to THAT'S IT if lion exists) |
| Hippo | 11 | Recurring: passes weaker animals toward gate, blocked by zebra |
| Crocodile | 10 | Recurring: eats weaker animals ahead, blocked by zebra |
| Snake | 9 | Sorts queue by strength descending |
| Giraffe | 8 | Recurring: jumps over one weaker animal per turn |
| Zebra | 7 | Permanent: blocks hippo/croc for itself and animals in front |
| Seal | 6 | Reverses entire queue |
| Chameleon | 5 | Clones another species' effect temporarily |
| Monkey | 4 | If 2+: bounces hippos/crocs, monkeys move to front |
| Kangaroo | 3 | Jumps over 1-2 animals at back |
| Parrot | 2 | Bounces any one animal |
| Skunk | 1 | Expels highest and second-highest strength animals |

### _02_agents — AI Players

| Component | Purpose |
|-----------|---------|
| `random_agent.py` | Uniform random selection (baseline) |
| `heuristic.py` | Rule-based material evaluation |
| `mcts/` | Monte Carlo Tree Search with neural guidance |
| `neural/` | Transformer policy-value network |

**Heuristic weights:**
```python
bar_weight = 2.0        # Cards in Beasty Bar
queue_front_weight = 1.1
queue_back_weight = 0.3
thats_it_weight = -0.5  # Penalty for bounced cards
hand_weight = 0.1
```

**Heuristic variants:** `create_heuristic_variants()` creates 6 variants (aggressive, defensive, bar-focused, queue-controller, skunk-specialist, online-strategies).

### _03_training — Training Infrastructure

| File | Purpose |
|------|---------|
| `trainer.py` | PPO training orchestrator |
| `mcts_trainer.py` | AlphaZero-style MCTS training |
| `game_generator.py` | Self-play with opponent diversity |
| `opponent_pool.py` | Mixed opponent sampling |
| `evaluation.py` | Win rate evaluation with confidence intervals |
| `ppo.py` | Proximal Policy Optimization (GAE, clipped objective) |
| `checkpoint_manager.py` | Model persistence |

### _04_ui — Web Interface

FastAPI REST API with HTML/JS frontend.

**Endpoints:** `/new_game`, `/legal_moves`, `/play`, `/history`

**Features:** AI opponent selection, rate limiting (60 req/min), WebSocket support

---

## Neural Network

**Architecture:** Transformer-based policy-value network (~1.3M parameters)

| Layer | Details |
|-------|---------|
| Input | 988-dim observation |
| Card encoder | Species embedding (64-dim) + feature projection |
| Zone encoders | TransformerEncoder (queue, ordered) / SetTransformer (bar/hand, unordered) |
| Fusion | 3x FusionBlock layers |
| Policy head | Linear → GELU → Linear → 124 logits |
| Value head | Linear → GELU → Linear → Tanh → [-1, 1] |

**Observation structure (988 dims):**
- Queue: 5 slots × 17 features
- Beasty Bar: 24 slots × 17 features
- That's It: 24 slots × 17 features
- Own hand: 4 cards × 17 features
- Opponent hand: 4 slots × 3 features (masked)
- Scalars: 7 dims (turn, active player, counts)

**Action space:** 124 actions = 4 hand slots × 31 max parameters

---

## Opponent Pool

Pure self-play causes collapse. The opponent pool provides diversity:

| Type | Weight | Purpose |
|------|--------|---------|
| Current network | 60% | Main learning signal |
| Past checkpoints | 20% | Prevent catastrophic forgetting |
| Random agent | 10% | Baseline calibration |
| Heuristic agent | 10% | Quality anchor |

Checkpoints added every 20 iterations (max 10 kept).

---

## Configuration Reference

### Network

```yaml
network_config:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.1
  species_embedding_dim: 64
```

### MCTS

```yaml
mcts_config:
  num_simulations: 200     # Search depth
  c_puct: 2.0              # Exploration constant
  temperature: 1.0         # Action sampling
  temperature_drop_move: 30
  final_temperature: 0.25
  dirichlet_alpha: 0.5     # Root noise
  dirichlet_epsilon: 0.4
  batch_size: 32           # BatchMCTS leaf evaluation
  virtual_loss: 3.0        # Parallel MCTS diversity
```

### Training

```yaml
learning_rate: 0.00005
entropy_bonus_weight: 0.05
games_per_iteration: 128
batch_size: 512
epochs_per_iteration: 4
total_iterations: 200
```

### Opponent Pool

```yaml
opponent_pool:
  current_weight: 0.6
  checkpoint_weight: 0.2
  random_weight: 0.1
  heuristic_weight: 0.1
  checkpoint_every: 20
  max_checkpoints: 10
```

---

## Performance

### Game Simulation

| Mode | Speed |
|------|-------|
| Pure Python | ~2ms/game |
| Cython | ~0.01ms/game (200x) |

### MCTS

| Mode | Speed |
|------|-------|
| Sequential | ~0.03 games/sec |
| BatchMCTS (batch=16) | ~0.5 games/sec (10x) |

---

## Key Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| Policy loss | Decreasing | Network matching MCTS |
| Value loss | Decreasing | Predicting outcomes |
| Entropy | 0.5–2.0 | Healthy exploration |
| Win vs random | >50% | Basic competence |
| Win vs heuristic | >40% | Strategic understanding |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | PPO training |
| `train_mcts.py` | MCTS training |
| `evaluate.py` | Model evaluation |
| `play.py` | Interactive play |
| `benchmark_cython.py` | Performance testing |
| `build_cython.sh` | Build Cython extensions |

---

## Design Principles

1. **Immutability** — All state mutations return new objects
2. **Determinism** — Seeds enable reproducible replay
3. **Fixed-size tensors** — 988-dim observation, 124-dim action space
4. **Opponent diversity** — Mixed opponents prevent self-play collapse
5. **Modular agents** — Interchangeable via common interface
