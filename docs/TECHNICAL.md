# Technical Reference

## Environment

Python >=3.10, uv or pip. Core deps: torch, numpy, fastapi.

## Modules

### _01_simulator — Game Engine

Deterministic, immutable state. Complete Beasty Bar rules.

| File | Purpose |
|------|---------|
| `state.py` | `Card`, `State`, `initial_state()` |
| `engine.py` | `legal_actions()`, `step()`, `is_terminal()`, `score()` |
| `cards.py` | Species mechanics (12 animals) |
| `action_space.py` | 124-action catalog |
| `observations.py` | 988-dim state encoding |
| `_cython/` | Optional 200x speedup |

### _02_agents — AI Players

| Agent | Purpose |
|-------|---------|
| `random_agent.py` | Baseline |
| `heuristic.py` | Rule-based material evaluation |
| `neural/` | Transformer policy-value network |
| `mcts/` | Tree search (experimental, not used in training) |

### _03_training — PPO Training

| File | Purpose |
|------|---------|
| `trainer.py` | Training orchestrator |
| `game_generator.py` | Self-play with opponent diversity |
| `opponent_pool.py` | Mixed opponent sampling |
| `ppo.py` | Proximal Policy Optimization |
| `evaluation.py` | Win rate with confidence intervals |

## Neural Network

Transformer policy-value network (~1.3M params).

```
Input (988) → Zone Encoders → Fusion → Policy (124) + Value [-1,1]
```

**Observation (988 dims):**
- Queue: 5×17 (ordered → TransformerEncoder)
- Bar/Hand: unordered → SetTransformer
- Scalars: 7 dims

**Output:** 124 action logits + value in [-1,1]

## Opponent Pool

Pure self-play collapses. Mix fixes it:

| Type | Weight |
|------|--------|
| Current network | 50% |
| Heuristic | 30% |
| Past checkpoints | 10% |
| Random | 10% |

Supports adaptive weighting (`use_adaptive_weights: true`) that adjusts opponent mix based on win rates, with a configurable minimum weight floor.

## Configuration

```yaml
# configs/iter600_to_1000.yaml
network_config:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4

ppo_config:
  learning_rate: 0.0001
  minibatch_size: 2048

games_per_iteration: 8192
total_iterations: 1000

opponent_config:
  current_weight: 0.5
  checkpoint_weight: 0.1
  random_weight: 0.1
  heuristic_weight: 0.3
```

## Performance

| Mode | Speed |
|------|-------|
| Pure Python | ~2ms/game |
| Cython | ~0.01ms/game |
