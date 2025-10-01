# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beasty Bar is a board game simulator built to discover optimal strategies through reinforcement learning. The codebase implements full game rules, multiple agent strategies, PPO-based self-play training, tournament evaluation, and a web UI for human play.

## Development Commands

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run Web UI
```bash
uvicorn _04_ui.app:create_app --reload
# UI available at http://localhost:8000
```

### Run Tournaments
```bash
# Basic tournament between two agents
python -m _03_training.tournament first diego --games 25 --seed 2025

# Self-play evaluation with manifest
python -m _03_training.tournament self-play diego \
  --self-play-manifest _03_training/artifacts/champion.json \
  --games 200
```

### Self-Play Training
```bash
# From config file (recommended)
python -m _03_training.self_play --config _03_training/configs/self_play_local.json

# With CLI flags
python -m _03_training.self_play \
  --phase p3 \
  --seed 2025 \
  --opponent first --opponent random --opponent greedy --opponent diego \
  --total-steps 1000000 \
  --eval-frequency 50000 \
  --rollout-steps 2048 \
  --eval-games 200 \
  --eval-seed 4096
```

### Testing
```bash
# Run all tests
pytest _05_other/tests -ra

# Run specific test file
pytest _05_other/tests/test_simulate.py -v

# Run tests matching pattern
pytest _05_other/tests -k "test_action" -v
```

## Architecture

### Module Organization

The codebase uses underscore-prefixed numbered modules to enforce dependency ordering:

- **`_01_simulator/`**: Core game engine (deterministic, side-effect free)
  - `state.py`: Immutable game state using frozen dataclasses
  - `rules.py`: Species definitions and game constants
  - `engine.py`: State transitions and card effect resolution
  - `actions.py`: Action representation and validation
  - `action_space.py`: Canonical action enumeration for RL
  - `observations.py`: Feature extraction for neural networks
  - `rewards.py`: Reward shaping with margin weighting
  - `simulate.py`: High-level game simulation API

- **`_02_agents/`**: Strategy implementations inheriting from `Agent`
  - `base.py`: Abstract `Agent` class with `select_action` interface
  - `first.py`, `random_agent.py`, `greedy.py`: Simple baselines
  - `diego.py`: Heuristic agent with card-specific rules
  - `self_play_rl.py`: Neural network policy wrapper
  - `evaluation.py`: Evaluation and exploration utilities
  - **Export new agents via `_02_agents/__init__.py` for UI discovery**

- **`_03_training/`**: RL training and evaluation infrastructure
  - `ppo.py`: Proximal Policy Optimization with action masking
  - `rollout.py`: Self-play data collection with GAE
  - `models.py`: PolicyValueNet with orthogonal initialization
  - `encoders.py`: Observation → tensor encoding
  - `self_play.py`: Full training loop with checkpoint reservoir
  - `tournament.py`: Round-robin evaluation with Elo tracking
  - `policy_loader.py`: Manifest-driven checkpoint loading

- **`_04_ui/`**: FastAPI server + static HTML viewer
  - `app.py`: REST API exposing game state and agent selection
  - `static/`: Zero-build frontend assets

- **`_05_other/tests/`**: pytest suite with deterministic fixtures

### Key Architectural Patterns

#### Immutable State Design
All game state is represented with frozen dataclasses. State transitions return new `State` objects rather than mutating in place:

```python
@dataclass(frozen=True)
class State:
    seed: int
    turn: int
    active_player: int
    players: Tuple[PlayerState, ...]
    zones: Zones

# State updates return new instances
new_state = state.set_active_player(game_state, next_player, advance_turn=True)
```

#### Deterministic Simulation
All randomness is threaded through explicit `seed` parameters. Games are fully reproducible given a seed:

```python
state = simulate.new_game(seed=2025, starting_player=0)
# Same seed → same shuffle → same game
```

#### Action Masking for RL
Neural networks output logits over a canonical action space. Legal action masks prevent invalid moves:

```python
action_space_view = action_space.legal_action_space(game_state, player)
legal_indices = action_space_view.legal_indices  # [3, 7, 12, ...]
legal_logits = logits[legal_indices]  # Only sample from legal actions
```

#### Hidden Information Masking
Agents receive masked views of game state to prevent cheating:

```python
# Hide opponent's deck and hand
masked_view = state.mask_state_for_player(game_state, player_index)
action = agent.select_action(masked_view, legal_actions)
```

#### Manifest-Based Policy Loading
Self-play checkpoints use JSON manifests for tournament/UI integration:

```json
{
  "model": "_03_training.policy_loader:load_policy",
  "checkpoint": "checkpoints/step_120000.pt",
  "factoryKwargs": {"device": "cpu"},
  "exploration": {"temperature": 0.0, "epsilon": 0.05}
}
```

Manifests resolve checkpoints relative to manifest location, enabling portable artifact directories.

#### Checkpoint Reservoir
Self-play maintains a rolling window of recent checkpoints as opponents to prevent forgetting:

```python
reservoir = CheckpointReservoir(max_size=3, device=device)
reservoir.bootstrap_from_directory(artifacts.checkpoints)
# Active opponents = baselines + reservoir
```

### State Flow

```
simulate.new_game(seed)
  → State (immutable)
  → simulate.legal_actions(state, player)
  → Agent.select_action(state, legal)
  → Action
  → engine.step(state, action)
  → new State
  → ... → simulate.is_terminal(state)
  → simulate.score(state)
```

### Training Flow

```
collect_rollouts()
  → plays games with current policy vs opponents
  → records (obs, action, reward, value) transitions
  → computes GAE advantages
  → RolloutBatch

ppo_update()
  → mini-batch sampling
  → policy loss (clipped surrogate)
  → value loss (MSE)
  → entropy bonus
  → gradient clipping

Every N steps:
  → save checkpoint
  → add to reservoir
  → run evaluation tournament
  → compute Elo
  → maybe promote champion
```

## Important Implementation Details

### Agent Registration
New agents must be exported in `_02_agents/__init__.py` to appear in the UI agent dropdown and tournament CLI.

### Self-Play Artifacts
Training runs create structured artifact directories:
```
_03_training/artifacts/<run_id>/
├── run_manifest.json          # Full run config
├── champion.json               # Latest promoted policy manifest
├── checkpoints/
│   ├── step_50000.pt
│   └── step_100000.pt
├── metrics/
│   ├── step_*.json            # Per-iteration metrics
│   └── rolling_metrics.json   # Aggregated sliding window
└── eval/
    └── step_*_eval.json       # Tournament results
```

### Reward Shaping
Rewards use margin-weighted terminal scores with deterministic jitter to break ties:

```python
shaped_reward(state, margin_weight=0.25, jitter_scale=0.01)
# Returns: base_score + margin_weight * (score_delta) + jitter
```

### Action Space Encoding
The canonical action space enumerates all possible `(hand_index, params)` tuples. Agents receive legal action masks at each decision point to prevent invalid actions from being sampled.

### Starting Player Randomization
**CRITICAL BUG (identified in diegonext.md)**: Current rollout code always starts learner as P0. This creates first-player bias. Fix by randomizing `starting_player` in `collect_rollouts()`.

## Self-Play Training Configuration

Training accepts either CLI flags or JSON config files. CLI flags override config file values. Key hyperparameters:

- `rollout_steps`: Minimum learner decisions per PPO update (default: 2048)
- `reservoir_size`: Max checkpoints retained as self-play opponents (default: 3)
- `eval_games`: Games per opponent in evaluation tournaments (default: 200)
- `gamma`: Discount factor for returns (default: 0.99)
- `gae_lambda`: GAE smoothing parameter (default: 0.95)
- `margin_weight`: Weight for score margin in shaped rewards (default: 0.25)
- `promotion_min_win_rate`: Champion promotion threshold (default: 0.55)
- `promotion_min_elo_delta`: Minimum Elo gain for promotion (default: 25.0)

Use `--eval-games N` and `--eval-seed S` for reproducible evaluation. JSON configs mirror CLI flags via camelCase or snake_case keys.

## Game Rules Summary

Beasty Bar is a 2-player queue-management game. Players play animal cards with unique abilities into a shared queue (max 5 cards). When the queue fills, the front 4 cards enter the Beasty Bar (scoring zone) and earn points. The 5th card is sent to "That's It" (eliminated). Each species has special abilities that manipulate queue order or remove opponent cards.

See `_01_simulator/rules.py` for species definitions (strength, points, recurring/permanent flags) and `_01_simulator/engine.py` for card effect resolution logic.
