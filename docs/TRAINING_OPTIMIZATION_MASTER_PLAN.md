# Training Optimization Master Plan - Local Changes

**Goal:** Build the strongest possible Beasty Bar AI agent - 100% win rate against all opponents.

**Approach:** All-in on Cython for training and game generation. No Python fallback.

---

## Baseline Model: iter600

| File | Size | Purpose |
|------|------|---------|
| `checkpoints/v4/final.pt` | 521MB | Full checkpoint (resume training from here) |
| `checkpoints/v4/model_inference.pt` | 65MB | Inference only (deployment) |

**Current Performance (iter600):**

| Opponent | Win Rate |
|----------|----------|
| random | 93% |
| heuristic | 81% |
| aggressive | 75% |
| defensive | 78% |
| queue | 82% |
| skunk | 79% |
| noisy | 72% |
| online | 72% |
| outcome_heuristic | TBD |
| distilled_outcome | TBD |

**Target: 100% win rate against ALL opponents**

---

## All Opponents (10 total)

| Name | Class | Description |
|------|-------|-------------|
| `random` | RandomAgent | Random legal actions |
| `heuristic` | HeuristicAgent | Base material evaluator |
| `aggressive` | HeuristicAgent | High bar weight, aggression=0.8 |
| `defensive` | HeuristicAgent | Low aggression=0.2 |
| `queue` | HeuristicAgent | Prioritizes queue front control |
| `skunk` | HeuristicAgent | Skunk specialist |
| `noisy` | HeuristicAgent | Human-like bounded rationality |
| `online` | OnlineStrategies | Reactive counter-play |
| `outcome_heuristic` | OutcomeHeuristic | Forward simulation, hand-tuned |
| `distilled_outcome` | DistilledOutcomeHeuristic | Forward simulation, PPO-extracted |

---

## Architecture: All-In on Cython

**Key Decision:** Training uses Cython for ALL game generation, including opponent pool.

```
┌─────────────────────────────────────────────────────────┐
│                    CYTHON PATH ONLY                     │
├─────────────────────────────────────────────────────────┤
│  Self-play (neural vs neural)     → Cython vectorized   │
│  vs Random                        → Cython random action│
│  vs Checkpoint                    → Cython vectorized   │
│  vs Heuristic variants            → Cython + C→Python   │
└─────────────────────────────────────────────────────────┘
```

**No Python fallback** - if Cython can't handle it, we fix Cython.

---

## Adaptive Opponent Weighting

Instead of fixed weights, use **win-rate-based adaptive weighting**:

```python
# Concept: Play MORE against opponents we're losing to, LESS against those we beat
# But never drop any opponent to 0% (prevent overfitting)

class AdaptiveOpponentPool:
    """Weight opponents inversely to win rate, with minimum floor."""

    MIN_WEIGHT = 0.05  # Never drop below 5% for any opponent

    def compute_weights(self, win_rates: dict[str, float]) -> dict[str, float]:
        """
        win_rates: {"random": 0.93, "online": 0.72, ...}
        Returns: {"random": 0.07, "online": 0.28, ...} (normalized)
        """
        # Invert win rates: opponents we lose to get higher weight
        raw_weights = {}
        for opponent, win_rate in win_rates.items():
            # 100% win rate → 0.05 weight (minimum)
            # 50% win rate → 0.50 weight
            # 0% win rate → 1.0 weight (maximum focus)
            raw_weights[opponent] = max(1.0 - win_rate, self.MIN_WEIGHT)

        # Normalize to sum to 1.0
        total = sum(raw_weights.values())
        return {k: v / total for k, v in raw_weights.items()}
```

**Example with current win rates:**

| Opponent | Win Rate | Raw Weight | Normalized |
|----------|----------|------------|------------|
| random | 93% | 0.07 | 4% |
| online | 72% | 0.28 | 16% |
| noisy | 72% | 0.28 | 16% |
| aggressive | 75% | 0.25 | 14% |
| defensive | 78% | 0.22 | 13% |
| skunk | 79% | 0.21 | 12% |
| heuristic | 81% | 0.19 | 11% |
| queue | 82% | 0.18 | 10% |
| outcome_heuristic | TBD | TBD | TBD |
| distilled_outcome | TBD | TBD | TBD |

**Result:** Model focuses on `online`, `noisy`, `aggressive` (hardest opponents) while still playing all others.

---

## Local Code Changes (This Document)

### Task 1: Fix Async Game Generation

**File:** `_03_training/trainer.py`

**Problem:** Workers fail with `ModuleNotFoundError` - `sys.path` not inherited in `spawn` context.

**Changes:**
1. Add `pythonpath` parameter to `_game_generation_worker_process()`
2. Insert `sys.path` modification BEFORE any project imports
3. Pass pythonpath from `_start_game_workers()`

### Task 2: Add Entropy Scheduling

**Files:** `_03_training/trainer.py`, `_03_training/ppo.py`

**Add function:**
```python
def get_entropy_coef(
    iteration: int,
    total_iterations: int,
    start_coef: float = 0.04,
    end_coef: float = 0.01,
    decay_type: str = "linear",
) -> float:
    """Decay entropy coefficient over training."""
```

**Integrate into:** `_train_on_buffer()` - compute dynamic entropy coef per iteration

### Task 3: Add Temperature Scheduling

**File:** `_03_training/trainer.py`

**Add function:**
```python
def get_temperature(
    iteration: int,
    total_iterations: int,
    start_temp: float = 1.0,
    end_temp: float = 0.5,
    decay_type: str = "linear",
) -> float:
    """Decay self-play temperature over training."""
```

**Integrate into:** `_generate_self_play_games()` - use dynamic temperature

### Task 4: Add Adaptive Opponent Weighting

**File:** `_03_training/opponent_pool.py`

**Add class:**
```python
class AdaptiveOpponentPool(OpponentPool):
    """Opponent pool with win-rate-based adaptive weighting."""

    def update_weights_from_win_rates(self, win_rates: dict[str, float]) -> None:
        """Recompute opponent weights based on recent evaluation."""
```

**Integrate into:** `Trainer._run_evaluation()` - update weights after each eval

### Task 5: Port RandomAgent to Cython

**File:** `_01_simulator/_cython/_cython_core.pyx`

**Add function:**
```cython
cdef int select_random_action_nogil(GameState* state, int player, uint32_t* rng) noexcept nogil:
    """Select random legal action in pure C."""
```

**Integrate into:** `vectorized_env_cython.py` - use Cython random for opponent pool

### Task 6: Add C→Python State Conversion

**File:** `_01_simulator/_cython/_cython_core.pyx`

**Add function:**
```cython
def c_state_to_python(GameStateArray arr, int index) -> State:
    """Convert C GameState to Python State for heuristic agents."""
```

**Integrate into:** `vectorized_env_cython.py` - hybrid path for heuristic opponents

### Task 7: Batch Games by Opponent Type

**File:** `_03_training/game_generator.py`

**Add function:**
```python
def generate_games_batched_by_opponent(
    network,
    num_games: int,
    opponent_pool: AdaptiveOpponentPool,
) -> list[Transition]:
    """Generate games, batching by opponent type for efficiency."""
```

### Task 8: Add Gradient Norm Logging

**File:** `_03_training/trainer.py`

**Add in `_train_on_buffer()`:**
```python
# Before gradient clipping
grad_norm_pre = torch.nn.utils.clip_grad_norm_(params, float('inf'))
metrics["train/grad_norm_pre_clip"] = grad_norm_pre.item()

# After gradient clipping
grad_norm_post = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
metrics["train/grad_norm_post_clip"] = grad_norm_post.item()
```

### Task 9: Add All Opponents to Eval List

**File:** `_03_training/trainer.py` (TrainingConfig)

**Update default:**
```python
eval_opponents: list[str] = field(
    default_factory=lambda: [
        "random",
        "heuristic",
        "aggressive",
        "defensive",
        "queue",
        "skunk",
        "noisy",
        "online",
        "outcome_heuristic",
        "distilled_outcome",
    ]
)
```

### Task 10: Create Config File

**File:** `configs/iter600_to_1000.yaml`

**Content:** Configuration for resuming from iter600

### Task 11: Run Typecheck

**Command:** `mypy _03_training/`

**Fix any errors found**

### Task 12: Run Linter

**Command:** `ruff check _03_training/ --fix`

**Fix any remaining issues**

### Task 13: Run Tests

**Command:** `pytest _05_other/tests/ -v`

**Ensure all tests pass with new changes**

---

## Execution Order

```
1. Fix async pythonpath bug          (Task 1)
2. Add entropy scheduling            (Task 2)
3. Add temperature scheduling        (Task 3)
4. Add adaptive opponent weighting   (Task 4)
5. Port RandomAgent to Cython        (Task 5)
6. Add C→Python state conversion     (Task 6)
7. Batch games by opponent type      (Task 7)
8. Add gradient norm logging         (Task 8)
9. Add all opponents to eval         (Task 9)
10. Create config file               (Task 10)
11. Run typecheck and fix            (Task 11)
12. Run linter and fix               (Task 12)
13. Run tests                        (Task 13)
```

---

## Config for iter600 → iter1000

```yaml
# configs/iter600_to_1000.yaml
# Resume from iter600, run 400 more iterations
# ALL-IN ON CYTHON - no Python fallback

network_config:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.1
  species_embedding_dim: 64

ppo_config:
  learning_rate: 0.0001
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.04        # Starting value (will be scheduled)
  gamma: 0.99
  gae_lambda: 0.95
  ppo_epochs: 4
  minibatch_size: 2048
  max_grad_norm: 0.5
  normalize_advantages: true
  clip_value: true
  target_kl: 0.02

total_iterations: 1000      # 600 existing + 400 new
games_per_iteration: 8192
checkpoint_frequency: 50
eval_frequency: 25

# Schedules (iterations 600-1000)
entropy_schedule: "linear"
entropy_start: 0.04
entropy_end: 0.01
temperature_schedule: "linear"
temperature_start: 1.0
temperature_end: 0.5

# Opponent pool with ADAPTIVE weighting
use_opponent_pool: true
use_adaptive_weights: true  # NEW: win-rate-based weighting
min_opponent_weight: 0.05   # Never drop below 5%

opponent_config:
  # Initial weights (will be overridden by adaptive)
  current_weight: 0.5
  checkpoint_weight: 0.1
  random_weight: 0.1
  heuristic_weight: 0.3
  max_checkpoints: 10

use_heuristic_variants: true
use_exploit_patch: true
exploit_patch_interval: 100

# CYTHON ONLY - no Python fallback
force_cython: true
async_game_generation: true
async_num_workers: 32

buffer_size: 800000
seed: 42
device: "cuda"
experiment_name: "iter600_to_1000"

# ALL 10 OPPONENTS
eval_opponents:
  - random
  - heuristic
  - aggressive
  - defensive
  - queue
  - skunk
  - noisy
  - online
  - outcome_heuristic
  - distilled_outcome
```

---

## Success Criteria

| Opponent | Current | Target |
|----------|---------|--------|
| random | 93% | **100%** |
| heuristic | 81% | **100%** |
| aggressive | 75% | **100%** |
| defensive | 78% | **100%** |
| queue | 82% | **100%** |
| skunk | 79% | **100%** |
| noisy | 72% | **100%** |
| online | 72% | **100%** |
| outcome_heuristic | TBD | **100%** |
| distilled_outcome | TBD | **100%** |

---

## After Local Changes Complete

See `docs/RUNPOD_TRAINING_PLAN.md` for:
- RunPod instance setup
- Training execution
- Evaluation and validation
