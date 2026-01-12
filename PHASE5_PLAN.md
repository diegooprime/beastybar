# Phase 5: Training Optimization - Implementation Plan

**Date:** 2026-01-12

---

## Overview

Phase 5 implements advanced training optimizations to achieve 100% win rate against all opponents. The plan is derived from `docs/TRAINING_OPTIMIZATION_MASTER_PLAN.md`.

---

## Architecture

### Core Components

1. **Async Game Generation Fix** (`_03_training/trainer.py`)
   - Pass `pythonpath` to worker processes
   - Required because 'spawn' context doesn't inherit sys.path

2. **Entropy Scheduling** (`_03_training/trainer.py`)
   - `get_entropy_coef()` function
   - Linear/cosine decay from exploration (0.04) to exploitation (0.01)

3. **Temperature Scheduling** (`_03_training/trainer.py`)
   - `get_temperature()` function
   - Decay from standard softmax (1.0) to sharper (0.5)

4. **Adaptive Opponent Weighting** (`_03_training/opponent_pool.py`)
   - `AdaptiveOpponentPool` class
   - Win-rate-based weighting (lose more = higher sampling weight)
   - MIN_WEIGHT floor (5%) prevents dropping any opponent

5. **Cython Random Agent** (`_01_simulator/_cython/_cython_core.pyx`)
   - `select_random_action_nogil()` - GIL-free random action selection
   - `select_random_action()` - Python wrapper

6. **C to Python State Conversion** (`_01_simulator/_cython/_cython_core.pyx`)
   - `c_state_to_python()` - Convert C GameState to Python State
   - Enables hybrid path for heuristic opponents

7. **Batched Game Generation** (`_03_training/game_generator.py`)
   - `generate_games_batched_by_opponent()` function
   - Batches games by opponent type for efficiency

8. **Gradient Norm Logging** (`_03_training/trainer.py`)
   - Pre-clip and post-clip gradient norm tracking
   - Logged as `train/grad_norm_pre_clip` and `train/grad_norm_post_clip`

9. **Full Opponent Evaluation** (`_03_training/trainer.py`)
   - All 10 opponents in default `eval_opponents` list
   - Comprehensive performance tracking

10. **Config File** (`configs/iter600_to_1000.yaml`)
    - Full configuration for resuming training from iter600
    - All scheduling and adaptive weighting options

---

## Implementation Details

### Entropy Scheduling

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

### Temperature Scheduling

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

### Adaptive Opponent Weighting

```python
class AdaptiveOpponentPool(OpponentPool):
    """Weight opponents inversely to win rate, with minimum floor."""

    MIN_WEIGHT = 0.05

    def update_weights_from_win_rates(self, win_rates: dict[str, float]) -> None:
        """Recompute opponent weights based on recent evaluation."""
```

---

## File Structure

```
_03_training/
├── trainer.py             # Entropy/temperature scheduling, async fix, grad norm logging
├── opponent_pool.py       # AdaptiveOpponentPool
├── game_generator.py      # Batched game generation

_01_simulator/_cython/
└── _cython_core.pyx       # select_random_action_nogil, c_state_to_python

configs/
└── iter600_to_1000.yaml   # Training configuration
```

---

## Success Criteria

- [x] All 13 implementation tasks complete
- [x] Async pythonpath bug fixed
- [x] Entropy/temperature scheduling working
- [x] Adaptive opponent weighting functional
- [x] Cython random agent implemented
- [x] State conversion for hybrid path
- [x] Batched game generation available
- [x] Gradient norm logging active
- [x] All 10 opponents in eval
- [x] Config file created
- [x] Typecheck passing (path resolution warning only, no type errors)
- [x] Linter clean (fixed unused imports, 28 style warnings remain)
- [x] Tests passing (529/529)

---

## Dependencies

- Phase 4 (Population Training) - COMPLETE
- AlphaZero trainer (Phase 2) - COMPLETE
- Network architecture (Phase 3) - COMPLETE

---

## Scope Boundaries

**Files verified/implemented:**
- `_03_training/trainer.py` - scheduling, async fix, grad norm
- `_03_training/opponent_pool.py` - adaptive weighting
- `_03_training/game_generator.py` - batched generation
- `_01_simulator/_cython/_cython_core.pyx` - Cython extensions
- `configs/iter600_to_1000.yaml` - configuration

**Will validate:**
- Type annotations (mypy)
- Code style (ruff)
- Test coverage (pytest)
