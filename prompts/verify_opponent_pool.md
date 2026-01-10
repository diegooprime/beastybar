# Verify Opponent Pool Implementation

## Context

A comprehensive opponent pool system was just implemented to break a training plateau (79% vs heuristic). The implementation adds:

1. **Parameterized Heuristic Agents** (`_02_agents/heuristic.py`)
2. **MCTS Opponent Support** (`_03_training/opponent_pool.py`)
3. **Opponent Statistics Tracking** (`_03_training/opponent_statistics.py`) - NEW FILE
4. **Exploiter Training** (`_03_training/exploiter_training.py`) - NEW FILE
5. **Exploit-Patch Cycle Manager** (`_03_training/exploit_patch_cycle.py`) - NEW FILE
6. **Trainer Integration** (`_03_training/trainer.py`, `_03_training/game_generator.py`)

## Your Task

**Rigorously verify this implementation.** Do not accept code that "looks right" - actually run it, test edge cases, and fix any issues you find.

### Phase 1: Static Analysis

Run these checks and fix ALL issues:

```bash
# Lint check - must pass with zero errors
.venv/bin/ruff check _02_agents/heuristic.py _03_training/opponent_pool.py _03_training/opponent_statistics.py _03_training/exploiter_training.py _03_training/exploit_patch_cycle.py _03_training/game_generator.py _03_training/trainer.py

# Type check - fix any type errors
.venv/bin/mypy _03_training/opponent_statistics.py _03_training/exploiter_training.py _03_training/exploit_patch_cycle.py --ignore-missing-imports
```

### Phase 2: Unit Tests

Write and run unit tests for each new module. Create test files if they don't exist.

#### 2.1 Test `opponent_statistics.py`

```python
# Test these scenarios:
# 1. OpponentStats win_rate returns 0.5 when no games played
# 2. OpponentStatsTracker.update() correctly increments counts
# 3. update_batch() handles empty list
# 4. compute_learning_weights() returns normalized weights summing to 1.0
# 5. compute_learning_weights() gives highest weight to 50% win rate opponents
# 6. Sliding window works (old results don't count after window_size games)
# 7. to_dict()/from_dict() round-trips correctly
```

#### 2.2 Test `opponent_pool.py` MCTS additions

```python
# Test these scenarios:
# 1. OpponentConfig with mcts_weight validates correctly (weights sum to 1.0)
# 2. MCTSOpponentConfig auto-generates name if not provided
# 3. create_default_mcts_configs() returns 6 distinct configs
# 4. OpponentPool.sample_opponent() returns MCTS type with correct probability
# 5. MCTS weight redistributes to CURRENT when no mcts_configs provided
# 6. set_mcts_network() stores network for MCTS agents
# 7. SampledOpponent.name returns correct format for MCTS
```

#### 2.3 Test `heuristic.py` parameterization

```python
# Test these scenarios:
# 1. HeuristicConfig defaults match original hardcoded values
# 2. HeuristicAgent with custom config uses those weights
# 3. aggression parameter biases action selection
# 4. noise_epsilon adds randomness (same state, different actions over N trials)
# 5. species_weights multipliers work correctly
# 6. create_heuristic_variants() returns 5 agents with distinct names
# 7. Default HeuristicAgent() is backward compatible
```

#### 2.4 Test `exploiter_training.py`

```python
# Test these scenarios:
# 1. ExploiterConfig has sensible defaults
# 2. ExploiterTrainer.__init__ loads target checkpoint correctly
# 3. ExploiterTrainer creates fresh network (not copy of target)
# 4. evaluate() returns float between 0 and 1
# 5. train() stops early if win_rate_threshold reached
# 6. train() respects max_iterations
# 7. ExploiterResult contains all required fields
```

#### 2.5 Test `exploit_patch_cycle.py`

```python
# Test these scenarios:
# 1. CycleConfig defaults are reasonable
# 2. record_win_rate() stores history correctly
# 3. detect_plateau() returns True when win rate is stagnant
# 4. detect_plateau() returns False when win rate is improving
# 5. should_start_cycle() triggers on plateau OR interval
# 6. to_dict()/from_dict() preserves all state
# 7. get_exploiter_opponents() returns correct format
```

### Phase 3: Integration Tests

Test the components working together:

#### 3.1 GameGenerator with new opponent types

```python
# Create GameGenerator with opponent pool containing MCTS
# Generate games and verify:
# 1. MCTS opponents are actually sampled (check opponent_name)
# 2. Only P0 transitions collected when using MCTS opponent
# 3. Stats tracker is updated with results
# 4. Win rate is returned correctly
```

#### 3.2 Trainer initialization with new config

```python
# Create Trainer with:
config = TrainingConfig(
    total_iterations=10,
    games_per_iteration=8,
    use_mcts_opponents=True,
    use_heuristic_variants=True,
    use_exploit_patch=True,
)
# Verify:
# 1. Trainer creates without errors
# 2. opponent_pool has MCTS network set
# 3. stats_tracker is initialized
# 4. exploit_patch_manager is initialized
# 5. Heuristic variants are available in pool
```

#### 3.3 Full training loop (short)

```python
# Run 5-10 iterations of actual training
# Verify:
# 1. No crashes during game generation
# 2. Different opponents are sampled (check logs)
# 3. Stats are being tracked
# 4. Checkpoints save/load correctly with new state
```

### Phase 4: Edge Cases & Error Handling

Test these explicitly:

1. **Empty opponent pool**: What happens if checkpoint_weight > 0 but no checkpoints exist?
2. **Zero MCTS configs**: What if mcts_weight > 0 but mcts_configs is empty?
3. **Invalid weights**: Config with weights summing to 0.5 should raise
4. **Missing network for MCTS**: Sampling MCTS before set_mcts_network() called
5. **Exploit cycle with no checkpoint**: run_cycle() with non-existent checkpoint path
6. **Stats tracker overflow**: 1 million updates shouldn't crash

### Phase 5: Performance Sanity Check

Verify no performance regressions:

```python
import time

# Baseline: Generate 100 games with standard self-play
# New: Generate 100 games with full opponent pool
# Difference should be < 50% slower (MCTS opponents are inherently slower)
```

## Deliverables

1. **Test report**: List of all tests run with pass/fail status
2. **Bug fixes**: Any code changes made to fix issues found
3. **Code quality**: All lint/type errors resolved
4. **Confirmation**: Statement that the implementation is production-ready OR list of remaining issues

## Important Guidelines

- **Actually run the code** - don't just read it and say "looks good"
- **Test with real data** - use actual game generation, not mocks
- **Check return types** - verify functions return what they claim
- **Verify state changes** - ensure mutable state is actually modified
- **Test boundaries** - empty lists, zero values, None parameters
- **Read error messages carefully** - don't skip stack traces

If you find bugs, FIX THEM. Don't just report them.

Use `.venv/bin/python` for all Python commands.
