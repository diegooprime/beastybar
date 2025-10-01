# Beasty Bar - Ten-Step Action Plan

**Version**: 1.0
**Date**: 2025-10-01
**Owner**: Technical Program Lead

---

## Overview

This document provides a **dependency-aware, actionable execution plan** for the next 10 critical tasks to accelerate delivery of the Beasty Bar RL platform. Each step includes clear objectives, scope, success criteria, and verification procedures.

**Sequencing Note**: Steps are ordered by dependency; some can be parallelized (noted explicitly).

---

## Step 1: Fix First-Player Bias in Rollouts

### Objective
Randomize starting player in `collect_rollouts()` to eliminate systemic P0 advantage in learned policies.

### Rationale
Current training always starts learner as P0 (`_03_training/rollout.py:77`), creating exploitable bias. CLAUDE.md acknowledges this as **CRITICAL BUG** but it remains unfixed. All existing checkpoints are scientifically invalid.

### Scope
**Files Modified**:
- `_03_training/rollout.py` (line 77)
- `_05_other/tests/test_self_play_training.py` (new test)

**Modules Touched**:
- Training infrastructure (rollout collection)
- Test suite

**Cross-Team Impact**:
- Invalidates all artifacts in `_03_training/artifacts/`
- Researchers must archive compromised checkpoints

### Task Breakdown

**Inputs**:
- Analysis from CLAUDE.md "CRITICAL BUG" section
- Current rollout code inspection

**Implementation**:
```python
# _03_training/rollout.py (lines 72-78)
# BEFORE:
while steps < config.min_steps:
    opponent_factory = rng.choice(opponent_factories)
    opponent = opponent_factory()
    game_seed = rng.randrange(0, 10_000_000)
    current = simulate.new_game(game_seed, starting_player=0)  # ← BUG

# AFTER:
while steps < config.min_steps:
    opponent_factory = rng.choice(opponent_factories)
    opponent = opponent_factory()
    game_seed = rng.randrange(0, 10_000_000)
    starting_player = rng.choice([0, 1])  # ← FIX: randomize
    current = simulate.new_game(game_seed, starting_player=starting_player)
```

**Outputs**:
- Corrected rollout logic with balanced P0/P1 distribution
- Unit test validating both starting positions

### Owner
**Primary**: ML Engineer
**Reviewer**: Simulation Engineer (for state.py compatibility check)

### Dependencies
**Blocked By**: None (can start immediately)
**Blocks**: Steps 2, 4 (checkpoint archival and retraining depend on this fix)

### Verification Procedure

1. **Unit Test** (`_05_other/tests/test_self_play_training.py`):
```python
def test_rollout_alternates_starting_player():
    """Verify learner plays as both P0 and P1 across rollouts."""
    # Collect 100 episodes with fixed seed
    # Count starting_player distribution
    # Assert 40 ≤ P0_count ≤ 60 (within binomial confidence interval)
```

2. **Integration Test**:
   - Run 10k-step training with `--seed 2025`
   - Parse `metrics/step_*.json` for P0/P1 episode counts
   - Verify 48% ≤ P0_rate ≤ 52%

3. **Tournament Validation**:
   - Train new checkpoint for 20k steps
   - Evaluate vs diego with `alternate_start=True` in tournament
   - Measure P0 vs P1 win rate delta; assert |delta| < 5%

### Success Criteria
- ✅ `pytest _05_other/tests/test_self_play_training.py::test_rollout_alternates_starting_player` passes
- ✅ Training metrics log confirms 50% ± 2% P0 starts over 100+ episodes
- ✅ New checkpoint shows balanced P0/P1 performance in tournament (delta <5%)

### Estimated Effort
**Time**: 2 hours (1hr implementation + 1hr testing)
**Complexity**: Low

---

## Step 2: Invalidate Compromised Checkpoints

### Objective
Archive all existing training artifacts as scientifically invalid due to P0 bias; prevent accidental reuse.

### Rationale
Checkpoints in `_03_training/artifacts/` were trained with hardcoded `starting_player=0`. They exhibit exploitable first-player patterns and should not be used for research conclusions or further training.

### Scope
**Files Modified**:
- `.gitignore` (add exclusion rules)
- `_03_training/artifacts_invalid/INVALIDATED.md` (new documentation)

**Directories Affected**:
- `_03_training/artifacts/` → renamed to `_03_training/artifacts_invalid/`

**Cross-Team Impact**:
- Researchers relying on existing champions must retrain
- UI telemetry endpoints will show no active runs until Step 4 completes

### Task Breakdown

**Inputs**:
- Existing artifact directories: `smoke_cli/`, `soak_cli/`, `loop_cli/`
- Analysis from PROJECT_HEALTH.md showing loop_cli Elo=592 failure

**Execution Steps**:
```bash
# 1. Archive existing artifacts
mv _03_training/artifacts _03_training/artifacts_invalid

# 2. Recreate empty artifacts directory
mkdir -p _03_training/artifacts

# 3. Create documentation
cat > _03_training/artifacts_invalid/INVALIDATED.md << 'EOF'
# Invalidated Training Artifacts

**Date Invalidated**: 2025-10-01
**Reason**: First-player bias bug in rollout collection

## Summary
All checkpoints in this directory were trained with `starting_player=0` hardcoded
in `_03_training/rollout.py:77`. This creates systematic P0 advantage in learned
policies, making them scientifically invalid.

## Affected Runs
- smoke_cli: 2,049 steps
- soak_cli: 20,523 steps
- loop_cli: 100,575 steps (Elo=592, catastrophic failure)

## Recommendation
DO NOT use these checkpoints for research or production. They are preserved for
historical reference only. Retrain with corrected rollout code (post-2025-10-01).

## Fix Applied
See commit SHA: [to be filled after merge]
EOF

# 4. Update .gitignore
cat >> .gitignore << 'EOF'

# Training artifacts (large checkpoint files)
_03_training/artifacts*/
*.pt
*.pth
EOF
```

**Outputs**:
- Archived artifacts in clearly marked invalid directory
- `.gitignore` prevents future checkpoint bloat
- Documentation explains invalidation reason

### Owner
**Primary**: ML Engineer
**Reviewer**: Technical Lead (for communication to stakeholders)

### Dependencies
**Blocked By**: Step 1 (fix must be merged before archiving)
**Blocks**: Step 4 (retraining should use clean artifact directory)

### Verification Procedure

1. **Directory Structure**:
```bash
tree -L 2 _03_training/
# Expected:
# _03_training/
# ├── artifacts/          (empty)
# ├── artifacts_invalid/
# │   ├── INVALIDATED.md
# │   ├── smoke_cli/
# │   ├── soak_cli/
# │   └── loop_cli/
```

2. **UI Validation**:
   - Start UI: `uvicorn _04_ui.app:create_app --reload`
   - Visit `/api/telemetry/runs`
   - Assert response: `{"runs": []}`

3. **Git Status**:
```bash
git status | grep artifacts
# Should show no tracked .pt files
```

### Success Criteria
- ✅ `_03_training/artifacts/` exists and is empty
- ✅ `_03_training/artifacts_invalid/INVALIDATED.md` documents reason
- ✅ `.gitignore` contains `artifacts*/` and `*.pt` rules
- ✅ UI telemetry returns empty run list
- ✅ No checkpoint files tracked in git

### Estimated Effort
**Time**: 30 minutes
**Complexity**: Trivial

---

## Step 3: Add CI Pipeline with Smoke Tests

### Objective
Automate pytest suite + deterministic tournament validation on every PR to prevent simulator regressions.

### Rationale
No existing CI means changes to `_01_simulator/` can break agent compatibility without detection. A tournament smoke test with locked seed provides deterministic regression detection.

### Scope
**Files Modified**:
- `.github/workflows/ci.yml` (new)
- `README.md` (add CI badge)
- `pyproject.toml` (validate pytest config)

**Modules Touched**:
- CI/CD infrastructure
- Documentation

**Cross-Team Impact**:
- All developers must ensure PRs pass CI before merge
- Merge velocity may decrease initially (offset by reduced debugging)

### Task Breakdown

**Implementation**:

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .
          pip install pytest

      - name: Run pytest
        run: pytest _05_other/tests -ra

      - name: Run deterministic tournament smoke test
        run: |
          python -m _03_training.tournament first diego \
            --games 10 --seed 2025 > tournament_output.txt

          # Validate expected outcome (update after determining ground truth)
          # grep "first.*5.*wins" tournament_output.txt
```

`README.md` addition:
```markdown
[![CI](https://github.com/USER/beastybar/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/beastybar/actions/workflows/ci.yml)
```

**Outputs**:
- Automated testing on every PR/push to main
- CI badge in README showing build status
- Deterministic tournament baseline for regression detection

### Owner
**Primary**: Full-Stack Engineer
**Reviewer**: ML Engineer (for tournament command validation)

### Dependencies
**Blocked By**: None (can run in parallel with Step 1-2)
**Blocks**: None (but improves confidence for all future steps)

### Verification Procedure

1. **Initial Setup**:
   - Create PR with `.github/workflows/ci.yml`
   - Verify workflow triggers on PR creation
   - Check Actions tab shows job running

2. **Test Validation**:
   - Intentionally break a test (e.g., modify `_01_simulator/state.py` incorrectly)
   - Push to PR branch
   - Assert CI fails with clear error message

3. **Performance Check**:
   - Measure CI runtime end-to-end
   - Assert total time <10 minutes (target: <5 minutes)

4. **Tournament Determinism**:
```bash
# Run tournament 3 times with same seed
for i in {1..3}; do
  python -m _03_training.tournament first diego --games 10 --seed 2025 \
    | tee run_$i.txt
done

# Compare outputs
diff run_1.txt run_2.txt  # Should be identical
diff run_2.txt run_3.txt  # Should be identical
```

### Success Criteria
- ✅ CI runs on every PR and push to main
- ✅ All pytest tests pass in CI environment
- ✅ Tournament smoke test produces deterministic output
- ✅ CI runtime <10 minutes
- ✅ CI badge added to README and shows green

### Estimated Effort
**Time**: 4 hours (2hr GitHub Actions setup + 2hr tournament baseline determination)
**Complexity**: Low-Medium

---

## Step 4: Retrain Baseline Champion (200k steps)

### Objective
Produce first scientifically valid checkpoint with corrected starting-player randomization and sufficient training budget.

### Rationale
Establishes new baseline for all future research. Validates that training pipeline fixes (Step 1) actually improve policy strength. 200k steps chosen based on literature for similar game complexity.

### Scope
**Files Modified**:
- `_03_training/configs/baseline_v2.json` (new config)
- Training artifacts created in `_03_training/artifacts/baseline_v2/`

**Modules Touched**:
- Training infrastructure
- Evaluation pipeline
- Checkpoint management

**Cross-Team Impact**:
- Requires GPU access (8-12 hours wall-clock time)
- Blocks research work until baseline available

### Task Breakdown

**Configuration** (`_03_training/configs/baseline_v2.json`):
```json
{
  "phase": "baseline_v2",
  "seed": 2025,
  "opponents": ["first", "random", "greedy", "diego"],
  "totalSteps": 200000,
  "evalFrequency": 40000,
  "rolloutSteps": 2048,
  "reservoirSize": 3,
  "evalGames": 500,
  "evalSeed": 4096,
  "promotionMinWinRate": 0.55,
  "promotionMinEloDelta": 25.0,
  "learningRate": 0.0003,
  "ppoEpochs": 4,
  "ppoBatchSize": 512,
  "gamma": 0.99,
  "gaeLambda": 0.95,
  "marginWeight": 0.25,
  "device": "auto"
}
```

**Execution**:
```bash
# Start training
python -m _03_training.self_play \
  --config _03_training/configs/baseline_v2.json \
  | tee baseline_v2_log.txt

# Monitor progress (in separate terminal)
watch -n 60 'ls -lh _03_training/artifacts/baseline_v2/checkpoints/'
```

**Outputs**:
- Checkpoint sequence: `step_40000.pt`, `step_80000.pt`, ..., `step_200000.pt`
- Evaluation JSONs at each checkpoint
- Promoted champion in `champion.json` if criteria met
- Training metrics in `metrics/rolling_metrics.json`

### Owner
**Primary**: ML Engineer
**Support**: DevOps (for GPU provisioning if needed)

### Dependencies
**Blocked By**: Steps 1 & 2 (fix + archival must complete first)
**Blocks**: Steps 5, 6, 8 (need valid checkpoint for analysis)

### Verification Procedure

1. **Pre-Flight Checks**:
```bash
# Validate config
python -c "
import json
from pathlib import Path
cfg = json.loads(Path('_03_training/configs/baseline_v2.json').read_text())
assert cfg['totalSteps'] == 200000
assert 'first' in cfg['opponents']
print('Config validated')
"

# Confirm P0 fix merged
grep -n "starting_player = rng.choice" _03_training/rollout.py
```

2. **Training Monitoring**:
   - Check `metrics/step_*.json` every 10k steps
   - Verify `avgEpisodeReward` increasing trend
   - Assert `meanEntropy` decreasing but >1.0 (exploration maintained)
   - Confirm P0 episode fraction ≈ 50% in rolling metrics

3. **Champion Evaluation**:
```bash
# After training completes, run extended tournament
python -m _03_training.tournament self-play diego \
  --self-play-manifest _03_training/artifacts/baseline_v2/champion.json \
  --games 1000 --seed 9999

# Expected: win rate ≥ 0.55, p-value < 0.05 (binomial test)
```

4. **Statistical Validation**:
```python
from scipy.stats import binomtest
wins = 570  # Example from tournament
games = 1000
result = binomtest(wins, games, 0.5, alternative='greater')
assert result.pvalue < 0.05, f"Win rate not significant: p={result.pvalue}"
```

### Success Criteria
- ✅ Training completes 200k steps without crashes
- ✅ Champion promoted with Elo ≥1550 (vs baseline 1500)
- ✅ Win rate vs diego ≥55% over 500+ games with p<0.05
- ✅ P0/P1 starting player balance: 48-52%
- ✅ Final entropy >1.0 (policy not collapsed)
- ✅ `champion.json` manifest valid and loadable

### Estimated Effort
**Time**: 12 hours (8-10hr wall-clock GPU time + 2hr monitoring/validation)
**Complexity**: Medium (requires GPU access and patience)

---

## Step 5: Expand Evaluation Metrics Suite

### Objective
Track opening diversity, card usage heatmaps, and per-species win contributions to enable deeper policy analysis.

### Rationale
Current evaluation outputs only Elo and win rate. These are insufficient for diagnosing specific policy weaknesses (e.g., "Is the agent learning to use Parrot effectively?" or "Does it overfit to specific opening sequences?").

### Scope
**Files Modified**:
- `_03_training/tournament.py` (add metric collection)
- `_03_training/self_play.py` (expand `step_*_eval.json` schema)

**New Metrics**:
1. **Opening Diversity**: Shannon entropy of first 3 cards played
2. **Card Usage Heatmap**: 12 species × 2 players → play frequency
3. **Species Win Contribution**: Per-species win rate when entering Beasty Bar

**Cross-Team Impact**:
- Evaluation runtime increases ~10% (additional stat collection)
- Downstream analysis tools (notebooks, dashboards) gain richer data

### Task Breakdown

**Implementation** (`_03_training/tournament.py`):
```python
def _compute_opening_diversity(game_history: List[Action]) -> float:
    """Shannon entropy of first 3 unique cards played."""
    from collections import Counter
    import math

    first_three = [action.card.species for action in game_history[:3]]
    if not first_three:
        return 0.0

    counts = Counter(first_three)
    total = sum(counts.values())
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return entropy

def _build_card_usage_heatmap(games: List[GameRecord]) -> Dict:
    """Count species play frequency by player."""
    from collections import defaultdict

    heatmap = defaultdict(lambda: [0, 0])  # species -> [P0_count, P1_count]

    for game in games:
        for action in game.history:
            species = action.card.species
            player = action.player
            heatmap[species][player] += 1

    return dict(heatmap)

def _species_win_contributions(games: List[GameRecord]) -> Dict:
    """Win rate when each species enters Beasty Bar."""
    from collections import defaultdict

    species_outcomes = defaultdict(lambda: {'wins': 0, 'total': 0})

    for game in games:
        winner = game.winner
        for card in game.final_state.zones.beasty_bar:
            if card.owner == winner:
                species_outcomes[card.species]['wins'] += 1
            species_outcomes[card.species]['total'] += 1

    return {
        species: stats['wins'] / max(stats['total'], 1)
        for species, stats in species_outcomes.items()
    }
```

**Updated Schema** (`step_*_eval.json`):
```json
{
  "timestamp": "...",
  "step": 200000,
  "opponents": [...],
  "finalElo": 1580,

  "extendedMetrics": {
    "openingDiversity": {
      "entropy": 2.73,
      "uniqueOpenings": 87,
      "mostCommonOpening": ["lion", "snake", "giraffe"]
    },
    "cardUsageHeatmap": {
      "lion": [42, 38],
      "snake": [40, 41],
      ...
    },
    "speciesWinContributions": {
      "lion": 0.68,
      "snake": 0.61,
      "parrot": 0.23,
      ...
    }
  }
}
```

**Outputs**:
- Enhanced evaluation JSONs with new metrics
- Validation that heatmap sums match game counts
- Baseline entropy measurement for healthy policies

### Owner
**Primary**: ML Engineer
**Reviewer**: Simulation Engineer (for stats validation)

### Dependencies
**Blocked By**: Step 4 (need valid checkpoint to evaluate)
**Blocks**: None (enables Step 8 ablation analysis)

### Verification Procedure

1. **Unit Tests**:
```python
def test_opening_diversity_calculation():
    # Hand-crafted game history
    actions = [
        Action(card=Card(species="lion", ...)),
        Action(card=Card(species="lion", ...)),
        Action(card=Card(species="snake", ...)),
    ]
    entropy = _compute_opening_diversity(actions)
    # Expected: -2*(1/3 * log2(1/3)) - 1*(1/3 * log2(1/3)) ≈ 0.918
    assert abs(entropy - 0.918) < 0.01
```

2. **Integration Test**:
   - Run 100-game tournament with metrics enabled
   - Validate `extendedMetrics` present in eval JSON
   - Check heatmap sums: `sum(heatmap.values())` ≈ `games * avg_cards_per_game`

3. **Sanity Checks**:
   - Opening diversity: 0 < entropy < log2(12) ≈ 3.58
   - Card heatmap: All 12 species represented (or 0 if not played)
   - Species contributions: 0 ≤ win_rate ≤ 1.0 for all species

### Success Criteria
- ✅ `step_200000_eval.json` contains `extendedMetrics` field
- ✅ Opening entropy ≥2.0 (indicates healthy diversity)
- ✅ Heatmap includes all 12 species
- ✅ Species contributions sum sensibly (high-strength cards higher win contribution)
- ✅ Evaluation runtime overhead <10%

### Estimated Effort
**Time**: 6 hours (4hr implementation + 2hr testing/validation)
**Complexity**: Medium

---

## Step 6: Implement Hyperparameter Sweep Framework

### Objective
Automate grid search over learning rate, entropy coefficient, and margin weight using Optuna for Bayesian optimization.

### Rationale
Current config (`learning_rate=3e-4`, `entropy_coef=0.01`, etc.) are manual guesses from RL literature. Systematic search can find 3-5% performance improvements with modest compute investment.

### Scope
**Files Modified**:
- `_03_training/sweep.py` (new module)
- `_03_training/configs/sweep_example.json` (example config)

**Search Space**:
- `learning_rate`: [1e-4, 3e-4, 1e-3] (log-uniform)
- `entropy_coef`: [0.005, 0.01, 0.02] (log-uniform)
- `margin_weight`: [0.1, 0.25, 0.4] (uniform)

**Cross-Team Impact**:
- Requires significant GPU budget (9-27 trials × 50k steps each)
- Researchers gain principled way to explore config space

### Task Breakdown

**Implementation** (`_03_training/sweep.py`):
```python
import optuna
from pathlib import Path
from . import self_play

def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna optimization."""

    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    entropy_coef = trial.suggest_float('entropy_coef', 0.005, 0.02, log=True)
    margin_weight = trial.suggest_float('margin_weight', 0.1, 0.4)

    # Build config
    config = {
        'phase': f'sweep_trial_{trial.number}',
        'seed': 2025 + trial.number,
        'opponents': ['first', 'random', 'greedy', 'diego'],
        'totalSteps': 50000,  # Shorter for sweep efficiency
        'evalFrequency': 50000,  # Evaluate only at end
        'learningRate': learning_rate,
        'entropyCoef': entropy_coef,
        'marginWeight': margin_weight,
        # ... other fixed params
    }

    # Run training
    context = self_play.main_from_config(config)

    # Extract final Elo as objective
    eval_path = context.artifacts.eval / f'step_{config["totalSteps"]}_eval.json'
    eval_data = json.loads(eval_path.read_text())
    final_elo = eval_data['finalElo']

    return final_elo

def run_sweep(n_trials: int = 20, study_name: str = 'beastybar_sweep'):
    """Run Optuna sweep with TPE sampler."""

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=2025),
    )

    study.optimize(objective, n_trials=n_trials)

    # Save results
    sweep_dir = Path('_03_training/sweeps') / study_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = [
        {
            'trial': trial.number,
            'elo': trial.value,
            'params': trial.params,
        }
        for trial in sorted(study.trials, key=lambda t: t.value, reverse=True)
    ]

    (sweep_dir / 'leaderboard.json').write_text(json.dumps(leaderboard, indent=2))

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best Elo: {study.best_value:.1f}")
    print(f"Best params: {study.best_params}")

    return study

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--study-name', default='beastybar_sweep')
    args = parser.parse_args()

    run_sweep(n_trials=args.trials, study_name=args.study_name)
```

**Outputs**:
- `_03_training/sweeps/<study_name>/leaderboard.json` ranked by Elo
- Trial artifacts in `_03_training/artifacts/sweep_trial_*/`
- Optuna study database for visualization

### Owner
**Primary**: ML Engineer
**Support**: DevOps (for multi-GPU parallelization if available)

### Dependencies
**Blocked By**: Step 4 (need baseline for comparison)
**Blocks**: None (but informs best config for future training)

### Verification Procedure

1. **Dry Run**:
```bash
# Test with 3 trials, 5k steps each
python -m _03_training.sweep --trials 3 --study-name test_sweep

# Validate leaderboard created
cat _03_training/sweeps/test_sweep/leaderboard.json
```

2. **Full Sweep**:
```bash
# Production sweep: 20 trials, 50k steps each
python -m _03_training.sweep --trials 20 --study-name sweep_v1
```

3. **Validation**:
   - Leaderboard shows monotonic Elo ranking
   - Best trial Elo ≥ baseline_v2 final Elo (from Step 4)
   - Top-3 trials show consistent hyperparameter patterns

4. **Statistical Test**:
```python
# Compare best sweep config vs baseline
# Run 5 independent seeds with best params
# t-test for significant improvement
```

### Success Criteria
- ✅ Sweep completes 20 trials without crashes
- ✅ `leaderboard.json` correctly ranked by Elo
- ✅ Best config achieves ≥3% win rate improvement over baseline
- ✅ Hyperparameter trends interpretable (e.g., higher entropy_coef → better exploration)
- ✅ Optuna study serializable for later analysis

### Estimated Effort
**Time**: 2 days (8hr implementation + testing, 16hr compute wait time)
**Complexity**: Medium-High

---

## Step 7: Add Performance Profiling Dashboard

### Objective
Instrument training loop with steps/sec, GPU utilization, memory footprint; expose via telemetry API.

### Rationale
Cannot optimize what isn't measured. Current logs lack performance telemetry. Profiling identifies whether rollout collection or PPO update is the bottleneck.

### Scope
**Files Modified**:
- `_03_training/self_play.py` (add profiling hooks)
- `_04_ui/app.py` (new `/api/telemetry/runs/{run_id}/profiling` endpoint)

**Metrics Collected**:
- `steps_per_second`: Learner decisions per wall-clock second
- `gpu_memory_allocated_mb`: Peak VRAM usage
- `gpu_utilization_pct`: GPU busy percentage (via nvidia-smi)
- `rollout_time_ms`: Time spent collecting rollouts
- `ppo_update_time_ms`: Time spent in PPO gradient steps

**Cross-Team Impact**:
- Overhead <5% (profiler sampling rate tuned)
- Enables hardware budget justification (e.g., "GPU only 40% utilized → CPU-bound")

### Task Breakdown

**Implementation** (`_03_training/self_play.py`):
```python
import time
import torch

def _profile_iteration(config, rollout_fn, ppo_fn):
    """Wrap iteration with profiling."""

    iter_start = time.perf_counter()

    # Rollout phase
    rollout_start = time.perf_counter()
    batch = rollout_fn()
    rollout_time = (time.perf_counter() - rollout_start) * 1000  # ms

    # PPO phase
    ppo_start = time.perf_counter()
    metrics = ppo_fn(batch)
    ppo_time = (time.perf_counter() - ppo_start) * 1000  # ms

    iter_time = time.perf_counter() - iter_start
    steps_per_sec = batch.steps / iter_time

    # GPU metrics (if CUDA available)
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        torch.cuda.reset_peak_memory_stats()

        # Sample GPU utilization via nvidia-smi
        import subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=1
            )
            gpu_util = float(result.stdout.strip().split()[0])
        except:
            gpu_util = None
    else:
        gpu_mem = None
        gpu_util = None

    return {
        'steps_per_second': steps_per_sec,
        'rollout_time_ms': rollout_time,
        'ppo_update_time_ms': ppo_time,
        'gpu_memory_allocated_mb': gpu_mem,
        'gpu_utilization_pct': gpu_util,
    }
```

**Updated Metrics Schema** (`metrics/step_*.json`):
```json
{
  "timestamp": "...",
  "iteration": 42,
  "steps": 86016,
  "avgEpisodeReward": 0.234,

  "profiling": {
    "steps_per_second": 487.3,
    "rollout_time_ms": 3215,
    "ppo_update_time_ms": 987,
    "gpu_memory_allocated_mb": 1843,
    "gpu_utilization_pct": 76.5
  }
}
```

**UI Endpoint** (`_04_ui/app.py`):
```python
@app.get("/api/telemetry/runs/{run_id}/profiling")
def api_telemetry_profiling(run_id: str) -> dict:
    """Aggregate profiling metrics across iterations."""

    metrics_dir = artifacts_dir / run_id / "metrics"
    if not metrics_dir.exists():
        raise HTTPException(404, f"Run '{run_id}' not found")

    profiling_data = []
    for metrics_file in sorted(metrics_dir.glob("step_*.json")):
        data = json.loads(metrics_file.read_text())
        if 'profiling' in data:
            profiling_data.append({
                'step': data['steps'],
                **data['profiling']
            })

    return {
        "runId": run_id,
        "profiling": profiling_data
    }
```

**Outputs**:
- Per-iteration profiling stats in metrics JSON
- API endpoint for time-series profiling data
- Identification of bottleneck phase (rollout vs PPO)

### Owner
**Primary**: ML Engineer
**Support**: Full-Stack Engineer (for API endpoint)

### Dependencies
**Blocked By**: None (can parallelize with other steps)
**Blocks**: None (but informs optimization priorities)

### Verification Procedure

1. **Local Test**:
```bash
# Run short training with profiling
python -m _03_training.self_play \
  --config configs/baseline_v2.json \
  --total-steps 5000

# Inspect metrics
jq '.profiling' _03_training/artifacts/*/metrics/step_*.json | head -20
```

2. **Validation**:
   - `steps_per_second` >0 and <10000 (sanity bounds)
   - `rollout_time_ms + ppo_update_time_ms` ≈ iteration time (±10%)
   - GPU metrics non-null when running on CUDA

3. **Overhead Check**:
```python
# Compare training time with/without profiling
# Overhead should be <5%
```

4. **API Test**:
```bash
curl http://localhost:8000/api/telemetry/runs/baseline_v2/profiling | jq
```

### Success Criteria
- ✅ Profiling metrics in `step_*.json` for all iterations
- ✅ API endpoint returns valid time-series data
- ✅ Profiling overhead <5% of total training time
- ✅ Bottleneck identified (e.g., "Rollout takes 75% of iteration time")
- ✅ GPU utilization metrics match manual `nvidia-smi` readings (±10%)

### Estimated Effort
**Time**: 4 hours (3hr implementation + 1hr validation)
**Complexity**: Medium

---

## Step 8: Build Ablation Testing Framework

### Objective
Disable specific card abilities to measure strategic importance; validate simulator coverage.

### Rationale
Understanding which species drive learned behavior is critical for:
- Validating that policy leverages all card types (not just brute-force high-strength)
- Identifying simulator bugs (e.g., if disabling Chameleon has no effect → ability not working)
- Guiding future game balance changes

### Scope
**Files Modified**:
- `_01_simulator/cards.py` (add `disabled_species` parameter to `resolve_play`)
- `_03_training/ablation.py` (new module for running ablation suite)

**Ablation Experiments**:
- 12 runs: disable each species individually
- 1 control: baseline champion with no ablations
- 1 extreme: disable all on-play abilities (degrades to strength-only game)

**Cross-Team Impact**:
- Requires 14 × 500-game tournaments (compute-intensive, ~2hr total)
- Simulator changes must be backward-compatible (flag-gated)

### Task Breakdown

**Implementation** (`_01_simulator/cards.py`):
```python
def resolve_play(
    game_state: state.State,
    card: state.Card,
    action: actions.Action,
    disabled_species: set[str] | None = None,
) -> state.State:
    """Apply card on-play ability.

    Args:
        disabled_species: Species whose abilities should be no-ops (for ablation testing)
    """
    disabled_species = disabled_species or set()

    if card.species in disabled_species:
        # Ablation: skip ability resolution
        return game_state

    # Existing ability resolution logic...
```

**Ablation Runner** (`_03_training/ablation.py`):
```python
from pathlib import Path
import json
from . import tournament

SPECIES_LIST = [
    'lion', 'snake', 'giraffe', 'kangaroo', 'monkey',
    'parrot', 'seal', 'chameleon', 'skunk', 'zebra', 'hippo', 'crocodile'
]

def run_ablation_suite(
    champion_manifest: Path,
    games_per_ablation: int = 500,
    seed: int = 9999,
):
    """Run ablation experiments for all species."""

    results = {}

    # Baseline (no ablations)
    print("Running baseline (no ablations)...")
    baseline = _run_ablation_tournament(
        champion_manifest,
        disabled_species=set(),
        games=games_per_ablation,
        seed=seed,
    )
    results['baseline'] = baseline

    # Ablate each species
    for species in SPECIES_LIST:
        print(f"Ablating {species}...")
        ablated = _run_ablation_tournament(
            champion_manifest,
            disabled_species={species},
            games=games_per_ablation,
            seed=seed + hash(species) % 10000,
        )
        results[species] = ablated

    # Ablate all (extreme test)
    print("Ablating all species...")
    all_ablated = _run_ablation_tournament(
        champion_manifest,
        disabled_species=set(SPECIES_LIST),
        games=games_per_ablation,
        seed=seed + 99999,
    )
    results['all_ablated'] = all_ablated

    # Compute deltas
    report = _generate_ablation_report(results, baseline)

    # Save results
    output_path = Path('_03_training/ablation_results.json')
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Ablation report saved to {output_path}")

    return report

def _run_ablation_tournament(
    champion_manifest: Path,
    disabled_species: set[str],
    games: int,
    seed: int,
) -> dict:
    """Run tournament with ablation patch."""

    # Monkey-patch cards.resolve_play to pass disabled_species
    # (requires adding global state or refactoring tournament to accept ablation config)

    # For now, simplified tournament call
    # TODO: Extend tournament.py to accept disabled_species parameter

    series = tournament.SeriesConfig(
        games=games,
        seed=seed,
        agent_a=champion,
        agent_b=diego,
        alternate_start=True,
    )
    result = tournament.play_series(series)

    return {
        'wins': result.summary.wins[0],
        'losses': result.summary.wins[1],
        'ties': result.summary.ties,
        'win_rate': result.summary.wins[0] / games,
    }

def _generate_ablation_report(results: dict, baseline: dict) -> dict:
    """Rank species by impact when ablated."""

    baseline_wr = baseline['win_rate']

    species_impact = []
    for species, ablated in results.items():
        if species in ['baseline', 'all_ablated']:
            continue

        ablated_wr = ablated['win_rate']
        delta = baseline_wr - ablated_wr

        species_impact.append({
            'species': species,
            'baseline_win_rate': baseline_wr,
            'ablated_win_rate': ablated_wr,
            'delta': delta,
            'impact_pct': (delta / baseline_wr) * 100 if baseline_wr > 0 else 0,
        })

    # Sort by descending impact
    species_impact.sort(key=lambda x: abs(x['delta']), reverse=True)

    return {
        'baseline': baseline,
        'all_ablated': results['all_ablated'],
        'species_ranking': species_impact,
    }
```

**Outputs**:
- `_03_training/ablation_results.json` with species impact ranking
- Validation that disabling high-impact species degrades performance
- Confirmation that ablating all abilities drops to near-random

### Owner
**Primary**: Simulation Engineer
**Support**: ML Engineer (for tournament integration)

### Dependencies
**Blocked By**: Step 4 (need strong baseline champion)
**Blocks**: None (but informs research directions)

### Verification Procedure

1. **Simulator Patch Test**:
```python
def test_ablation_disables_ability():
    """Verify disabled species don't trigger abilities."""
    state = simulate.new_game(seed=2025)
    # Play Lion with ablation
    action = ... # Lion action
    state_ablated = cards.resolve_play(state, card, action, disabled_species={'lion'})
    state_normal = cards.resolve_play(state, card, action)

    # Assert queue unchanged in ablated version
    assert state_ablated.zones.queue == state.zones.queue + (card,)
    # Assert queue reordered in normal version (Lion pushes to front)
    assert state_normal.zones.queue[0] == card
```

2. **Ablation Suite Run**:
```bash
python -m _03_training.ablation \
  --champion _03_training/artifacts/baseline_v2/champion.json \
  --games 500
```

3. **Validation**:
   - Baseline win rate ≈ champion Elo from Step 4
   - High-impact species: Lion, Snake, Monkey (expect >10% delta)
   - Low-impact species: Zebra, Hippo (expect <5% delta)
   - All-ablated win rate ≈ random (50% ± 5%)

4. **Statistical Test**:
```python
from scipy.stats import binomtest

# For top-impact species (e.g., Lion)
baseline_wins = 320  # 64% of 500
ablated_wins = 220   # 44% of 500
delta_pvalue = binomtest(ablated_wins, 500, baseline_wins/500, alternative='less')
assert delta_pvalue.pvalue < 0.01, "Impact not statistically significant"
```

### Success Criteria
- ✅ Ablation suite completes 14 experiments (12 species + baseline + all)
- ✅ Top-3 species drop win rate by ≥10% (p<0.01)
- ✅ All-ablated degrades to 45-55% win rate (near-random)
- ✅ Results ranked in `ablation_results.json`
- ✅ Findings documented in `docs/RESEARCH.md`

### Estimated Effort
**Time**: 8 hours (4hr implementation + 2hr compute + 2hr analysis)
**Complexity**: Medium-High

---

## Step 9: Upgrade UI to React with Live Training Graphs

### Objective
Replace static HTML with interactive React dashboard showing real-time training metrics via WebSocket.

### Rationale
Current UI limits usability for monitoring long-running training. Researchers must SSH + tail logs or manually refresh JSON files. Live dashboard improves UX and demo-ability.

### Scope
**Files Modified**:
- `_04_ui/static/` → complete React app (Vite-based)
- `_04_ui/app.py` (add WebSocket support, CORS headers)

**New Features**:
- Live plots: Elo over steps, win rate trends, entropy decay
- Card usage heatmap visualization
- Training status indicator (running/paused/completed)
- Mobile-responsive layout

**Cross-Team Impact**:
- Frontend build tooling required (Node.js, npm)
- Deployment complexity increases (static assets bundled)
- Non-trivial effort (~1 week full-stack time)

### Task Breakdown

**Frontend Setup**:
```bash
cd _04_ui/static
npm create vite@latest . -- --template react-ts
npm install recharts axios
```

**Component Structure**:
```
_04_ui/static/
├── src/
│   ├── App.tsx                 # Main dashboard layout
│   ├── components/
│   │   ├── EloChart.tsx        # Recharts line chart
│   │   ├── WinRateTable.tsx    # Opponent matchup table
│   │   ├── CardHeatmap.tsx     # Species usage visualization
│   │   └── TrainingStatus.tsx  # Live connection indicator
│   ├── hooks/
│   │   └── useWebSocket.ts     # WebSocket connection hook
│   └── api/
│       └── client.ts           # FastAPI axios client
```

**WebSocket Backend** (`_04_ui/app.py`):
```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/training/{run_id}")
async def websocket_training_updates(websocket: WebSocket, run_id: str):
    """Stream training metrics updates."""
    await manager.connect(websocket)

    try:
        # Watch metrics directory for new files
        metrics_dir = Path(f"_03_training/artifacts/{run_id}/metrics")
        last_update = None

        while True:
            rolling_metrics = metrics_dir / "rolling_metrics.json"
            if rolling_metrics.exists():
                stat = rolling_metrics.stat()
                if last_update is None or stat.st_mtime > last_update:
                    data = json.loads(rolling_metrics.read_text())
                    await websocket.send_json({"type": "metrics", "data": data})
                    last_update = stat.st_mtime

            await asyncio.sleep(2)  # Poll every 2 seconds

    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

**React Component Example** (`EloChart.tsx`):
```tsx
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { useWebSocket } from '../hooks/useWebSocket';

export function EloChart({ runId }: { runId: string }) {
  const { data, isConnected } = useWebSocket(`/ws/training/${runId}`);

  const chartData = data?.metrics?.map(m => ({
    steps: m.steps,
    elo: m.finalElo,
  })) || [];

  return (
    <div>
      <h3>Training Progress {isConnected && <span>🟢 Live</span>}</h3>
      <LineChart width={800} height={400} data={chartData}>
        <XAxis dataKey="steps" />
        <YAxis domain={[1400, 1700]} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="elo" stroke="#8884d8" />
      </LineChart>
    </div>
  );
}
```

**Outputs**:
- Production-ready React app at `_04_ui/static/dist/`
- WebSocket streaming for live updates
- Mobile-responsive dashboard

### Owner
**Primary**: Full-Stack Engineer
**Support**: ML Engineer (for metrics schema validation)

### Dependencies
**Blocked By**: Step 5 (expanded metrics needed for rich visualizations)
**Blocks**: None (but critical for M4 demo milestone)

### Verification Procedure

1. **Development Build**:
```bash
cd _04_ui/static
npm run dev  # Vite dev server at localhost:5173
```

2. **E2E Test**:
   - Start training in background
   - Open dashboard in browser
   - Verify Elo chart updates within 30 seconds
   - Check mobile layout on tablet/phone

3. **Production Build**:
```bash
npm run build
uvicorn _04_ui.app:create_app --reload
# Verify static assets served at http://localhost:8000
```

4. **Performance Check**:
   - WebSocket connection stable for >10 minutes
   - Chart rendering smooth with 100+ data points
   - Memory usage <100MB in browser

### Success Criteria
- ✅ Dashboard deployed at `/` route
- ✅ WebSocket auto-reconnects on training start
- ✅ Live Elo chart updates within 30s of new metrics
- ✅ Mobile-responsive (tested on tablet)
- ✅ Build completes without errors
- ✅ Production bundle <2MB gzipped

### Estimated Effort
**Time**: 5 days (3 days React implementation + 2 days integration/testing)
**Complexity**: High

---

## Step 10: Document Research Findings & Deployment

### Objective
Consolidate learnings into comprehensive documentation; create one-command Docker deployment.

### Rationale
Knowledge transfer for future maintainers, external collaborators, and potential academic publication. Docker ensures reproducible demo environment.

### Scope
**Files Modified**:
- `README.md` (refresh with new findings)
- `docs/RESEARCH.md` (new: convergence plots, ablation results)
- `docs/ARCHITECTURE.md` (new: module diagrams)
- `Dockerfile` (new)
- `docker-compose.yml` (new)

**Documentation Deliverables**:
1. **README.md**: Quickstart, troubleshooting, research summary
2. **docs/RESEARCH.md**: Hyperparameter findings, species importance, convergence analysis
3. **docs/ARCHITECTURE.md**: Dependency graphs, data flow diagrams
4. **Deployment Guide**: Docker setup, cloud deployment options

**Cross-Team Impact**:
- Onboarding time reduced from days to hours
- External contributors can reproduce results

### Task Breakdown

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application code
COPY _01_simulator/ ./_01_simulator/
COPY _02_agents/ ./_02_agents/
COPY _03_training/ ./_03_training/
COPY _04_ui/ ./_04_ui/
COPY _05_other/ ./_05_other/

# Expose UI port
EXPOSE 8000

# Default command: start UI server
CMD ["uvicorn", "_04_ui.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  beastybar-ui:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./_03_training/artifacts:/app/_03_training/artifacts:ro
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/agents"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**docs/RESEARCH.md** (outline):
```markdown
# Beasty Bar Research Findings

## Executive Summary
- Baseline champion achieves 60% win rate vs diego after 200k steps
- Hyperparameter sweep improved Elo by 4.2% over default config
- Top-3 impactful species: Lion (18% delta), Snake (14%), Monkey (12%)

## Training Convergence
[Insert Elo over steps plot]
- Policy plateaus around 150k steps
- Entropy decay curve shows healthy exploration maintained (>1.0 final)

## Hyperparameter Analysis
Best configuration:
- learning_rate: 2.1e-4 (vs baseline 3e-4)
- entropy_coef: 0.015 (vs baseline 0.01)
- margin_weight: 0.32 (vs baseline 0.25)

## Ablation Study Results
[Insert species impact bar chart]
| Species | Win Rate Delta | Statistical Significance |
|---------|----------------|--------------------------|
| Lion    | -18.3%         | p<0.001                  |
| Snake   | -14.1%         | p<0.001                  |
| Monkey  | -12.4%         | p<0.001                  |
...

## Future Directions
- Multi-agent league (population diversity)
- Transfer learning from simplified game
- Human preference learning (RLHF)
```

**Updated README.md**:
```markdown
# Beasty Bar RL Platform

[![CI](https://github.com/USER/beastybar/actions/workflows/ci.yml/badge.svg)](...)

Reinforcement learning research platform for the board game Beasty Bar.

## Quick Start

### Docker (Recommended)
```bash
git clone https://github.com/USER/beastybar.git
cd beastybar
docker-compose up
# Visit http://localhost:8000
```

### Manual Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
uvicorn _04_ui.app:create_app --reload
```

## Research Highlights
- **Champion Performance**: 60% win rate vs strongest heuristic
- **Training Efficiency**: Converges in 200k steps (~12hr on single GPU)
- **Ablation Findings**: Lion/Snake/Monkey drive strategic behavior

See [docs/RESEARCH.md](docs/RESEARCH.md) for full analysis.

## Documentation
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Training Guide](CLAUDE.md)
- [Contributing](CONTRIBUTING.md)

## Citation
```bibtex
@misc{beastybar2025,
  title={Reinforcement Learning for Beasty Bar Strategy Optimization},
  author={...},
  year={2025}
}
```
```

**Outputs**:
- Comprehensive documentation suite
- One-command Docker deployment
- Research findings formatted for publication

### Owner
**Primary**: Full-Stack Engineer (Docker), ML Engineer (research doc)
**Reviewer**: Technical Lead (documentation quality check)

### Dependencies
**Blocked By**: Steps 4-9 (need results to document)
**Blocks**: None (final deliverable)

### Verification Procedure

1. **Docker Test (Fresh Clone)**:
```bash
cd /tmp
git clone <repo> beastybar-test
cd beastybar-test
docker-compose up -d

# Wait for healthcheck
sleep 30

# Verify UI accessible
curl http://localhost:8000/api/agents
# Expected: {"agents": ["first", "random", "greedy", "diego"]}

# Play game
curl -X POST http://localhost:8000/api/new-game \
  -H "Content-Type: application/json" \
  -d '{"opponent": "diego"}'
```

2. **Documentation Review**:
   - Peer review by external reader (not on core team)
   - Check for broken links, outdated commands
   - Validate code examples execute correctly

3. **Research Validation**:
   - Convergence plots match artifact data
   - Ablation tables statistically sound
   - Hyperparameter claims reproducible

### Success Criteria
- ✅ Fresh clone → playable UI in <5 minutes via `docker-compose up`
- ✅ README quickstart tested on clean Ubuntu VM
- ✅ `docs/RESEARCH.md` peer-reviewed and approved
- ✅ CI validates Docker build on every PR
- ✅ All documentation up-to-date with current code

### Estimated Effort
**Time**: 3 days (1 day Docker + 2 days documentation)
**Complexity**: Medium

---

## Dependency Graph

```
Step 1 (Fix P0 Bias)
  └─> Step 2 (Archive Checkpoints)
      └─> Step 4 (Retrain Baseline)
          ├─> Step 5 (Expand Metrics)
          │   └─> Step 9 (React UI)
          ├─> Step 6 (Hyperparameter Sweep)
          └─> Step 8 (Ablation Framework)

Step 3 (CI Pipeline) [Parallel - no blockers]

Step 7 (Profiling) [Parallel - no blockers]

Step 10 (Documentation)
  └─> Blocked by: Steps 4-9 (need results to document)
```

**Critical Path**: 1 → 2 → 4 → (5,6,8) → 10
**Parallelizable**: Steps 3, 7 can start immediately

---

## Risk Mitigation

| Risk | Step | Mitigation |
|------|------|------------|
| GPU access bottleneck | 4, 6 | Negotiate cloud credits; optimize CPU fallback |
| React migration breaks UI | 9 | Feature flag rollout; comprehensive E2E tests |
| Sweep finds no improvement | 6 | Expand search space; curriculum learning |
| Docker build fails on ARM | 10 | Multi-arch builds; document platform limitations |
| Documentation goes stale | 10 | CI job validates code examples; quarterly review |

---

## Timeline Summary

| Week | Steps | Deliverables |
|------|-------|--------------|
| 1 | 1-3 | P0 fix, checkpoints archived, CI running |
| 2-3 | 4 | Baseline champion (200k steps) |
| 3 | 5, 7 | Expanded metrics, profiling |
| 4 | 6 | Hyperparameter sweep complete |
| 5 | 8 | Ablation study results |
| 6-7 | 9 | React UI deployed |
| 8 | 10 | Documentation finalized |

**Total Duration**: 8 weeks (with GPU access and full-time effort)

---

**End of Action Plan**
