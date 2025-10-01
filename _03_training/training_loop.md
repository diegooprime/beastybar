# Self-play RL training loop

## Why reset the focus
- Keep pushing the strongest agent without needing curated opponents by letting it iterate against fresh versions of itself.
- Capture stable learning signals (state, reward, policy targets) directly from the simulator rather than hand-authored heuristics.
- Maintain compatibility with existing tournaments and UI so the latest checkpoint can always be compared with legacy baselines.

## Core requirements to unlock self-play

### 1. Simulator readiness
- Deterministic rollouts: `_01_simulator` already exposes seed-threaded transitions; double-check every state mutation is pure and repeatable.
- Observation builder: `_01_simulator.observations.build_observation` maps a `State` into fixed-size card feature tensors (queue, beasty bar, discard, hand, turn context).
- Reward surface: `_01_simulator.rewards` exposes terminal win/loss, normalized margin, and shaped composite helpers with deterministic seeding.
- Action space spec: `_01_simulator.action_space` exposes a fixed action catalog, mask, and index helper for policy heads.

### 2. Learnable agent scaffold
- `SelfPlayRLAgent` in `_02_agents.self_play_rl` loads policy logits, supports deterministic seeds, and exposes exploration toggles for training/eval.
- Share code paths: evaluation should reuse the same policy/value heads that training updates; inject exploration only when flagged (e.g., epsilon or temperature).
- Configure serialization: checkpoints live under `_03_training/artifacts/checkpoints/<run_id>/step_<N>.pt` and include metadata (commit, simulator hash, hyperparameters).

### 3. Experience and optimization pipeline
- Self-play matchmaker: pit the current policy against a mixture of older checkpoints (reservoir sampling, prioritized by Elo) to avoid catastrophic forgetting.
- Rollout workers: collect transitions `(obs, action, log_prob, value, reward, done)` with deterministic seeds for reproducibility; optionally leverage multiprocessing.
- Training stepper: run PPO / actor-critic updates with clipped objectives, generalized advantage estimation, and gradient norm control.
- Replay and curriculum: decide whether to store full episodes on disk (for auditing) or keep transient buffers; document retention policy.

### 4. Evaluation and promotion policy
- Periodic tournaments: after each checkpoint (or every K updates) run `_03_training.tournament` against the champion set (First, Random, Greedy, Diego, latest promoted RL).
- Elo tracking: update leaderboard history, but gate promotions on statistically significant improvement (e.g., 200 games, 55% win vs champion, +50 Elo).
- Archive snapshots: keep defeated but relevant checkpoints for future training mixture and regression testing.

### 5. Telemetry, logging, and tooling
- Metrics dashboard: emit JSON/Parquet with episode rewards, KL divergence, entropy, loss components, and win rates per opponent.
- Trace sampling: optionally persist action traces for a configurable slice of games when `collect_actions=True` so debugging stays cheap.
- CLI ergonomics: `self_play.py` provides comprehensive flags for training control, evaluation tuning, and hyperparameter selection.

## Proposed end-to-end loop
1. Seed run configuration (hyperparameters, checkpoint roots, opponent mix) and record it in an immutable run manifest.
2. Spin up rollout workers that repeatedly:
   - Sample an opponent checkpoint,
   - Play deterministic games via `_01_simulator.simulate`,
   - Yield batched trajectories with masks/rewards ready for training.
3. Aggregate batches, compute advantages, and run a fixed number of optimization epochs on-policy.
4. Write updated model weights + optimizer state to disk and append scalar metrics to the training log.
5. Trigger evaluation tournaments; update Elo, produce promotion decisions, and notify the UI layer of newly published checkpoints.
6. Continue until convergence criteria (target Elo, plateau, or resource budget) are met; archive final manifest and metrics bundle.

## CLI Reference

### Training Control
- `--phase`: Phase tag used in generated run ids (default: `p3`)
- `--seed`: Global seed for reproducibility (default: `2025`)
- `--total-steps`: Total learner decisions to train on (default: `1,000,000`)
- `--eval-frequency`: Save checkpoints every N learner steps (default: `50,000`)
- `--rollout-steps`: Minimum learner decisions per rollout batch (default: `2048`)
- `--artifact-root`: Root directory for artifacts (default: `_03_training/artifacts`)
- `--run-id`: Override auto-generated run id
- `--resume-from`: Optional checkpoint file/directory to resume from
- `--notes`: Free-form notes stored in the run manifest

### Opponent Configuration
- `--opponent`: Opponent name to include in evaluation pool; repeat flag for multiple (default: `first`, `random`, `greedy`, `diego`)

### Evaluation & Promotion
- `--eval-games`: Games per opponent when evaluating checkpoints (default: `200`)
- `--eval-seed`: Base seed used for evaluation tournaments (default: `4096`)
- `--reservoir-size`: Maximum checkpoints to retain as self-play opponents (default: `3`)
- `--promotion-min-games`: Minimum total games before a checkpoint can be promoted (default: `opponents * eval_games`)
- `--promotion-min-win-rate`: Average win rate threshold for champion promotion (default: `0.55`)
- `--promotion-min-elo-delta`: Minimum Elo delta above baseline required for promotion (default: `25.0`)

### PPO Hyperparameters
- `--learning-rate`: Adam learning rate (default: `3e-4`)
- `--ppo-epochs`: PPO epochs per rollout (default: `4`)
- `--ppo-batch-size`: PPO minibatch size (default: `512`)
- `--clip-coef`: PPO clipping coefficient (default: `0.2`)
- `--value-coef`: Value loss coefficient (default: `0.5`)
- `--entropy-coef`: Entropy bonus coefficient (default: `0.01`)
- `--max-grad-norm`: Gradient clipping norm (default: `0.5`)

### Reward Shaping
- `--gamma`: Discount factor (default: `0.99`)
- `--gae-lambda`: GAE lambda (default: `0.95`)
- `--margin-weight`: Weight applied to margin reward shaping (default: `0.25`)
- `--jitter-scale`: Magnitude of deterministic reward jitter (default: `0.01`)

### Device & Config
- `--device`: Torch device to run on (default: `auto`)
- `--config`: Optional path to a JSON config file (CLI flags override config values)

## Toy Self-Play Walkthrough

This quick walkthrough demonstrates a smoke test run you can execute locally to validate your environment:

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run a Short Training Session
```bash
# Option A: Use the provided local config (20k steps, ~5 minutes on CPU)
python -m _03_training.self_play --config _03_training/configs/self_play_local.json

# Option B: Use CLI flags for a minimal smoke run (5k steps)
python -m _03_training.self_play \
  --phase smoke \
  --seed 2025 \
  --opponent first --opponent random \
  --total-steps 5000 \
  --eval-frequency 2500 \
  --rollout-steps 512 \
  --eval-games 50 \
  --eval-seed 4096
```

### 3. Monitor Progress
Training will create artifacts under `_03_training/artifacts/<run_id>/`:
- `run_manifest.json`: Full run configuration
- `checkpoints/step_*.pt`: Model checkpoints saved at eval frequency
- `metrics/step_*.json`: Per-iteration training metrics
- `metrics/rolling_metrics.json`: Aggregated sliding window stats
- `eval/step_*_eval.json`: Tournament results with Elo tracking
- `champion.json`: Latest promoted policy manifest (if promotion occurs)

### 4. Evaluate the Trained Policy
```bash
# Run a tournament using the champion manifest
python -m _03_training.tournament self-play diego \
  --self-play-manifest _03_training/artifacts/<run_id>/champion.json \
  --games 200 \
  --seed 2025

# Or test against all baselines
python -m _03_training.tournament self-play first random greedy diego \
  --self-play-manifest _03_training/artifacts/<run_id>/champion.json \
  --games 100
```

### 5. Play Against the AI in the UI
```bash
uvicorn _04_ui.app:create_app --reload
# Visit http://localhost:8000
# Select "SelfPlayRL" from the agent dropdown
# The UI will use the latest champion.json automatically
```

### What to Expect
- **First checkpoint (step 2500/5000)**: Random-level performance, high entropy, exploration-heavy
- **Second checkpoint (step 5000)**: Slight improvement over random, beginning to prefer higher-value cards
- **Champion promotion**: May occur if win rate > 55% and Elo delta > 25 against baselines
- **Training metrics**: Watch `avg_episode_reward`, `policy_loss`, `value_loss`, and `entropy` in rolling_metrics.json

### Troubleshooting
- If training is slow on CPU, reduce `--rollout-steps` to 256 or fewer
- If memory usage is high, reduce `--ppo-batch-size` to 256
- For faster convergence, increase `--learning-rate` to 1e-3 (less stable)
- To debug action selection, add `--notes "debug run"` and inspect rollout JSON files

## Documentation and next steps
- Add deterministic unit tests that rebuild observations/rewards for canned states and verify action masking.
- Extend the toy walkthrough with visualization of training curves once plotting utilities are added.
- Document edge cases in promotion logic (ties, insufficient games, Elo plateau handling).
