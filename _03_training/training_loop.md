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
- CLI ergonomics: extend `tournament.py` or add `self_play.py` with options like `--total-steps`, `--eval-frequency`, `--checkpoint-dir`, `--resume-from`.

## Proposed end-to-end loop
1. Seed run configuration (hyperparameters, checkpoint roots, opponent mix) and record it in an immutable run manifest.
2. Spin up rollout workers that repeatedly:
   - Sample an opponent checkpoint,
   - Play deterministic games via `_01_simulator.play_series`,
   - Yield batched trajectories with masks/rewards ready for training.
3. Aggregate batches, compute advantages, and run a fixed number of optimization epochs on-policy.
4. Write updated model weights + optimizer state to disk and append scalar metrics to the training log.
5. Trigger evaluation tournaments; update Elo, produce promotion decisions, and notify the UI layer of newly published checkpoints.
6. Continue until convergence criteria (target Elo, plateau, or resource budget) are met; archive final manifest and metrics bundle.

## Documentation and next steps
- Document observation and reward specs in `_03_training/README.md` (or expand this file) once finalized.
- Add deterministic unit tests that rebuild observations/rewards for canned states and verify action masking.
- Provide a quick-start guide for running a toy self-play session so contributors can validate their environment locally.
