# codex5.2 Implementation Plan

## Scope & Success Criteria
- Harden simulator correctness, automated evaluation, and H200 performance.
- Success: expanded tests green, automated arena/gating in place, documented perf baselines and recommended train/eval configs.

## Workstreams
### 1) Testing & Correctness
- Add rule regressions covering all species interactions (on-play/recurring, chameleon params, Heaven’s Gate edge cases).
- Round-trip tests for observations/action masks vs engine validation.
- CI smoke: tiny MCTS self-play + tiny training iteration (PPO disabled by default) to catch NaNs/shape drift.

### 2) Evaluation Harness
- Arena runner: fixed seeds, both starting sides, vs baselines (random/heuristic) and rolling past checkpoints.
- Metrics: win rate + Wilson bounds, Elo table, CSV/JSON artifacts; promotion gate based on lower-bound threshold vs gate model.
- Plotting/notebook hooks for win-rate/Elo trends from artifacts.

### 3) Training Pipeline Cleanup
- Canonicalize AlphaZero/MCTS path; park/deprecate PPO configs.
- Align YAML configs with MCTSConfig/NetworkConfig; remove stale ones.
- Tiny smoke training run for CI health.
- Log: policy/value loss, entropy, visit-count KL vs prior, win rates vs baselines.

### 4) Performance Profiling (H200)
- Bench batched forward latency across batch sizes (32/64/128), precisions (fp16/bf16), torch.compile on/off.
- Derive sims/move and games/sec for train vs eval settings; document recommended configs and trade-offs.
- Eliminate CPU-GPU syncs; keep obs/masks on device.

### 5) Diagnostics & Guards
- Entropy/value calibration plots; collapse detectors (entropy floor, value variance).
- NaN/Inf guards in training; alert/logging hooks for anomalies (win-rate drops, loss spikes).

### 6) Opponent Pool Policy
- Define add/evict cadence (e.g., every N iterations, cap K checkpoints), sampling weights, and GC rules for stored checkpoints.

### 7) Dependencies & Docs
- Ensure pyproject lists train deps (torch, pyyaml; wandb/tensorboard optional extras).
- Update README/docs with train/eval commands, gate criteria, and perf tuning guidance once benchmarks land.

### 8) UI/Security (Lower Priority)
- Tighten CORS/auth/rate limits for demo UI; validate Claude bridge inputs if exposed.

