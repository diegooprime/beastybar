# Self-Play RL Design Brief

## Objective
- Train a `SelfPlayRLAgent` that surpasses the strongest heuristic baselines (Diego, Greedy) and can replace Diego as the default tournament champion.
- Maintain compatibility with existing tournament tooling so the agent can be evaluated via `_03_training.tournament` without bespoke code paths.

## Success Metrics
- Target +75 Elo versus Diego and +100 Elo versus Greedy across 400 evaluation games (5 seeds × 80 games) with 95% confidence intervals that exclude zero.
- Require at least 55% win rate versus the active champion over the same evaluation bundle before promotion.
- Track policy stability via KL divergence between consecutive checkpoints; flag runs where KL > 0.15 on average as regression risk.

## Resource Assumptions
- Primary training host: 1× NVIDIA RTX 4090 or Apple M-series GPU equivalent, 16 GB+ VRAM, paired with 16 CPU cores and 32 GB RAM for rollout workers.
- Iteration budget: 2.5 hours per 1M environment steps (includes rollouts + PPO updates); plan for 40M steps over two weeks of nightly runs.
- Storage budget: 250 GB for checkpoints, metrics, and sampled trajectories under `_03_training/artifacts/` with weekly pruning of stale rollouts.

## Experiment Scope
- Baseline opponents: `FirstLegalAgent`, `RandomAgent`, `GreedyAgent`, `DiegoAgent`, and the most recent promoted `SelfPlayRLAgent` checkpoint.
- Seed discipline: use deterministic seeds `[2025, 3031, 4047, 5053, 6067]` for evaluation tournaments and rotate rollout seeds every 10M steps.
- Observation/reward specs: adopt the canonical builder/margin helpers landing in `_01_simulator` Phase 1 tasks; lock schema once tests are green.

## Logging & Telemetry
- Run IDs: `{phase}-sp-{yyyymmdd}-{git_sha[:7]}-{seed}` (example: `p1-sp-20250315-a1b2c3d-2025`).
- Artifact layout: `_03_training/artifacts/<run_id>/{checkpoints,metrics,rollouts,eval}` with JSON manifests describing hyperparameters and simulator hash.
- Retention: keep latest 5 checkpoints + any promoted champion; purge rollouts older than 30 days or once artifact directory exceeds 200 GB.
- Metrics catalog: episode rewards, win rates per opponent, loss components (policy, value, entropy), KL, gradient norms, and inference latency.

## Open Questions
- Do we require off-policy evaluation against archived human replays before promotion?
- Should rollout workers stream trajectories via shared memory or persist to disk for auditability?
- Is there appetite to budget additional compute for population-based exploration if PPO plateaus?
