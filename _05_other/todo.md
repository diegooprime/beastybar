# Project TODO — Self-play RL shift

## Phase 0 — Strategy & scope
- [ ] Draft a short design brief describing the self-play RL objective, success metrics (Elo deltas vs Greedy/Diego), and resource assumptions.
- [ ] Inventory current agents/checkpoints; decide which baselines remain in the evaluation pool.
- [ ] Define experiment logging conventions (run id structure, artifact directories, retention policy).

## Phase 1 — Simulator readiness
- [ ] Add an observation builder in `_01_simulator` that exports fixed-size tensors/dicts for queue, bar, discard, hand, and turn context.
- [ ] Expose reward helpers (win/loss, normalized margin, optional shaped signals) with deterministic seeds for reproducibility.
- [ ] Implement legal-action enumeration plus masking utilities, and cover them with pytest fixtures.
- [ ] Extend simulator tests under `_05_other/tests` to validate observation encoding, reward calculations, and mask correctness.

## Phase 2 — Learnable agent scaffold
- [ ] Implement `SelfPlayRLAgent` in `_02_agents` with model loading, inference, and stochastic exploration toggles.
- [ ] Surface the new agent via `_02_agents/__init__.py` and ensure tournaments can instantiate it by name.
- [ ] Write smoke tests that confirm deterministic outputs when exploration is disabled and seeds match.

## Phase 3 — Training pipeline
- [ ] Create a self-play training entry point (e.g., `_03_training/self_play.py`) that wires up configs, seeding, and logging.
- [ ] Build rollout workers that sample opponents (current policy + checkpoint reservoir) and collect trajectories.
- [ ] Implement PPO/actor-critic update loop with advantage computation, gradient clipping, and optimizer state saves.
- [ ] Store checkpoints and metrics under `_03_training/artifacts/` with manifest files describing hyperparameters.

## Phase 4 — Evaluation & promotion
- [ ] Automate periodic tournaments against the baseline pool and recent checkpoints; record win rates and Elo changes.
- [ ] Define promotion criteria (minimum games, win-rate threshold, Elo delta) and codify it in the training loop.
- [ ] Archive defeated checkpoints for replay/regression while flagging the active "champion" for downstream use.

## Phase 5 — Tooling & UI integration
- [ ] Extend `_04_ui` to list available checkpoints, show latest Elo trajectory, and expose quick replays of sampled games.
- [ ] Publish metrics (entropy, KL, reward mean/std, eval outcomes) as JSON/Parquet for dashboards or notebooks.
- [ ] Add CLI affordances (e.g., `--checkpoint-dir`, `--eval-frequency`, `--resume-from`) and document them in README/tournament docs.

## Phase 6 — Documentation & onboarding
- [ ] Update `_03_training/training_loop.md`, README, and AGENTS.md to reflect RL workflows, configs, and promotion rules.
- [ ] Provide a "toy run" guide that executes a short self-play session locally, including expected metrics.
- [ ] Ensure CI hooks (lint/tests) cover new modules and add regression tests for self-play configuration parsing.

> Track progress by checking boxes as milestones land; revisit the phases after the first end-to-end run to incorporate learnings.
