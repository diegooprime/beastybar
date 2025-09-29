# Baseline Agents & Checkpoint Inventory

## Agents
- `FirstLegalAgent`
  - Deterministic first-legal baseline; exercises engine plumbing without heuristics.
  - Determinism: pure (no RNG usage).
  - Known Elo (vs Diego baseline): TBD — leaderboard history not yet generated in this repo snapshot.
- `RandomAgent`
  - Uniformly samples from legal actions; accepts optional seed for reproducibility.
  - Determinism: stochastic unless seeded; leverage seed propagation during tournaments.
  - Known Elo (vs Diego baseline): TBD — requires fresh tournament run to log statistics.
- `GreedyAgent`
  - Scores moves via `_02_agents.evaluation.best_action` using `material_advantage` heuristic by default.
  - Determinism: deterministic given simulator seed and heuristic; no internal RNG.
  - Known Elo (vs Diego baseline): historical estimates place it ~100 Elo below Diego, but confirm with new leaderboard export.
- `DiegoAgent`
  - Species-specific heuristics with limited RNG for chameleon/giraffe/snake tie-breaking using state-derived seed.
  - Determinism: effectively deterministic because RNG is seeded from state via `_rng_seed`.
  - Known Elo: reference champion; use current tournament CLI to refresh actual rating once baselines are rerun.

## Checkpoints
- Searched `_03_training/artifacts/` and workspace for `*.pt` / `*.pth` files; none present besides editable install metadata under `.venv/`.
- No self-play checkpoints have been published; first qualifying run should export to `_03_training/artifacts/<run_id>/checkpoints/` following the design brief.
- Action: block tournament evaluations on agents alone until the first RL checkpoint lands; document the inaugural checkpoint as `champion.json` manifest.

## Next Steps
1. Run `python3 -m _03_training.tournament first diego --games 200 --seed 2025 --export-leaderboard _03_training/logs/baselines.json` to capture baseline Elo references.
2. Populate `_03_training/artifacts/` structure with empty README describing expected contents before the first training run commits artifacts.
