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
