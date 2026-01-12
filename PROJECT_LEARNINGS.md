# Project Learnings

What I learned building a Beasty Bar AI over 3.5 months and 94 commits.

I kept making the same mistakes. This doc exists so I don't repeat them.

## The Big Ones

**Self-play collapses.** Pure self-play produces agents that beat themselves but lose 90% to random. Fix: opponent pool (60% self, 20% checkpoints, 20% diverse).

**MCTS cold-start fails.** AlphaZero from scratch dropped win rate 46% → 12% over 81 iterations. Value network learns nothing without a reasonable policy. Fix: PPO warmstart first, then AlphaZero refinement.

**Rule bugs invalidate everything.** Crocodile, Skunk, and Parrot were all implemented wrong. Discovered after weeks of training. Had to delete 7,500 lines and start over.

**60-70% ceiling is real.** PPO cannot beat forward-simulation heuristics above 70%. This isn't a bug. Breaking through requires MCTS integration during training.

**100% win rate is impossible.** Hidden information (opponent's 4 cards) means some games are unwinnable regardless of play. Realistic ceiling: 85-90% vs heuristics.

## Timeline

| Date | Event |
|------|-------|
| Sep 26, 2025 | Started. Game engine + basic UI in 24 hours. |
| Sep 27, 2025 | First 50k games. PPO training began. |
| Oct 1, 2025 | Discovered rule bugs. Training data invalid. |
| Oct 12, 2025 | Clean slate. Deleted everything. |
| Nov 2025 | Opponent pool working. 85% vs random. |
| Jan 10, 2026 | PPO iter 949. 98% vs random, 88% vs heuristic. |
| Jan 12, 2026 | AlphaZero + tablebase + population training implemented. |

## What Failed

| Failure | Why | Fix |
|---------|-----|-----|
| MCTS cold-start | Value network learns nothing useful | PPO warmstart |
| Pure self-play | Exploits own weaknesses, not sound play | Opponent pool |
| Entropy 0.04→0.01 | Reduced exploration too fast | End at 0.02 |
| Crocodile/Skunk/Parrot bugs | Didn't test rules before training | Verify first |

## What Stuck

- Immutable state (frozen dataclasses)
- Numbered folders (`_01_simulator/`, `_02_agents/`, etc.)
- Opponent pool with fixed ratios
- YAML configs for experiments
- Checkpoint reservoir for diversity

## Current State

98% vs random. 88% vs heuristic. 60-70% vs outcome_heuristic (ceiling).

| Component | Status |
|-----------|--------|
| Network V1 | Trained, 1.3M params |
| Network V2 | Implemented, 12.8M params, not trained |
| AlphaZero trainer | Functional |
| Tablebase | 1M endgame positions |
| Population training | Implemented, not run |

## Next

1. Train V2 with AlphaZero
2. Run population training to find weaknesses
3. Break 80% vs outcome_heuristic
4. Generate opening book

## Key Files

```
_02_agents/neural/network_v2.py     # V2 architecture
_03_training/alphazero_trainer.py   # MCTS training
_03_training/population.py          # Exploit-patch cycles
_02_agents/tablebase/endgame.py     # Solved positions
checkpoints/v4/iter_949.pt          # Best model
```

---

*Jan 2026 | 94 commits | 15M+ games trained*
