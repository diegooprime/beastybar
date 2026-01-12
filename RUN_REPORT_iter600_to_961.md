# Training Run Report: h200_iter600_to_1000

**Date:** 2026-01-11 to 2026-01-12
**Duration:** ~10 hours
**Hardware:** RunPod A100 80GB SXM
**Final Iteration:** 961/1000 (stopped early by user)

---

## Summary

Resumed training from iteration 600, ran to iteration 961 (361 iterations completed).
Training was stable throughout with consistent loss values and steady performance.

---

## Final Checkpoints

| File | Size | Location |
|------|------|----------|
| `iter_949.pt` | 847MB | `checkpoints/v4/iter_949.pt` (local) |
| `final.pt` | 521MB | Original starting checkpoint |
| `model_inference.pt` | 66MB | Inference-only weights |

---

## Final Performance (Iteration 959 Eval)

| Opponent | Win Rate | Assessment |
|----------|----------|------------|
| random | 98% | Mastered |
| defensive | 92% | Strong |
| heuristic | 88% | Strong |
| aggressive | 84% | Good |
| skunk | 84% | Good |
| noisy | 82% | Good |
| queue | 78% | Moderate |
| online | 70% | Weak |
| outcome_heuristic | 60% | Struggling |
| distilled_outcome | 60% | Struggling |

**Average vs simple opponents (random, heuristic variants):** ~85%
**Average vs outcome-based opponents:** ~63%

---

## Performance Trends Over Training

### Win Rate Evolution (36 evaluations)

```
Opponent            Start(609)  Peak      Final(959)  Trend
─────────────────────────────────────────────────────────────
random              88%         100%      98%         Stable high
defensive           88%         96%       92%         Improved
heuristic           78%         88%       88%         Improved
aggressive          76%         88%       84%         Improved
skunk               80%         92%       84%         Improved
noisy               74%         90%       82%         Improved
queue               80%         88%       78%         Flat
online              66%         78%       70%         Slight improve
outcome_heuristic   66%         72%       60%         Declined
distilled_outcome   62%         74%       60%         Declined
```

### Key Observations

1. **Strong against simple opponents**: Consistent 85-98% vs random/heuristic variants
2. **Ceiling at 60-70% vs outcome-based**: Never broke through despite more training
3. **Late-stage decline vs hard opponents**: outcome_heuristic dropped from 66%→60%
4. **Entropy decay may be too aggressive**: Reduced exploration hurt performance vs smart opponents

---

## Training Stability

| Metric | Value | Assessment |
|--------|-------|------------|
| Loss Range | 0.08 - 0.17 | Stable |
| Final Loss | ~0.11 | Normal |
| LR Start | 3.64e-05 | Post-warmup |
| LR Final | 1.41e-06 | Cosine decay |
| GPU Util | 100% | Optimal |
| VRAM | 76GB/80GB | 95% utilized |

---

## Configuration Used

```yaml
# Key settings
total_iterations: 1000
games_per_iteration: 8192
minibatch_size: 16384
learning_rate: 0.0001
entropy_coef: 0.04 → 0.01 (linear decay)
temperature: 1.0 → 0.5 (linear decay)
eval_frequency: 10
checkpoint_frequency: 25
```

---

## Lessons Learned

### What Worked
- Stable training with no crashes or divergence
- Good GPU utilization (100%)
- Adaptive opponent weighting adjusted appropriately
- Strong performance vs heuristic opponents

### What Didn't Work
- PPO self-play alone couldn't break 70% vs outcome_heuristic
- Entropy decay too aggressive (0.04→0.01)
- No MCTS integration during training
- Value function not accurate enough for hard opponents

### Recommendations for Next Run
1. Slower entropy decay (end at 0.02, not 0.01)
2. Add MCTS-100 to opponent pool
3. Integrate MCTS into training (AlphaZero-style)
4. Improve value head architecture
5. Consider 2000+ iterations for harder opponents

---

## Wandb Run

**URL:** https://wandb.ai/diegoships101-none/beastybar/runs/4j03upe1

---

## Files Generated

```
checkpoints/v4/iter_949.pt          # Latest checkpoint (847MB)
thoughts.md                          # ML engineer evaluation
ROADMAP_TO_SUPERHUMAN.md            # Future development plan
RUN_REPORT_iter600_to_961.md        # This file
```

---

## Next Steps

See `ROADMAP_TO_SUPERHUMAN.md` for detailed plan to reach 90%+ vs all opponents.

**Immediate:** Use iter_949.pt as warm start for Phase 1 training with enhanced config.

---

*Report generated: 2026-01-12*
