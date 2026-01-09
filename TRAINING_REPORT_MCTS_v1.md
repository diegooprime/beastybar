# MCTS Training Report: Opponent Diversity v1

**Date**: 2026-01-08
**Duration**: ~2.5 hours (iterations 0-81 of 200)
**Status**: Stopped early - model failing to learn winning strategies

---

## Summary

AlphaZero-style MCTS training with opponent diversity **failed to produce a competent agent**. Despite losses decreasing consistently, the model's win rate against random opponents dropped from 46% (early) to 12% (final) - worse than random play.

### Key Result
| Metric | Start | Best | Final |
|--------|-------|------|-------|
| vs Random | 46% | 46% (iter 4) | 12% (iter 79) |
| vs Heuristic | 2% | 12% (iter 24) | 0% (iter 79) |
| Total Loss | 2.71 | - | 1.44 |
| Value Loss | 1.30 | - | 0.18 |

**Diagnosis**: Classic AlphaZero cold-start failure. The value network never learned useful game evaluation, so MCTS search produced poor policies, which trained the network to match poor policies, perpetuating the cycle.

---

## Configuration

### Network
- **Architecture**: Transformer (17M parameters)
- **Hidden dim**: 256, 8 heads, 4 layers
- **Input**: 988-dim observation
- **Output**: 124-dim policy + scalar value

### MCTS
```yaml
num_simulations: 200
c_puct: 2.0              # High exploration
temperature_drop_move: 30 # Stay stochastic longer
final_temperature: 0.25
dirichlet_alpha: 0.5
dirichlet_epsilon: 0.4
batch_size: 32
```

### Training
```yaml
learning_rate: 0.00005   # Slow learning
entropy_bonus: 0.05
games_per_iteration: 128
batch_size: 512
epochs_per_iteration: 4
```

### Opponent Diversity
```yaml
current_weight: 0.6      # 60% self-play
checkpoint_weight: 0.2   # 20% past checkpoints
random_weight: 0.1       # 10% random agent
heuristic_weight: 0.1    # 10% heuristic agent
```

---

## Training Progression

### Win Rate vs Random (Target: >50%)
```
Iter  4: 46%  ████████████████████████
Iter  9: 44%  ███████████████████████
Iter 14: 40%  ████████████████████
Iter 19: 30%  ███████████████
Iter 24: 34%  █████████████████
Iter 29: 24%  ████████████
Iter 34:  8%  ████
Iter 39: 14%  ███████
Iter 44: 16%  ████████
Iter 49: 14%  ███████
Iter 54: 16%  ████████
Iter 59: 16%  ████████
Iter 64: 18%  █████████
Iter 69: 18%  █████████
Iter 74: 16%  ████████
Iter 79: 12%  ██████
```

### Win Rate vs Heuristic (Target: >30%)
```
Iter  4:  2%  █
Iter  9:  4%  ██
Iter 14: 10%  █████
Iter 19:  4%  ██
Iter 24: 12%  ██████  (best)
Iter 29:  0%
Iter 34:  0%
Iter 39:  0%
Iter 44:  2%  █
Iter 49:  4%  ██
Iter 54:  2%  █
Iter 59:  4%  ██
Iter 64:  0%
Iter 69:  0%
Iter 74:  2%  █
Iter 79:  0%
```

### Loss Curves (Decreasing = Normal)
```
Iter  1: Total=2.71, Value=1.30, Policy=1.47
Iter 10: Total=1.97, Value=0.63, Policy=1.41
Iter 20: Total=1.88, Value=0.54, Policy=1.41
Iter 30: Total=1.71, Value=0.38, Policy=1.40
Iter 40: Total=1.62, Value=0.30, Policy=1.39
Iter 50: Total=1.56, Value=0.26, Policy=1.37
Iter 60: Total=1.52, Value=0.23, Policy=1.36
Iter 70: Total=1.48, Value=0.19, Policy=1.36
Iter 80: Total=1.45, Value=0.19, Policy=1.33
```

---

## What Went Wrong

### 1. Cold Start Problem
AlphaZero requires the value network to provide meaningful position evaluations for MCTS to work. With random initialization:
- Value network outputs ~0 for all positions
- MCTS with uninformative values ≈ random search
- Training on random search outputs reinforces randomness

### 2. Opponent Diversity Didn't Help
We added opponent diversity to prevent self-play collapse, but the fundamental problem was upstream - the MCTS itself was producing garbage policies because the value network never bootstrapped.

### 3. Loss ≠ Performance
Losses decreased steadily (2.71 → 1.44) but this just meant the network got better at matching MCTS visit distributions. Those distributions were never good to begin with.

---

## Lessons Learned

1. **AlphaZero needs warm start**: Either pretrain value network on expert games, or use much higher simulation counts initially

2. **Evaluation is truth**: Watch win rates, not losses. Losses can decrease while model gets worse.

3. **MCTS quality depends on value quality**: 200 simulations with bad value ≈ random. Need either:
   - Much more simulations (1000+)
   - Or good value estimates

4. **Opponent diversity is necessary but not sufficient**: Prevents collapse but doesn't fix cold start

---

## Recommendations for Next Attempt

### Option A: Pretrain Value Network
1. Generate games between heuristic agents
2. Train value network to predict game outcomes
3. Then run AlphaZero with warm-started value

### Option B: Hybrid PPO + MCTS
1. Train basic policy with PPO first (faster iteration)
2. Once PPO beats random, switch to MCTS refinement
3. Use PPO-trained value as MCTS initialization

### Option C: More MCTS Simulations
1. Increase from 200 → 800+ simulations
2. Slower but more robust search even with weak value
3. May need to reduce games_per_iteration for GPU memory

### Option D: Expert Iteration (ExIt)
1. Use heuristic agent to generate initial training data
2. Train network via imitation learning
3. Then self-play to exceed heuristic

---

## Files

- **Log**: `logs/mcts_diversity_training.log`
- **Checkpoints**: `checkpoints/mcts_opponent_diversity_v1/`
- **Config**: `configs/h100_mcts.yaml`

---

## Raw Metrics

### Opponent Distribution (Observed)
- Current network: ~60%
- Checkpoints: ~20% (checkpoint_19, checkpoint_39, etc.)
- Random: ~10%
- Heuristic: ~10%

### Game Generation Speed
- 128 games in ~25-30 seconds
- ~0.2s per game with BatchMCTS
- ~200x faster than sequential MCTS

### GPU Utilization
- H100 80GB
- ~2.5GB VRAM for network + MCTS batching
- Training batch: 512 samples
