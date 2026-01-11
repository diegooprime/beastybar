# Beasty Bar PPO Model Evaluation Report

**Model**: `v4/final.pt`
**Date**: 2026-01-10
**Games per opponent**: 100 (alternating starting positions)

## Summary

| Metric | Value |
|--------|-------|
| Total games played | 800 |
| Overall record | 632W / 115L / 53D |
| Overall win rate | **79.0%** |

## Detailed Results by Opponent

| Opponent | Win Rate | 95% CI | Avg Point Margin | Avg Game Length |
|----------|----------|--------|------------------|-----------------|
| random | 93.0% | [86.3%, 96.6%] | +6.8 | 22.7 |
| queue | 82.0% | [73.3%, 88.3%] | +4.1 | 22.8 |
| heuristic | 81.0% | [72.2%, 87.5%] | +3.3 | 22.9 |
| skunk | 79.0% | [70.0%, 85.8%] | +4.3 | 22.9 |
| defensive | 78.0% | [68.9%, 85.0%] | +3.7 | 22.7 |
| aggressive | 75.0% | [65.7%, 82.5%] | +2.8 | 22.9 |
| noisy | 72.0% | [62.5%, 79.9%] | +3.3 | 22.9 |
| online | 72.0% | [62.5%, 79.9%] | +2.9 | 22.8 |

## Performance by Tier

| Tier | Win Rate |
|------|----------|
| vs random | 93% |
| vs heuristic family (6 variants) | 78% avg |
| vs online (hardest) | 72% |

## Analysis

### Strengths
- **Dominates random play**: 93% win rate shows strong tactical fundamentals
- **Consistent across heuristics**: 72-82% against all 6 heuristic variants
- **Positive point margins**: Wins decisively (avg +2.8 to +6.8 points)

### Observations
- **Queue controller is beatable**: 82% vs queue shows model learned queue management
- **Aggressive play slightly harder**: 75% vs aggressive (lowest among pure heuristics)
- **Noisy/human-like opponents harder**: 72% vs noisy matches online difficulty
- **Online is hardest**: 72% - this reactive counter-play agent is the toughest opponent

### Training Configuration
- **Algorithm**: PPO with opponent pool diversity
- **Hardware**: RunPod H200 GPU
- **Training time**: ~109 minutes
- **Total games**: 5M+ games across 600 iterations
- **Opponent pool**: 60% self-play, 20% checkpoints, 10% random, 10% heuristics

### Model Architecture
- Hidden dimension: 256
- Attention heads: 8
- Transformer layers: 4
- Species embedding: 64-dim

## Opponent Reference

| Short Name | Description |
|------------|-------------|
| `random` | Random action selection |
| `heuristic` | Baseline heuristic agent |
| `aggressive` | High bar weight, aggression=0.8 |
| `defensive` | Low aggression=0.2 |
| `queue` | Prioritizes queue front control |
| `skunk` | Skunk specialist |
| `noisy` | Human-like bounded rationality |
| `online` | Reactive counter-play (OnlineStrategies) |

## Files

- Model checkpoint: `checkpoints/v4/final.pt`
- Hugging Face: https://huggingface.co/shiptoday101/beastybar-ppo
