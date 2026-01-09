# RunPod H200 Training Guide

## Quick Start

### 1. Create RunPod Instance
- **GPU**: 1x H200 SXM (141GB HBM3e) or H100 (80GB)
- **Template**: PyTorch 2.x (CUDA 12.x)
- **Storage**: 50GB minimum

### 2. Upload Code
```bash
# Option A: Git clone (if repo is accessible)
git clone <your-repo-url> beastybar
cd beastybar

# Option B: Upload zip and extract
unzip beastybar.zip
cd beastybar
```

### 3. Set wandb API Key
```bash
export WANDB_API_KEY=your_wandb_api_key_here
```

### 4. Run Training
```bash
bash scripts/runpod_setup.sh
```

Or manually:
```bash
pip install wandb pyyaml
python scripts/train_h200.py --config configs/h200_optimized.yaml
```

---

## Configuration

### h200_optimized.yaml Key Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_dim` | 128 | Balanced (deep_narrow showed depth > width) |
| `num_layers` | 3 | More depth for sequential reasoning |
| `learning_rate` | 0.0001 | Most stable from experiments |
| `entropy_coef` | 0.02 | Increased to prevent early collapse |
| `games_per_iteration` | 512 | Full sample diversity with GPU speed |
| `total_iterations` | 500 | Target for 1-hour training |

### Command-Line Overrides

```bash
# Example with overrides
python scripts/train_h200.py \
    --config configs/h200_optimized.yaml \
    --iterations 750 \
    --lr 0.00015 \
    --eval-games 150 \
    --experiment-name h200_v2
```

---

## Expected Performance

### Throughput Estimates

| Metric | CPU (Intel NUC) | H200 (expected) |
|--------|-----------------|-----------------|
| Games/iter | 128 | 512 |
| Iter time | ~30s | ~8-12s |
| Iters/hour | ~120 | ~300-450 |

### Target Metrics

| Opponent | Baseline (80 iter) | Target (500 iter) |
|----------|-------------------|-------------------|
| Random | 66% | >75% |
| Heuristic | 16% (peak) | >25% |

---

## Monitoring

### wandb Dashboard
Check your wandb project at: https://wandb.ai/<your-username>/beastybar-experiments

Key metrics to watch:
- `eval/heuristic/win_rate` - Primary success metric
- `train/entropy` - Should stay above 0.5 to avoid collapse
- `train/policy_loss` - Should decrease steadily

### Console Output
```
Iter  100/500 | Loss: 0.0342 | LR: 9.50e-05 | Time: 9.2s | ETA: 61.3min
================================================================================
Evaluation Report (Iteration 100)
================================================================================
Opponent        Games    W-L-D        Win%    CI (95%)             Margin   Length
--------------------------------------------------------------------------------
random            100    72-25-3      72.0%   [0.62, 0.80]          +2.31     18.4
heuristic         100    18-78-4      18.0%   [0.11, 0.27]          -1.87     19.2
================================================================================
```

---

## Output Files

```
checkpoints/h200_optimized_v1/
├── best_model.pt              # Best by heuristic win rate
├── final.pt                   # Final checkpoint
├── iter_000025.pt             # Periodic checkpoints
├── iter_000050.pt
├── ...
├── heuristic_tracker_state.json  # Best model tracking
└── *.json                     # Config snapshots
```

---

## Troubleshooting

### CUDA Out of Memory
Reduce `minibatch_size` in config:
```yaml
ppo_config:
  minibatch_size: 128  # down from 256
```

### Training Stalls
- Check `train/entropy` - if near 0, increase `entropy_coef`
- Check learning rate schedule - may need longer warmup

### wandb Connection Issues
Run without wandb:
```bash
python scripts/train_h200.py --no-wandb
```

---

## After Training

1. **Download best model**: `checkpoints/h200_optimized_v1/best_model.pt`
2. **Check wandb** for full training curves
3. **Run final comparison** against previous best:
```python
from _03_training.model_selection import compare_checkpoints
result = compare_checkpoints("best_model.pt", "old_best.pt", num_games=200)
print(f"Winner: {result['winner']} (p={result['p_value']:.4f})")
```
