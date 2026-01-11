# CLI Scripts

## Training

```bash
# PPO (fast)
uv run scripts/train.py --config configs/default.yaml

# MCTS/AlphaZero (recommended)
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml

# With W&B tracking
uv run scripts/train.py --config configs/default.yaml --tracker wandb

# Resume from checkpoint
uv run scripts/train.py --resume checkpoints/iter_000100.pt
```

## Evaluation

```bash
uv run scripts/evaluate.py --model checkpoints/model.pt --opponent heuristic
uv run scripts/evaluate.py --model checkpoints/model.pt --opponents random,heuristic --games 200
```

## Interactive Play

```bash
uv run scripts/play.py --model checkpoints/model.pt
uv run scripts/play.py --model checkpoints/model.pt --you-start
```

## Cython Acceleration

```bash
bash scripts/build_cython.sh
uv run scripts/benchmark_cython.py
```

## Configuration

Configs in `configs/`:

| Config | Use |
|--------|-----|
| `default.yaml` | Standard PPO |
| `fast.yaml` | Quick testing |
| `h100_mcts.yaml` | H100 GPU MCTS |
| `runpod_h200.yaml` | RunPod H200 |

## Common Issues

**CUDA OOM:** Reduce `minibatch_size` or `games_per_iteration`

**NaN loss:** Reduce `learning_rate`, check gradient clipping

**Import errors:** Run from project root
