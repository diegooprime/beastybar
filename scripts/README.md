# Scripts

## Training

```bash
uv run scripts/train.py --config configs/default.yaml
uv run scripts/train.py --config configs/default.yaml --tracker wandb
uv run scripts/train.py --resume checkpoints/iter_000100.pt
```

## Evaluation

```bash
uv run scripts/evaluate.py --model checkpoints/model.pt --opponents random,heuristic --games 200
```

## Play

```bash
uv run scripts/play.py --model checkpoints/model.pt
```

## Cython

```bash
bash scripts/build_cython.sh
uv run scripts/benchmark_cython.py
```
