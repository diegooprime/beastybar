# Scripts

## Training

```bash
uv run scripts/train.py --config configs/iter600_to_1000.yaml
uv run scripts/train.py --config configs/iter600_to_1000.yaml --tracker wandb
uv run scripts/train.py --resume checkpoints/iter_949.pt
```

## Evaluation

```bash
uv run scripts/evaluate.py --model checkpoints/iter_949.pt --opponents random,heuristic --games 200
```

## Tablebase Generation

```bash
uv run scripts/generate_tablebase.py --output data/endgame_5card.tb --max-cards 5
```

## Cython Build

```bash
bash scripts/build_cython.sh
```
