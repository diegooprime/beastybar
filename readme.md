# Beasty Bar AI

Neural network that plays [Beasty Bar](https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf).

I wanted to see if I could train an AI to beat me at my favorite card game. Current model hits 79% win rate against heuristic opponents.

## Demo

```bash
uvicorn _04_ui.app:create_app --reload
# http://localhost:8000
```

## Model

| Metric | Value |
|--------|-------|
| Win rate vs heuristics | 79% |
| Win rate vs random | 93% |
| Training time | 109 min on H200 |
| Games trained | 5M+ |

Checkpoint: `checkpoints/v4/final.pt`
Hugging Face: https://huggingface.co/shiptoday101/beastybar-ppo

## Training

PPO with opponent pool diversity. Pure self-play causes collapse—mixing in random/heuristic opponents fixes it.

```bash
uv sync
uv run scripts/train.py --config configs/default.yaml
```

## Stack

- Transformer policy-value network (988-dim observation → 124 actions)
- PPO with GAE
- Optional Cython acceleration (200x speedup)
- FastAPI web UI

## Structure

```
_01_simulator/   # Game engine
_02_agents/      # AI players
_03_training/    # PPO training
_04_ui/          # Web interface
```

## Links

- [Game rules](_05_other/rules.md)
- [Technical docs](docs/TECHNICAL.md)
- [Beasty Bar PDF](https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf)

## License

MIT
