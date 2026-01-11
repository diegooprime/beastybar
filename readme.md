# Beasty Bar AI

Train AI agents to play [Beasty Bar](https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf), a 2-player card game where animals jostle to enter a bar.

**Current best model: 79% win rate** against heuristic opponents.

## Quick Start

```bash
# Install
uv sync

# Run tests
pytest _05_other/tests -ra

# Start web UI
uvicorn _04_ui.app:create_app --reload
# Visit http://localhost:8000
```

## Training

```bash
# PPO (fast, good for experimentation)
uv run scripts/train.py --config configs/default.yaml

# MCTS/AlphaZero (slower, stronger results)
uv run scripts/train_mcts.py --config configs/h100_mcts.yaml
```

## Evaluation

```bash
uv run scripts/evaluate.py \
  --model checkpoints/model.pt \
  --opponents random,heuristic \
  --games 100
```

## Project Structure

```
_01_simulator/     # Game engine (deterministic, immutable state)
_02_agents/        # AI players (random, heuristic, MCTS, neural)
_03_training/      # Training infrastructure (PPO, MCTS)
_04_ui/            # Web interface
_05_other/         # Tests and game rules
scripts/           # CLI tools
configs/           # Training configurations
```

## Documentation

| Doc | Description |
|-----|-------------|
| [Technical Reference](docs/TECHNICAL.md) | Architecture, API, configuration |
| [Game Rules](_05_other/rules.md) | Official Beasty Bar rules |
| [MCTS Guide](_02_agents/mcts/README.md) | Monte Carlo Tree Search |
| [Training Guide](_03_training/MCTS_TRAINING.md) | AlphaZero-style training |
| [CLI Reference](scripts/README.md) | Script usage |

## The Game

Each player has 12 animals with unique abilities. Play cards to the queue, trigger effects, and score points when animals enter the bar.

**Turn:** Play card → Execute effect → Process recurring effects → 5-card check (front 2 enter bar, last 1 bounced) → Draw

**Animals:** Lion (12), Hippo (11), Crocodile (10), Snake (9), Giraffe (8), Zebra (7), Seal (6), Chameleon (5), Monkey (4), Kangaroo (3), Parrot (2), Skunk (1)

See [full rules](_05_other/rules.md) for details.

## Development

```bash
uv run ruff check .
uv run mypy _01_simulator _02_agents _03_training
uv run pytest _05_other/tests -ra
```

## References

- [Beasty Bar Rules (PDF)](https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
