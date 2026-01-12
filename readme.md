# Beasty Bar AI

**Goal: Create the best Beasty Bar player ever, for any opponent, all the time.**

I searched online for Beasty Bar AI projects and found nothing. So I built one.

## Demo

```bash
uvicorn _04_ui.app:create_app --reload
# http://localhost:8000
```

## Model

| File | Size | Purpose |
|------|------|---------|
| `model_inference.pt` | ~66 MB | **Use this** - weights only, for inference/deployment |
| `iter_949.pt` | ~847 MB | Full training checkpoint (resume training) |

| Metric | Value |
|--------|-------|
| Win rate vs heuristics | 88% |
| Win rate vs random | 98% |
| Win rate vs outcome_heuristic | 60% |
| Training | 949 iterations, 15M+ games |
| Time | ~10 hrs on A100 |

Hugging Face: https://huggingface.co/shiptoday101/beastybar-ppo

### Loading for Inference

```python
from _03_training.checkpoint_manager import load_for_inference
from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig

state_dict, config = load_for_inference("model_inference.pt")
network = BeastyBarNetwork(NetworkConfig.from_dict(config))
network.load_state_dict(state_dict)
```

### Converting Existing Checkpoints

```python
from _03_training.checkpoint_manager import export_for_inference

# Convert training checkpoint to inference checkpoint
export_for_inference("checkpoints/iter_949.pt", "model_inference.pt")
```

## Training

Two training approaches available:

**PPO (original):** Self-play with opponent pool diversity. Pure self-play causes collapse—mixing in random/heuristic opponents fixes it.

```bash
uv sync
uv run scripts/train.py --config configs/default.yaml
```

**AlphaZero (recommended):** MCTS policy targets with tablebase integration for superhuman play.

```bash
uv run python train_alphazero.py --config configs/alphazero_h100.yaml
```

## Endgame Tablebase

1 million solved endgame positions (≤4 cards per player) with perfect play.

| File | Size | Positions |
|------|------|-----------|
| `endgame_4card_final.tb` | 19 MB | 1,000,000 |

Hugging Face: https://huggingface.co/datasets/shiptoday101/beastybar-tablebase

```python
from _02_agents.tablebase import EndgameTablebase, TablebaseAgent

tablebase = EndgameTablebase.load("endgame_4card_final.tb")
agent = TablebaseAgent(tablebase, fallback_agent=neural_agent)
```

## Stack

- Transformer policy-value network (17M parameters)
- PPO with GAE / AlphaZero with MCTS
- Endgame tablebase (minimax with alpha-beta)
- Adaptive MCTS (100-6400 simulations)
- Cython acceleration (200x speedup)
- FastAPI web UI

## Structure

```
_01_simulator/   # Game engine
_02_agents/      # AI players (neural, MCTS, tablebase, solver)
_03_training/    # PPO + AlphaZero training
_04_ui/          # Web interface
```

## Links

- [Game rules](_05_other/rules.md)
- [Technical docs](docs/TECHNICAL.md)
- [Roadmap to Superhuman](ROADMAP_TO_SUPERHUMAN.md)
- [Beasty Bar PDF](https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf)

## License

MIT
