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
| `model_inference.pt` | ~5 MB | **Use this** - weights only, for inference/deployment |
| `final.pt` | ~500 MB | Full training checkpoint (resume training) |

| Metric | Value |
|--------|-------|
| Win rate vs heuristics | 79% |
| Win rate vs random | 93% |
| Training | 600 iterations, 5,027,840 games |
| Time | 109 min on H200 |

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

# Convert 500MB training checkpoint to 5MB inference checkpoint
export_for_inference("checkpoints/final.pt", "model_inference.pt")
```

## Training

PPO with opponent pool diversity. Pure self-play causes collapse—mixing in random/heuristic opponents fixes it.

```bash
uv sync
uv run scripts/train.py --config configs/default.yaml
```

## Stack

- Transformer policy-value network (988-dim input → 124 actions)
- PPO with GAE
- Cython acceleration (200x speedup)
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
