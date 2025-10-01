Beasty Bar is a board game I like a lot. It's fun, simple, easy to learn and requires strategy. 
I played a ton with friends and wanted the “best” strategy. I searched online. Nothing.
So I’m building a simulator to test strategies until I find the best one. Then I’ll use the engine to sharpen my intuition and win more games.
First project of this type for me.
Purpose: Have the best Beasty Bar strategy in the world.

### Project structure:
- **_01_simulator**: Full game rules and state management (deterministic, side-effect free)
- **_02_agents**: Strategy implementations inheriting from `Agent` base class
- **_03_training**: Self-play RL training, tournaments, telemetry, Elo tracking
- **_04_ui**: FastAPI interface and static viewer for human play
- **_05_other**: Tests, utilities, docs, and references

The code for each section is independent so we can modify one without fucking up the other sections.

### Quick Start

#### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### 2. Run the UI (Human vs AI)
```bash
uvicorn _04_ui.app:create_app --reload
# Visit http://localhost:8000
```

#### 3. Run a Tournament (AI vs AI)
```bash
# Basic tournament
python -m _03_training.tournament first diego --games 25 --seed 2025

# Evaluate a trained RL agent
python -m _03_training.tournament self-play diego \
  --self-play-manifest _03_training/artifacts/champion.json \
  --games 200
```

#### 4. Train a New RL Agent (Self-Play)
```bash
# Quick local smoke test (20k steps, ~5 min on CPU)
python -m _03_training.self_play --config _03_training/configs/self_play_local.json

# Full training run with custom parameters
python -m _03_training.self_play \
  --phase p3 \
  --seed 2025 \
  --opponent first --opponent random --opponent greedy --opponent diego \
  --total-steps 1000000 \
  --eval-frequency 50000 \
  --eval-games 200 \
  --eval-seed 4096
```

Training artifacts (checkpoints, metrics, manifests) are saved to `_03_training/artifacts/<run_id>/`. The latest champion can be loaded in the UI or tournaments via `champion.json`.

#### 5. Run Tests
```bash
pytest _05_other/tests -ra
```

### How It Works

1. **Simulator** (`_01_simulator/`) provides deterministic game engine with seed-threaded randomness
2. **Agents** (`_02_agents/`) implement strategies: rule-based (First, Random, Greedy, Diego) and learned (SelfPlayRL)
3. **Training** (`_03_training/`) uses PPO self-play to discover optimal strategies:
   - Learner plays against baselines + recent checkpoints (reservoir sampling)
   - Evaluation tournaments run periodically with Elo tracking
   - Champion promotion occurs when win rate > 55% and Elo delta > 25
4. **UI** (`_04_ui/`) lets humans play against any agent, including the latest RL champion

See `_03_training/training_loop.md` for complete CLI reference and walkthrough.

### Misc
- Built in Python. Fast to build. I want to get better at it.
- FastAPI + static html. Fast, simple and zero build.
- Beasty Bar intro: https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf
- Beady Bar more rules: https://www.ultraboardgames.com/beasty-bar/game-rules.php
