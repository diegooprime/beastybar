Beasty Bar is a board game I like a lot. It's fun, simple, easy to learn and requires strategy. 
I played a ton with friends and wanted the “best” strategy. I searched online. Nothing.
So I’m building a simulator to test strategies until I find the best one. Then I’ll use the engine to sharpen my intuition and win more games.
First project of this type for me.
Purpose: Have the best Beasty Bar strategy in the world.

### Project structure:
- **_01_simulator**: Full game rules and state management (deterministic, side-effect free)
- **_02_agents**: AI players (Random, Heuristic, MCTS)
- **_03_training**: Tournament runner and Elo ratings
- **_04_ui**: FastAPI interface and static viewer for human vs. human play
- **_05_other**: Tests, utilities, docs, and references

The code for each section is independent so we can modify one without fucking up the other sections. 

### Quick Start

#### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### 2. Run the UI (Human vs Human)
```bash
uvicorn _04_ui.app:create_app --reload
# Visit http://localhost:8000
```

#### 3. Run Tests
```bash
pytest _05_other/tests -ra
```

#### 4. Run Training (on NUC)
All training runs happen on `primenuc@prime-nuc`. Use the remote script:

```bash
# Sync code and run benchmark
./scripts/remote.sh sync
./scripts/remote.sh run --games 100

# Long training in background (tmux)
./scripts/remote.sh train --games 500 --include-slow

# Monitor
./scripts/remote.sh status   # check if running
./scripts/remote.sh attach   # view live (Ctrl+B, D to detach)
./scripts/remote.sh logs     # tail output
```

### How It Works

1. **Simulator** (`_01_simulator/`) provides deterministic game engine with seed-threaded randomness
2. **Agents** (`_02_agents/`) play against each other to find optimal strategies
3. **Training** (`_03_training/`) runs tournaments and tracks Elo ratings
4. **UI** (`_04_ui/`) lets humans play both sides of a match, inspect turn history, and replay deterministic seeds

### Misc
- Built in Python. Fast to build. I want to get better at it.
- FastAPI + static html. Fast, simple and zero build.
- Beasty Bar intro: https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf
- Beady Bar more rules: https://www.ultraboardgames.com/beasty-bar/game-rules.php
