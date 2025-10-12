Beasty Bar is a board game I like a lot. It's fun, simple, easy to learn and requires strategy. 
I played a ton with friends and wanted the “best” strategy. I searched online. Nothing.
So I’m building a simulator to test strategies until I find the best one. Then I’ll use the engine to sharpen my intuition and win more games.
First project of this type for me.
Purpose: Have the best Beasty Bar strategy in the world.

### Project structure:
- **_01_simulator**: Full game rules and state management (deterministic, side-effect free)
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

### How It Works

1. **Simulator** (`_01_simulator/`) provides deterministic game engine with seed-threaded randomness
2. **UI** (`_04_ui/`) lets humans play both sides of a match, inspect turn history, and replay deterministic seeds

### Misc
- Built in Python. Fast to build. I want to get better at it.
- FastAPI + static html. Fast, simple and zero build.
- Beasty Bar intro: https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf
- Beady Bar more rules: https://www.ultraboardgames.com/beasty-bar/game-rules.php
