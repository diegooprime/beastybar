Beasty Bar is a board game I like a lot. It's fun, simple, easy to learn and requires strategy. 
I played a ton with friends and wanted the “best” strategy. I searched online. Nothing.
So I’m building a simulator to test strategies until I find the best one. Then I’ll use the engine to sharpen my intuition and win more games.
First project of this type for me.
Purpose: Have the best Beasty Bar strategy in the world.

### Project structure:
- _01_simulator: full game rules and state management
- _02_agents: each strategy as an agent implementation
- _03_training: tournaments, telemetry tooling, logs, Elo tracking
- _04_ui: FastAPI interface and static viewer
- _05_other: utilities, docs, and assorted references

The code for each section is independent so we can modify one without fucking up the other sections. 

### Misc
- Built in Python. Fast to build. I want to get better at it.
- FastAPI + static html. Fast, simple and zero build.
- Beasty Bar intro: https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf
- Beady Bar more rules: https://www.ultraboardgames.com/beasty-bar/game-rules.php


###
run the ui with:
source .venv/bin/activate
pip install -e .
uvicorn _04_ui.app:create_app --reload
python -m _03_training.tournament first diego --games 25 --seed 2025

Self-play runs accept `--eval-games` to control how many games each baseline plays per promotion check and `--eval-seed` to keep those tournaments deterministic. Drop the same knobs into a JSON config as `eval_games` / `eval_seed` when you want to drive `_03_training.self_play` from a file instead of the CLI.
