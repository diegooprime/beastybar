Beasty Bar is a board game I like a lot. It's fun, simple, easy to learn and requires strategy. 
I played a ton with friends and wanted the “best” strategy. I searched online. Nothing.
So I’m building a simulator to test strategies until I find the best one. Then I’ll use the engine to sharpen my intuition and win more games.
First project of this type for me.
Purpose: Have the best Beasty Bar strategy in the world.

### Project structure:
- simulator: full game rules and state management
- agents: each strategy as an agent implementation
- training: tournaments, telemetry tooling, logs, Elo tracking
- user_interface: FastAPI interface and static viewer
- other: utilities, docs, and assorted references

The code for each section is independent so we can modify one without fucking up the other sections. 

### Misc
- Built in Python. Fast to build. I want to get better at it.
- FastAPI + static html. Fast, simple and zero build.
- Beasty Bar intro: https://tesera.ru/images/items/1525203/BeastyBar_EN-online.pdf


###
run the ui with:
uvicorn user_interface.app:create_app --reload
source .venv/bin/activate
pip install -e .
python -m training.tournament first diego --games 25 --seed 2025


