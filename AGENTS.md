# Repository Guidelines

## Project Structure & Module Organization
- `_01_simulator/`: core rules, state transitions, and action logic; keep deterministic and side-effect free.
- `_02_agents/`: strategies inheriting from `_02_agents.base.Agent`; export new classes via `_02_agents/__init__.py` for discovery.
- `_03_training/`: tournament runners, telemetry, and Elo tracking; CLI entry point lives in `tournament.py`.
- `_04_ui/`: FastAPI surface exposed by `create_app` plus static assets under `_04_ui/static/`.
- `_05_other/tests/`: pytest suite covering rules, agents, replay, and UI glue; add new coverage with `test_*.py` modules.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create an isolated Python 3 environment before installing deps.
- `python3 -m pip install -e .`: install the package in editable mode so cross-package imports resolve while iterating.
- `uvicorn _04_ui.app:create_app --reload`: launch the API and static viewer with auto-reload for local testing.
- `pytest _05_other/tests -ra`: run the regression suite with failure summaries; add `-k` to focus subsets.
- `python3 -m _03_training.tournament first diego --games 25 --seed 2025`: sample tournament invocation; tweak agent names, game count, or seeds when experimenting.
- `python3 -m _03_training.tournament self-play diego --self-play-manifest _03_training/artifacts/champion.json --games 200`: evaluate the latest self-play checkpoint against a baseline using the manifest-driven loader.
- `python3 -m _03_training.self_play --config path/to/self_play_config.json`: stage artifact directories and a manifest before launching PPO workers (config file optional). Use `--eval-games N` and `--eval-seed S` to tune evaluation tournaments; JSON configs mirror these flags via `eval_games` / `eval_seed` keys so runs stay reproducible.

## Coding Style & Naming Conventions
- Code targets Python 3.10+; rely on `python3` when invoking modules or tooling, use 4-space indentation, type hints, and dataclasses (see `_01_simulator/state.py`).
- Keep modules pure and deterministic; thread randomness through explicit `seed` parameters.
- Name agents `PascalCaseAgent` and ensure they are surfaced in `_02_agents/__init__.py` so the UI can list them.
- Align function names, docstrings, and constants with Beasty Bar terminology for clarity.

## Testing Guidelines
- Write pytest cases under `_05_other/tests/test_*.py`, mirroring the module under test.
- Use deterministic fixtures by seeding games and reusing helpers from `_01_simulator` rather than ad-hoc mocks.
- Keep generated logs or large artifacts out of version control; store them under ignored paths if needed.

## Commit & Pull Request Guidelines
- Keep commits small and focused; prefer imperative, lower-case subjects (`mask opponents cards`, `fix elo`).
- State behavioral intent in the commit body when simulator math or agent heuristics change.
- PR descriptions should summarize motivation, call out tests run (`pytest`, tournament smoke runs, UI clicks), and link issues when relevant.

## Self-play manifest format
- Store loader metadata as JSON so both tournaments and training runs can reuse the same manifest (e.g., `_03_training/artifacts/<run_id>/champion.json`).
- Required fields:
  - `model`: dotted path in `module:function` form that returns a callable model when invoked.
  - Optional `checkpoint`: relative path resolved against the manifest directory; passed to the factory via the `checkpoint` keyword.
  - Optional `factoryArgs` / `factoryKwargs`: additional arguments for the factory. `factoryKwargs` must encode JSON-serialisable data (paths become strings).
- The manifest may embed `exploration` overrides (`temperature`, `epsilon`). CLI flags `--self-play-temperature` and `--self-play-epsilon` take precedence when provided.
- Example manifest:
  ```json
  {
    "model": "_03_training.policy_loader:load_policy",
    "checkpoint": "checkpoints/step_120000.pt",
    "factoryKwargs": {"device": "cpu"},
    "exploration": {"temperature": 0.0, "epsilon": 0.05}
  }
  ```
- Place shared manifests alongside checkpoints so round-robin tournaments can include `self-play` automatically by passing `--self-play-manifest`.
- Sample CLI config: `_03_training/configs/self_play_local.json` seeds a short CPU-friendly run you can tweak locally.
