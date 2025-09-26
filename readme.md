## Overview

This project has **three layers**:

1. **Simulator** — the official Beasty Bar rules. 
   * **Engine + UI** for humans to play the game.
   * **Batch interface** for agents to play millions of games automatically.
2. **Agents** — algorithms that use the simulator to make decisions (random, greedy, search, learning).
3. **Training and evaluation** — experiments to find strong strategies against baselines and humans.

We are currently focused only on **(1) the simulator**.

---

## Simulator specification

The simulator has **two parts**:

### 1. Engine + UI (for humans)

* Pure rule engine implementing every animal action, turn order, and point-based scoring.
* Lightweight web UI to let a human play against bots.
* Explanations panel: shows legal moves, what happened in a turn.
* Deterministic, seeded shuffles for reproducibility.

### 2. Batch simulation (for agents)

* Same rule engine, but exposed as a fast API:

  * `new_game(seed)`
  * `legal_actions(state, player)`
  * `apply(state, action)`
  * `is_terminal(state)`
  * `score(state)`
* Allows agents to run **millions of games** with fixed seeds for training or evaluation.
* Supports structured logs and replay serialization.

---

## After the simulator

Once the simulator is stable and tested:

* Build baseline agents (Random, Greedy).
* Add interpretable search (Information-Set MCTS with heuristic rollouts).
* Run tournaments and human playtests to measure strength.
* Optimize for both **strength** and **explainability**.

---

## Project layout

* `rules.md` — full rules, point table, pseudo-code.
* `beastybar/`

  * `rules.py` — constants, species, strengths, points.
  * `state.py` — immutable state model.
  * `cards.py` — animal actions.
  * `engine.py` — step function, legal moves, scoring.
  * `ui/` — human-play interface.
  * `agents/` — agent code (later).
  * `simulate.py` — batch simulation entry point.
* `tests/` — unit tests + golden replays.

---