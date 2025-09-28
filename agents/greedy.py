"""Greedy agent powered by configurable heuristics."""
from __future__ import annotations

from typing import Sequence

from simulator import actions, state
from .base import Agent, ensure_legal
from .evaluation import HeuristicFn, best_action, material_advantage


class GreedyAgent(Agent):
    """Agent that chooses the action with the highest heuristic value."""

    def __init__(self, heuristic: HeuristicFn | None = None) -> None:
        self._heuristic = heuristic or material_advantage

    def select_action(self, game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:
        if not legal:
            raise RuntimeError("GreedyAgent received no legal actions")
        action, _ = best_action(game_state, legal, self._heuristic)
        return ensure_legal(action, legal)


__all__ = ["GreedyAgent"]
