"""Uniform random baseline agent."""
from __future__ import annotations

import random
from typing import Optional, Sequence

from _01_simulator import actions, state
from .base import Agent, ensure_legal


class RandomAgent(Agent):
    """Agent that samples uniformly from the available legal actions."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:
        del game_state  # unused
        if not legal:
            raise RuntimeError("RandomAgent received no legal actions")
        choice = self._rng.randrange(len(legal))
        return ensure_legal(legal[choice], legal)


__all__ = ["RandomAgent"]
