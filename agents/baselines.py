"""Baseline agent implementations."""
from __future__ import annotations

import random
from typing import Optional, Sequence

from simulator import actions, state
from .base import Agent, ensure_legal


class FirstLegalAgent(Agent):
    """Deterministic agent that always picks the first legal action."""

    def select_action(self, game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:  # noqa: D401
        del game_state  # unused
        if not legal:
            raise RuntimeError("FirstLegalAgent received no legal actions")
        return legal[0]


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


__all__ = ["FirstLegalAgent", "RandomAgent"]
