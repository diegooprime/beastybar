"""First legal baseline agent."""
from __future__ import annotations

from typing import Sequence

from _01_simulator import actions, state
from .base import Agent


class FirstLegalAgent(Agent):
    """Deterministic agent that always picks the first legal action."""

    def select_action(self, game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:  # noqa: D401
        del game_state  # unused
        if not legal:
            raise RuntimeError("FirstLegalAgent received no legal actions")
        return legal[0]


__all__ = ["FirstLegalAgent"]
