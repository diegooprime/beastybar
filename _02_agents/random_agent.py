"""Random agent implementation."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .base import Agent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _01_simulator import actions, state


class RandomAgent(Agent):
    """Agent that selects uniformly random legal actions."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        return self._rng.choice(list(legal_actions))


__all__ = ["RandomAgent"]
