"""Agent abstractions for the Beasty Bar simulator."""
from __future__ import annotations

import abc
from typing import Callable, Iterable, Sequence

from ..simulator import actions, state

# Alias kept in sync with beastybar.simulator.simulate.AgentFn without importing the module
# to avoid circular dependencies at import time.
AgentFn = Callable[[state.State, Sequence[actions.Action]], actions.Action]


class Agent(abc.ABC):
    """Base class for agents that choose actions given a game state."""

    def start_game(self, game_state: state.State) -> None:
        """Hook invoked when a new game begins. Override if needed."""

    def end_game(self, game_state: state.State) -> None:
        """Hook invoked after a game concludes. Override if needed."""

    def __call__(self, game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:
        return self.select_action(game_state, legal)

    @abc.abstractmethod
    def select_action(self, game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:
        """Return one legal action to play for the active player."""

    def bind(self) -> AgentFn:
        """Return a callable compatible with simulate.run's AgentFn."""

        def _wrapped(game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:
            if not legal:
                raise RuntimeError("Agent received no legal actions to choose from")
            action = self.select_action(game_state, legal)
            if action not in legal:
                raise ValueError("Agent returned an illegal action")
            return action

        return _wrapped


def ensure_legal(action: actions.Action, legal: Iterable[actions.Action]) -> actions.Action:
    """Validate that the chosen action is legal, raising otherwise."""

    if action not in legal:
        raise ValueError("Illegal action selected")
    return action


__all__ = ["Agent", "AgentFn", "ensure_legal"]
