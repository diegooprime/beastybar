"""Base agent interface and type definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from _01_simulator import actions, state

AgentFn = Callable[[state.State, Sequence[actions.Action]], actions.Action]


class Agent(ABC):
    """Abstract base class for Beasty Bar agents."""

    @abstractmethod
    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select an action from the available legal moves.

        Args:
            game_state: The current game state (masked for opponent info).
            legal_actions: Available legal actions to choose from.

        Returns:
            The selected action.
        """
        ...

    def __call__(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Make the agent callable to satisfy AgentFn interface."""
        return self.select_action(game_state, legal_actions)

    @property
    def name(self) -> str:
        """Return the agent's name for display purposes."""
        return self.__class__.__name__


__all__ = ["Agent", "AgentFn"]
