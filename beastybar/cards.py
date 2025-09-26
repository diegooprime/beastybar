"""Card-specific action implementations."""
from __future__ import annotations

from typing import Callable, Dict

from . import actions, rules, state


Handler = Callable[[state.State, state.Card, actions.Action], state.State]


def resolve_play(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    """Apply species-specific effects after a card enters the queue."""

    handler = _HANDLERS.get(card.species, _noop_handler)
    return handler(game_state, card, action)


def process_recurring(game_state: state.State) -> state.State:
    """Process recurring abilities in gate-to-bounce order.

    Placeholder implementation until species logic is filled in.
    """

    return game_state


def _noop_handler(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    return game_state


_HANDLERS: Dict[str, Handler] = {}


__all__ = [
    "resolve_play",
    "process_recurring",
]
