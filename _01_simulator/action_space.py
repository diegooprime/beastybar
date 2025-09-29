"""Enumerate legal actions and produce fixed-size masks for RL agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from . import actions, engine, rules, state

_MAX_HAND_INDEX = rules.HAND_SIZE
_MAX_PARAM_VALUE = rules.MAX_QUEUE_LENGTH  # queue indices are 0..MAX_QUEUE_LENGTH-1
_MAX_PARAMS = 2  # Longest sequence required (chameleon copying parrot)


def _generate_catalog() -> Tuple[actions.Action, ...]:
    catalog: list[actions.Action] = []
    for hand_index in range(_MAX_HAND_INDEX):
        catalog.append(actions.Action(hand_index=hand_index))
        for first in range(_MAX_PARAM_VALUE):
            catalog.append(actions.Action(hand_index=hand_index, params=(first,)))
            for second in range(_MAX_PARAM_VALUE):
                catalog.append(actions.Action(hand_index=hand_index, params=(first, second)))
    return tuple(catalog)


_ACTION_CATALOG: Tuple[actions.Action, ...] = _generate_catalog()
_ACTION_KEY_TO_INDEX: Dict[Tuple[int, Tuple[int, ...]], int] = {
    (action.hand_index, action.params): index
    for index, action in enumerate(_ACTION_CATALOG)
}


def action_index(action: actions.Action) -> int:
    """Return the catalog index for a concrete action."""

    key = (action.hand_index, action.params)
    try:
        return _ACTION_KEY_TO_INDEX[key]
    except KeyError as exc:
        raise ValueError(f"Action not found in catalog: {action}") from exc


@dataclass(frozen=True)
class ActionSpace:
    """Fixed catalog of actions paired with a legal action mask."""

    actions: Tuple[actions.Action, ...]
    mask: Tuple[int, ...]
    legal_indices: Tuple[int, ...]

    @property
    def total_actions(self) -> int:
        return len(self.actions)


def canonical_actions() -> Tuple[actions.Action, ...]:
    """Return the canonical ordered action catalog."""

    return _ACTION_CATALOG


def legal_action_space(game_state: state.State, perspective: int) -> ActionSpace:
    """Return the fixed action catalog and mask for the perspective player."""

    if not (0 <= perspective < rules.PLAYER_COUNT):
        raise ValueError("Perspective player index out of range")

    legal = tuple(engine.legal_actions(game_state, perspective))
    mask = [0] * len(_ACTION_CATALOG)
    legal_indices: list[int] = []

    for action in legal:
        key = (action.hand_index, action.params)
        try:
            index = _ACTION_KEY_TO_INDEX[key]
        except KeyError as exc:
            raise ValueError(f"Legal action not found in catalog: {action}") from exc
        mask[index] = 1
        legal_indices.append(index)

    return ActionSpace(actions=_ACTION_CATALOG, mask=tuple(mask), legal_indices=tuple(legal_indices))


__all__ = [
    "ActionSpace",
    "action_index",
    "canonical_actions",
    "legal_action_space",
]
