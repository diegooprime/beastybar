"""Enumerate legal actions and produce fixed-size masks for RL agents.

Action Catalog Structure:
    The action catalog is a fixed-size tuple of all possible actions in the game.
    Each action is indexed by its position in the catalog, creating a mapping:
        index -> Action(hand_index, params)

    Actions are ordered as:
        - For each hand_index in [0, HAND_SIZE):
            - Action(hand_index, ())           # No params
            - For first in [0, MAX_QUEUE_LENGTH):
                - Action(hand_index, (first,)) # Single param
                - For second in [0, MAX_QUEUE_LENGTH):
                    - Action(hand_index, (first, second)) # Two params

    This ordering produces ACTION_DIM total actions (typically 124).
    The catalog index is used as the action dimension for neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import actions, engine, rules, state

_MAX_HAND_INDEX = rules.HAND_SIZE
_MAX_PARAM_VALUE = rules.MAX_QUEUE_LENGTH  # queue indices are 0..MAX_QUEUE_LENGTH-1
_MAX_PARAMS = 2  # Longest sequence required (chameleon copying parrot)


def _generate_catalog() -> tuple[actions.Action, ...]:
    catalog: list[actions.Action] = []
    for hand_index in range(_MAX_HAND_INDEX):
        catalog.append(actions.Action(hand_index=hand_index))
        for first in range(_MAX_PARAM_VALUE):
            catalog.append(actions.Action(hand_index=hand_index, params=(first,)))
            for second in range(_MAX_PARAM_VALUE):
                catalog.append(actions.Action(hand_index=hand_index, params=(first, second)))
    return tuple(catalog)


_ACTION_CATALOG: tuple[actions.Action, ...] = _generate_catalog()
_ACTION_KEY_TO_INDEX: dict[tuple[int, tuple[int, ...]], int] = {
    (action.hand_index, action.params): index for index, action in enumerate(_ACTION_CATALOG)
}

# Total number of actions in the catalog (dimension of action space)
ACTION_DIM: int = len(_ACTION_CATALOG)


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

    actions: tuple[actions.Action, ...]
    mask: tuple[int, ...]
    legal_indices: tuple[int, ...]

    @property
    def total_actions(self) -> int:
        return len(self.actions)


def canonical_actions() -> tuple[actions.Action, ...]:
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


def action_mask_to_tensor(action_space: ActionSpace) -> npt.NDArray[np.float32]:
    """Convert action mask tuple to float32 numpy tensor.

    Args:
        action_space: ActionSpace containing the mask tuple.

    Returns:
        Float32 numpy array of shape (ACTION_DIM,) with values 0.0 or 1.0.
        1.0 indicates a legal action, 0.0 indicates an illegal action.
    """
    return np.array(action_space.mask, dtype=np.float32)


def legal_action_mask_tensor(game_state: state.State, perspective: int) -> npt.NDArray[np.float32]:
    """Generate legal action mask tensor for a game state.

    Convenience function combining legal_action_space() and action_mask_to_tensor().

    Args:
        game_state: Current game state.
        perspective: Player index to generate mask for (0 or 1).

    Returns:
        Float32 numpy array of shape (ACTION_DIM,) with 1.0 for legal actions,
        0.0 for illegal actions.
    """
    action_space = legal_action_space(game_state, perspective)
    return action_mask_to_tensor(action_space)


def batch_action_masks(states: list[state.State], perspectives: list[int]) -> npt.NDArray[np.float32]:
    """Generate batch of action masks for multiple states.

    Args:
        states: List of game states.
        perspectives: List of player indices, one per state.

    Returns:
        Float32 numpy array of shape (batch_size, ACTION_DIM) where each row
        is the action mask for the corresponding state and perspective.

    Raises:
        ValueError: If states and perspectives have different lengths.
    """
    if len(states) != len(perspectives):
        raise ValueError(f"States ({len(states)}) and perspectives ({len(perspectives)}) must have same length")

    masks = np.zeros((len(states), ACTION_DIM), dtype=np.float32)
    for i, (game_state, perspective) in enumerate(zip(states, perspectives, strict=True)):
        masks[i] = legal_action_mask_tensor(game_state, perspective)

    return masks


def index_to_action(index: int) -> actions.Action:
    """Convert action catalog index to Action object.

    Reverse lookup from neural network output index to concrete Action.

    Args:
        index: Action catalog index in range [0, ACTION_DIM).

    Returns:
        Action object corresponding to the index.

    Raises:
        IndexError: If index is out of range.
    """
    if not (0 <= index < ACTION_DIM):
        raise IndexError(f"Action index {index} out of range [0, {ACTION_DIM})")
    return _ACTION_CATALOG[index]


def sample_masked_action(logits: npt.NDArray[np.float32], mask: npt.NDArray[np.float32], temperature: float = 1.0) -> int:
    """Sample action index from logits with mask and temperature scaling.

    Applies action mask to logits (setting illegal actions to -inf),
    applies temperature scaling, and samples from the resulting softmax distribution.

    Args:
        logits: Raw network output logits of shape (ACTION_DIM,).
        mask: Legal action mask of shape (ACTION_DIM,) with 1.0 for legal actions.
        temperature: Temperature for softmax scaling. Higher = more random, lower = more greedy.
            Must be positive. Default is 1.0 (no scaling).

    Returns:
        Sampled action index.

    Raises:
        ValueError: If no legal actions available (all mask values are 0).
        ValueError: If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    if not np.any(mask > 0):
        raise ValueError("No legal actions available (mask is all zeros)")

    # Apply mask: set illegal actions to -inf
    masked_logits = np.where(mask > 0, logits, -np.inf)

    # Apply temperature scaling
    scaled_logits = masked_logits / temperature

    # Compute softmax probabilities
    # Subtract max for numerical stability
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)

    # Sample from distribution
    return int(np.random.choice(ACTION_DIM, p=probs))


def greedy_masked_action(logits: npt.NDArray[np.float32], mask: npt.NDArray[np.float32]) -> int:
    """Select highest-logit legal action (greedy/deterministic selection).

    Args:
        logits: Raw network output logits of shape (ACTION_DIM,).
        mask: Legal action mask of shape (ACTION_DIM,) with 1.0 for legal actions.

    Returns:
        Index of highest logit among legal actions.

    Raises:
        ValueError: If no legal actions available (all mask values are 0).
    """
    if not np.any(mask > 0):
        raise ValueError("No legal actions available (mask is all zeros)")

    # Apply mask: set illegal actions to -inf
    masked_logits = np.where(mask > 0, logits, -np.inf)

    # Return argmax
    return int(np.argmax(masked_logits))


__all__ = [
    "ACTION_DIM",
    "ActionSpace",
    "action_index",
    "action_mask_to_tensor",
    "batch_action_masks",
    "canonical_actions",
    "greedy_masked_action",
    "index_to_action",
    "legal_action_mask_tensor",
    "legal_action_space",
    "sample_masked_action",
]
