"""Reward helpers for self-play reinforcement learning."""
from __future__ import annotations

import random
from typing import Tuple

from . import engine, rules, state

_TOTAL_POINTS_PER_PLAYER = sum(rules.SPECIES[name].points for name in rules.BASE_DECK)
_TOTAL_POINTS_BOUNDS = _TOTAL_POINTS_PER_PLAYER or 1


def win_loss(game_state: state.State) -> Tuple[int, ...]:
    """Return +1/-1/0 win-loss reward for each player."""

    scores = engine.score(game_state)
    if len(scores) != rules.PLAYER_COUNT:
        raise ValueError("Score vector must match player count")

    if scores[0] == scores[1]:
        return tuple(0 for _ in scores)

    winner = 0 if scores[0] > scores[1] else 1
    rewards = [-1 for _ in scores]
    rewards[winner] = 1
    return tuple(rewards)


def normalized_margin(game_state: state.State) -> Tuple[float, ...]:
    """Return normalized point margin rewards for each player."""

    scores = engine.score(game_state)
    if len(scores) != rules.PLAYER_COUNT:
        raise ValueError("Score vector must match player count")

    margin = scores[0] - scores[1]
    normalized = margin / _TOTAL_POINTS_BOUNDS
    return (normalized, -normalized)


def shaped_reward(
    game_state: state.State,
    *,
    margin_weight: float = 0.25,
    jitter_scale: float = 0.01,
    seed: int | None = None,
) -> Tuple[float, ...]:
    """Combine win/loss and margin signals with deterministic jitter.

    Args:
        game_state: Terminal state to score.
        margin_weight: Weight applied to the normalized margin component.
        jitter_scale: Maximum absolute jitter added for tie-breaking.
        seed: Optional explicit seed for deterministic jitter. Defaults to
            ``game_state.seed`` when not provided.
    """

    wl = win_loss(game_state)
    margin = normalized_margin(game_state)

    rewards = [wl[idx] + margin_weight * margin[idx] for idx in range(rules.PLAYER_COUNT)]
    if jitter_scale <= 0:
        return tuple(rewards)

    rng = random.Random(game_state.seed if seed is None else seed)
    jitter = rng.uniform(-jitter_scale, jitter_scale)
    rewards[0] += jitter
    rewards[1] -= jitter
    return tuple(rewards)


__all__ = [
    "win_loss",
    "normalized_margin",
    "shaped_reward",
]
