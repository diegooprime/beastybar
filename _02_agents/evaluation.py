"""Evaluation helpers for agent decision making."""
from __future__ import annotations

from typing import Callable, Iterable

from _01_simulator import actions, engine, state

HeuristicFn = Callable[[state.State, int], float]


def evaluate_action(
    game_state: state.State,
    action: actions.Action,
    heuristic: HeuristicFn,
    *,
    perspective: int | None = None,
) -> float:
    """Return the heuristic value of applying an action from the given state."""

    next_state = engine.step(game_state, action)
    view = perspective if perspective is not None else game_state.active_player
    return heuristic(next_state, view)


def material_advantage(game_state: state.State, perspective: int) -> float:
    """Score the state from a player's perspective using lightweight heuristics."""

    zones = game_state.zones
    score = 0.0

    for card in zones.beasty_bar:
        delta = card.points if card.owner == perspective else -card.points
        score += 2.0 * delta

    queue = zones.queue
    queue_len = len(queue)
    for idx, card in enumerate(queue):
        distance = queue_len - idx
        weight = 0.3 + 0.1 * distance
        delta = card.points if card.owner == perspective else -card.points
        score += weight * delta

    for card in zones.thats_it:
        delta = card.points if card.owner == perspective else -card.points
        score -= 0.5 * delta

    return score


def best_action(
    game_state: state.State,
    legal: Iterable[actions.Action],
    heuristic: HeuristicFn,
    *,
    perspective: int | None = None,
) -> tuple[actions.Action, float]:
    """Return the best action and its score according to the heuristic."""

    view = perspective if perspective is not None else game_state.active_player
    best_score = float("-inf")
    best = None
    for action in legal:
        value = evaluate_action(game_state, action, heuristic, perspective=view)
        if best is None or value > best_score:
            best_score = value
            best = action
    if best is None:
        raise RuntimeError("No legal actions provided for best_action")
    return best, best_score


__all__ = [
    "HeuristicFn",
    "evaluate_action",
    "material_advantage",
    "best_action",
]
