"""Data-driven agent derived from 50k self-play heuristics."""
from __future__ import annotations

from typing import Sequence

from .. import actions, engine, state
from .base import Agent, ensure_legal
from .greedy import GreedyAgent


# Aggregated ranking scores from the 50k-game telemetry analysis.
# Duplicate entries (e.g. from multiple agents) are summed to reflect
# reinforcing evidence for the same context.
_HEURISTIC_WEIGHTS = {
    ("giraffe", "mid", "ally"): 0.884 + 0.833,
    ("hippo", "mid", "ally"): 0.883,
    ("zebra", "mid", "ally"): 0.872,
    ("lion", "mid", "ally"): 0.862,
    ("kangaroo", "mid", "ally"): 0.861 + 0.831,
    ("monkey", "mid", "ally"): 0.859,
    ("parrot", "mid", "ally"): 0.849,
    ("chameleon", "mid", "ally"): 0.842,
}

_HEURISTIC_SCALE = 12.0


class Heuristic50kAgent(Agent):
    """Rule-based agent prioritising the strongest self-play heuristics."""

    def __init__(self) -> None:
        self._fallback = GreedyAgent()

    def start_game(self, game_state: state.State) -> None:  # noqa: D401 - inherit docstring
        self._fallback.start_game(game_state)

    def end_game(self, game_state: state.State) -> None:  # noqa: D401 - inherit docstring
        self._fallback.end_game(game_state)

    def select_action(
        self,
        game_state: state.State,
        legal: Sequence[actions.Action],
    ) -> actions.Action:
        if not legal:
            raise RuntimeError("Heuristic50kAgent received no legal actions")

        player = game_state.active_player
        queue = game_state.zones.queue
        bucket = _bucket_queue_len(len(queue))
        front_owner = _queue_front_owner(queue, player)

        best_action: actions.Action | None = None
        best_score = float("-inf")
        matched_rule = False

        for action in legal:
            card = game_state.players[player].hand[action.hand_index]
            base = _base_card_score(card)
            bonus = _heuristic_bonus(card.species, bucket, front_owner)
            lookahead = _lookahead_bonus(game_state, action, player)
            total = base + bonus + lookahead

            if bonus > 0:
                matched_rule = True

            if total > best_score or (
                total == best_score and best_action and action.hand_index < best_action.hand_index
            ):
                best_score = total
                best_action = action

        if not matched_rule:
            fallback = self._fallback.select_action(game_state, legal)
            return ensure_legal(fallback, legal)

        assert best_action is not None  # for type checkers; legal ensured non-empty
        return ensure_legal(best_action, legal)


def _base_card_score(card: state.Card) -> float:
    # Weight points higher than strength to reflect scoring pressure.
    return card.points * 3.0 + card.strength * 0.25


def _heuristic_bonus(species: str, bucket: str, front_owner: str) -> float:
    weight = _HEURISTIC_WEIGHTS.get((species, bucket, front_owner))
    if weight is None:
        return 0.0
    return weight * _HEURISTIC_SCALE


def _lookahead_bonus(game_state: state.State, action: actions.Action, player: int) -> float:
    next_state = engine.step(game_state, action)
    bar_delta = _bar_point_delta(next_state, player)
    queue_front_bonus = 1.2 if _queue_front_owner(next_state.zones.queue, player) == "ally" else 0.0
    queue_balance = _queue_balance(next_state.zones.queue, player) * 0.4
    return bar_delta * 0.75 + queue_front_bonus + queue_balance


def _bucket_queue_len(length: int) -> str:
    if length == 0:
        return "empty"
    if length <= 2:
        return "short"
    if length <= 4:
        return "mid"
    return "full"


def _queue_front_owner(queue: Sequence[state.Card], player: int) -> str:
    if not queue:
        return "none"
    return "ally" if queue[0].owner == player else "enemy"


def _bar_point_delta(game_state: state.State, player: int) -> float:
    ally = sum(card.points for card in game_state.zones.beasty_bar if card.owner == player)
    enemy = sum(card.points for card in game_state.zones.beasty_bar if card.owner != player)
    return ally - enemy


def _queue_balance(queue: Sequence[state.Card], player: int) -> float:
    if not queue:
        return 0.0
    ally_cards = sum(1 for card in queue if card.owner == player)
    enemy_cards = len(queue) - ally_cards
    return ally_cards - enemy_cards


__all__ = ["Heuristic50kAgent"]
