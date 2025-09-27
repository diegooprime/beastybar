"""Agent focused on maximising opponent point loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..simulator import actions, cards, engine, state
from .base import Agent, ensure_legal
from .baselines import FirstLegalAgent


@dataclass
class _ActionEvaluation:
    action: actions.Action
    score: int
    total_point_loss: int
    opponent_removed_count: int


class KillerAgent(Agent):
    """Agent that greedily converts opponent cards into losses."""

    def __init__(self) -> None:
        self._fallback = FirstLegalAgent()

    def select_action(self, game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:
        if not legal:
            raise RuntimeError("KillerAgent received no legal actions")

        legal_actions = tuple(legal)
        evaluations = [self._evaluate_action(game_state, action) for action in legal_actions]

        if all(evaluation.score == 0 for evaluation in evaluations):
            fallback = self._fallback.select_action(game_state, legal_actions)
            return ensure_legal(fallback, legal_actions)

        high_removal = [evaluation for evaluation in evaluations if evaluation.total_point_loss >= 8]
        if high_removal:
            best = max(high_removal, key=lambda item: (item.score, -item.action.hand_index, _params_key(item.action)))
            return ensure_legal(best.action, legal_actions)

        best = max(
            evaluations,
            key=lambda item: (item.score, item.opponent_removed_count, -item.action.hand_index, _params_key(item.action)),
        )
        return ensure_legal(best.action, legal_actions)

    def _evaluate_action(self, game_state: state.State, action: actions.Action) -> _ActionEvaluation:
        simulated = engine.step(game_state, action)

        bounce_cards, check_cards = _trace_losses(game_state, action)

        opponent = game_state.next_player()
        bounce_points = _points_for_owner(bounce_cards, opponent)
        check_points = _points_for_owner(check_cards, opponent)
        total_point_loss = bounce_points + check_points

        initial_queue = game_state.zones.queue
        final_queue = simulated.zones.queue
        top_two_loss = _top_two_loss(initial_queue, final_queue, opponent)
        opponent_removed_count = _opponent_removed_count(initial_queue, final_queue, opponent)

        score = 6 * bounce_points + 3 * check_points + top_two_loss
        return _ActionEvaluation(action=action, score=score, total_point_loss=total_point_loss, opponent_removed_count=opponent_removed_count)


def _trace_losses(game_state: state.State, action: actions.Action) -> tuple[list[state.Card], list[state.Card]]:
    player = game_state.active_player
    working_state, played = state.remove_hand_card(game_state, player, action.hand_index)
    working_state = state.append_queue(working_state, played)

    before_resolution = working_state.zones.thats_it
    working_state = cards.resolve_play(working_state, played, action)
    working_state = cards.process_recurring(working_state)
    after_recurring = working_state.zones.thats_it
    bounced = _new_cards(before_resolution, after_recurring)

    working_state = engine._apply_five_card_check(working_state)
    after_check = working_state.zones.thats_it
    checked = _new_cards(after_recurring, after_check)
    return bounced, checked


def _new_cards(previous: Iterable[state.Card], current: Iterable[state.Card]) -> list[state.Card]:
    seen = {id(card) for card in previous}
    return [card for card in current if id(card) not in seen]


def _points_for_owner(cards: Iterable[state.Card], owner: int) -> int:
    return sum(card.points for card in cards if card.owner == owner)


def _top_two_loss(initial_queue: Sequence[state.Card], final_queue: Sequence[state.Card], opponent: int) -> int:
    lost = 0
    remaining_ids = {id(card) for card in final_queue}
    for card in initial_queue[:2]:
        if card.owner != opponent:
            continue
        if id(card) not in remaining_ids:
            lost += card.points
    return lost


def _opponent_removed_count(initial_queue: Sequence[state.Card], final_queue: Sequence[state.Card], opponent: int) -> int:
    remaining_ids = {id(card) for card in final_queue}
    return sum(1 for card in initial_queue if card.owner == opponent and id(card) not in remaining_ids)


def _params_key(action: actions.Action) -> tuple[int, ...]:
    return action.params


__all__ = ["KillerAgent"]
