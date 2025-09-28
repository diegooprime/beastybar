"""FrontRunner agent focused on controlling the front of the queue."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from simulator import actions, cards, rules, state
from .base import Agent
from .baselines import FirstLegalAgent


@dataclass(frozen=True)
class _SimulationTrace:
    final_state: state.State
    played_card: state.Card
    queue_before: Sequence[state.Card]
    queue_after_resolve: Sequence[state.Card]
    queue_after_recurring: Sequence[state.Card]
    entrants: tuple[state.Card, ...]
    bar_before: Sequence[state.Card]
    thats_before: Sequence[state.Card]


@dataclass(frozen=True)
class _Evaluation:
    score: int
    own_front_points: int
    opp_front_points: int


class FrontRunnerAgent(Agent):
    """Agent that aggressively seeks to occupy the first two queue slots."""

    def __init__(self) -> None:
        self._fallback = FirstLegalAgent()

    def select_action(self, game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:
        if not legal:
            raise RuntimeError("FrontRunnerAgent received no legal actions")

        perspective = game_state.active_player
        best_action: actions.Action | None = None
        best_key: tuple[int, int, int, int] | None = None

        for action in legal:
            evaluation = self._evaluate_action(game_state, action, perspective)
            if evaluation is None:
                continue

            key = (
                evaluation.score,
                evaluation.own_front_points,
                -evaluation.opp_front_points,
                -action.hand_index,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_action = action

        if best_action is None:
            return self._fallback.select_action(game_state, legal)
        return best_action

    def _evaluate_action(
        self,
        game_state: state.State,
        action: actions.Action,
        perspective: int,
    ) -> _Evaluation | None:
        trace = self._simulate(game_state, action)
        final_queue = trace.final_state.zones.queue

        if self._reject_lion(game_state, action, final_queue, perspective):
            return None
        if self._reject_lone_front(final_queue, perspective):
            return None

        score = 0
        own_front_points = 0
        opp_front_points = 0

        for card in final_queue[:2]:
            if card.owner == perspective:
                score += 8
                own_front_points += card.points
            else:
                score -= 7
                opp_front_points += card.points

        bar_after = trace.final_state.zones.beasty_bar
        thats_after = trace.final_state.zones.thats_it

        for card in _added_cards(trace.bar_before, bar_after):
            delta = 3 * card.points
            if card.owner == perspective:
                score += delta
            else:
                score -= delta

        for card in _added_cards(trace.thats_before, thats_after):
            delta = 4 * card.points
            if card.owner == perspective:
                score -= delta
            else:
                score += delta

        removed_after_recurring = _removed_cards(trace.queue_after_resolve, trace.queue_after_recurring)
        own_removed = [card for card in removed_after_recurring if card.owner == perspective]
        if own_removed:
            score -= 5 * len(own_removed)

        if trace.entrants:
            own_entrants = sum(1 for card in trace.entrants if card.owner == perspective)
            opp_entrants = len(trace.entrants) - own_entrants
            if own_entrants - opp_entrants >= 1:
                score += 2

        return _Evaluation(score=score, own_front_points=own_front_points, opp_front_points=opp_front_points)

    def _simulate(self, game_state: state.State, action: actions.Action) -> _SimulationTrace:
        player = game_state.active_player
        queue_before = game_state.zones.queue
        bar_before = game_state.zones.beasty_bar
        thats_before = game_state.zones.thats_it

        working_state, played_card = state.remove_hand_card(game_state, player, action.hand_index)
        working_state = state.append_queue(working_state, played_card)
        working_state = cards.resolve_play(working_state, played_card, action)
        queue_after_resolve = working_state.zones.queue
        working_state = cards.process_recurring(working_state)
        queue_after_recurring = working_state.zones.queue
        working_state, entrants = _apply_five_card_check(working_state)

        working_state, _ = state.draw_card(working_state, player)
        next_player = working_state.next_player()
        working_state = state.set_active_player(working_state, next_player, advance_turn=True)

        return _SimulationTrace(
            final_state=working_state,
            played_card=played_card,
            queue_before=queue_before,
            queue_after_resolve=queue_after_resolve,
            queue_after_recurring=queue_after_recurring,
            entrants=entrants,
            bar_before=bar_before,
            thats_before=thats_before,
        )

    def _reject_lion(
        self,
        game_state: state.State,
        action: actions.Action,
        final_queue: Sequence[state.Card],
        perspective: int,
    ) -> bool:
        card_to_play = game_state.players[perspective].hand[action.hand_index]
        if card_to_play.species != "lion":
            return False
        if not final_queue:
            return True
        leader = final_queue[0]
        return leader.species != "lion" or leader.owner != perspective

    def _reject_lone_front(self, final_queue: Sequence[state.Card], perspective: int) -> bool:
        our_cards = [card for card in final_queue if card.owner == perspective]
        if len(our_cards) != 1:
            return False
        if not final_queue or final_queue[0].owner != perspective:
            return False
        if len(final_queue) < 2:
            return False
        behind = final_queue[1]
        return behind.owner != perspective and behind.species in {"seal", "crocodile"}


def _apply_five_card_check(game_state: state.State) -> tuple[state.State, tuple[state.Card, ...]]:
    queue = game_state.zones.queue
    if len(queue) != rules.MAX_QUEUE_LENGTH:
        return game_state, ()

    entering = queue[:2]
    bounced = queue[-1]
    remaining = queue[2:-1]

    game_state = state.replace_queue(game_state, remaining)
    for card in entering:
        game_state = state.push_to_zone(game_state, rules.ZONE_BEASTY_BAR, card)
    game_state = state.push_to_zone(game_state, rules.ZONE_THATS_IT, bounced)
    return game_state, entering


def _added_cards(before: Sequence[state.Card], after: Sequence[state.Card]) -> Iterable[state.Card]:
    before_counts = Counter(before)
    after_counts = Counter(after)
    return list((after_counts - before_counts).elements())


def _removed_cards(before: Sequence[state.Card], after: Sequence[state.Card]) -> Iterable[state.Card]:
    before_counts = Counter(before)
    after_counts = Counter(after)
    return list((before_counts - after_counts).elements())


__all__ = ["FrontRunnerAgent"]
