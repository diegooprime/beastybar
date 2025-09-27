"""Heuristic agent implementing Diego's card-specific rules."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..simulator import actions, engine, rules, state
from .base import Agent, ensure_legal
from .baselines import FirstLegalAgent


_RANDOM_SPECIES = {"chameleon", "giraffe", "snake"}
_GUARD_SPECIES = {"crocodile", "hippo"}
_REMOVAL_THRESHOLDS: Dict[str, int] = {
    "skunk": 4,
    "parrot": 4,
    "crocodile": 5,
}
_SEAL_FRIENDLY_THRESHOLD = 4
_SEAL_REMOVAL_THRESHOLD = 5


@dataclass(frozen=True)
class ActionOutcome:
    action: actions.Action
    card: state.Card
    our_bar_delta: int
    opp_bar_delta: int
    our_points_removed: int
    opp_points_removed: int
    new_bar_cards: tuple[state.Card, ...]
    new_thats_it_cards: tuple[state.Card, ...]
    next_queue: tuple[state.Card, ...]
    card_zone: str
    card_queue_index: int | None
    card_entered_bar: bool
    monkey_preferred: bool


@dataclass(frozen=True)
class Candidate:
    outcome: ActionOutcome
    score: tuple[int, ...]

    @property
    def action(self) -> actions.Action:
        return self.outcome.action


class DiegoAgent(Agent):
    """Heuristic agent that simulates outcomes respecting Diego's rules."""

    def __init__(self) -> None:
        self._fallback = FirstLegalAgent()

    def select_action(self, game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:  # noqa: D401
        if not legal:
            raise RuntimeError("DiegoAgent received no legal actions")

        player = game_state.active_player
        opponent = game_state.next_player()
        queue = game_state.zones.queue
        grouped = _group_actions_by_hand(legal)
        rng = random.Random(_rng_seed(game_state))

        candidates: List[Candidate] = []
        for hand_index, actions_for_card in grouped.items():
            card = game_state.players[player].hand[hand_index]
            species = card.species

            if species == "zebra" and _opponent_has_guard(queue, opponent):
                # Skip zebra when the opponent is protected for now.
                continue

            if species in _RANDOM_SPECIES:
                chosen = rng.choice(actions_for_card)
                outcome = _evaluate_action(game_state, card, chosen, has_other_monkey=False)
                candidates.append(Candidate(outcome=outcome, score=_score(outcome)))
                continue

            has_other_monkey = species == "monkey" and any(c.species == "monkey" for c in queue)
            outcomes = [
                _evaluate_action(game_state, card, action, has_other_monkey=has_other_monkey)
                for action in actions_for_card
            ]
            filtered = _apply_species_rules(species, outcomes)
            if not filtered:
                continue

            for outcome in filtered:
                candidates.append(Candidate(outcome=outcome, score=_score(outcome)))

        if not candidates:
            fallback = self._fallback.select_action(game_state, legal)
            return ensure_legal(fallback, legal)

        best = max(candidates, key=lambda cand: cand.score)
        return ensure_legal(best.action, legal)


def _group_actions_by_hand(legal: Sequence[actions.Action]) -> Dict[int, List[actions.Action]]:
    grouped: Dict[int, List[actions.Action]] = {}
    for action in legal:
        grouped.setdefault(action.hand_index, []).append(action)
    return grouped


def _rng_seed(game_state: state.State) -> int:
    return game_state.seed * 31 + game_state.turn


def _opponent_has_guard(queue: Sequence[state.Card], opponent: int) -> bool:
    return any(card.owner == opponent and card.species in _GUARD_SPECIES for card in queue)


def _evaluate_action(
    game_state: state.State,
    card: state.Card,
    action: actions.Action,
    *,
    has_other_monkey: bool,
) -> ActionOutcome:
    player = game_state.active_player
    before_bar = game_state.zones.beasty_bar
    before_thats = game_state.zones.thats_it

    next_state = engine.step(game_state, action)
    bar_after = next_state.zones.beasty_bar
    thats_after = next_state.zones.thats_it
    queue_after = next_state.zones.queue

    new_bar_cards = tuple(bar_after[len(before_bar) :])
    new_thats_cards = tuple(thats_after[len(before_thats) :])

    our_bar_delta = sum(card.points for card in new_bar_cards if card.owner == player)
    opp_bar_delta = sum(card.points for card in new_bar_cards if card.owner != player)
    our_removed = sum(card.points for card in new_thats_cards if card.owner == player)
    opp_removed = sum(card.points for card in new_thats_cards if card.owner != player)

    if card in queue_after:
        zone = "queue"
        index = queue_after.index(card)
    elif card in bar_after:
        zone = "beasty_bar"
        index = None
    elif card in thats_after:
        zone = "thats_it"
        index = None
    else:
        zone = "unknown"
        index = None

    entered_bar = card in new_bar_cards
    monkey_preferred = card.species == "monkey" and has_other_monkey and entered_bar

    return ActionOutcome(
        action=action,
        card=card,
        our_bar_delta=our_bar_delta,
        opp_bar_delta=opp_bar_delta,
        our_points_removed=our_removed,
        opp_points_removed=opp_removed,
        new_bar_cards=new_bar_cards,
        new_thats_it_cards=new_thats_cards,
        next_queue=queue_after,
        card_zone=zone,
        card_queue_index=index,
        card_entered_bar=entered_bar,
        monkey_preferred=monkey_preferred,
    )


def _apply_species_rules(species: str, outcomes: Iterable[ActionOutcome]) -> List[ActionOutcome]:
    if species in _REMOVAL_THRESHOLDS:
        threshold = _REMOVAL_THRESHOLDS[species]
        filtered = [o for o in outcomes if o.opp_points_removed >= threshold]
        if filtered:
            return filtered
        return []

    if species == "kangaroo":
        safe = [o for o in outcomes if o.card_zone != "thats_it"]
        return safe or list(outcomes)

    if species == "monkey":
        preferred = [o for o in outcomes if o.monkey_preferred]
        return preferred or list(outcomes)

    if species == "seal":
        filtered = [
            o
            for o in outcomes
            if o.our_bar_delta >= _SEAL_FRIENDLY_THRESHOLD
            or o.opp_points_removed >= _SEAL_REMOVAL_THRESHOLD
        ]
        if filtered:
            return filtered
        return []

    if species == "zebra":
        return list(outcomes)

    if species == "hippo":
        filtered = [
            o
            for o in outcomes
            if (o.card_zone == "queue" and o.card_queue_index == 0)
            or (o.card_zone == "beasty_bar" and o.card_entered_bar)
        ]
        if filtered:
            return filtered
        return []

    if species == "lion":
        filtered = [
            o
            for o in outcomes
            if (o.card_zone == "queue" and o.card_queue_index == 0)
            or (o.card_zone == "beasty_bar" and o.card_entered_bar)
        ]
        if filtered:
            return filtered
        return []

    if species in {"chameleon", "giraffe", "snake"}:
        return list(outcomes)

    return list(outcomes)


def _score(outcome: ActionOutcome) -> tuple[int, ...]:
    net_bar = outcome.our_bar_delta - outcome.opp_bar_delta
    opp_removed = outcome.opp_points_removed
    our_gain = outcome.our_bar_delta
    zone_rank = _zone_rank(outcome)
    position_score = _position_score(outcome)
    avoid_self_loss = -outcome.our_points_removed
    hand_index = -outcome.action.hand_index
    params_length = -len(outcome.action.params)
    return (
        net_bar,
        opp_removed,
        our_gain,
        zone_rank,
        position_score,
        avoid_self_loss,
        hand_index,
        params_length,
    ) + outcome.action.params


def _zone_rank(outcome: ActionOutcome) -> int:
    if outcome.card_zone == "beasty_bar":
        return 2
    if outcome.card_zone == "queue":
        return 1
    if outcome.card_zone == "thats_it":
        return 0
    return -1


def _position_score(outcome: ActionOutcome) -> int:
    if outcome.card_zone == "queue" and outcome.card_queue_index is not None:
        return rules.MAX_QUEUE_LENGTH - outcome.card_queue_index
    if outcome.card_zone == "beasty_bar":
        return rules.MAX_QUEUE_LENGTH + 1
    return -rules.MAX_QUEUE_LENGTH


__all__ = ["DiegoAgent"]
