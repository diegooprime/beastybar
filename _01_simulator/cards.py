"""Card-specific action implementations."""
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Callable, Dict, List, Optional, Tuple

from . import actions, rules, state


Handler = Callable[[state.State, state.Card, actions.Action], state.State]


def resolve_play(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    """Apply species-specific effects after a card enters the queue."""

    handler = _HANDLERS.get(card.species, _noop_handler)
    return handler(game_state, card, action)


def process_recurring(game_state: state.State) -> state.State:
    """Process recurring abilities from Heaven's Gate toward the bounce."""

    game_state, _ = process_recurring_with_trace(game_state)
    return game_state


def process_recurring_with_trace(game_state: state.State) -> Tuple[state.State, Tuple[str, ...]]:
    """Process recurring abilities and return a textual trace of outcomes."""

    events: List[str] = []
    index = 0
    while index < len(game_state.zones.queue):
        queue = game_state.zones.queue
        card = queue[index]
        species = card.species
        if species == "hippo":
            game_state, index, message = _recurring_hippo(game_state, index)
            events.append(message)
        elif species == "crocodile":
            game_state, index, message = _recurring_crocodile(game_state, index)
            events.append(message)
        elif species == "giraffe":
            game_state, index, message = _recurring_giraffe(game_state, index)
            events.append(message)
        else:
            index += 1
    return game_state, tuple(events)


def _card_label(card: state.Card) -> str:
    species = card.species.replace("_", " ").title()
    return f"{species} (P{card.owner})"


def _resolve_lion(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    queue = list(game_state.zones.queue)
    lion_count = sum(1 for c in queue if c.species == "lion")

    if lion_count > 1:
        queue.pop(queue.index(card))
        game_state = state.replace_queue(game_state, queue)
        return state.push_to_zone(game_state, rules.ZONE_THATS_IT, card)

    remaining: List[state.Card] = []
    scared: List[state.Card] = []
    for c in queue:
        if c.species == "monkey" and c is not card:
            scared.append(c)
        elif c is not card:
            remaining.append(c)

    for monkey in scared:
        game_state = state.push_to_zone(game_state, rules.ZONE_THATS_IT, monkey)

    new_queue = [card] + remaining
    return state.replace_queue(game_state, new_queue)


def _resolve_snake(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    queue = list(game_state.zones.queue)
    queue.sort(key=lambda c: c.strength, reverse=True)
    return state.replace_queue(game_state, queue)


def _resolve_giraffe(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    queue = list(game_state.zones.queue)
    idx = queue.index(card)
    if idx > 0 and queue[idx - 1].strength < card.strength:
        queue[idx], queue[idx - 1] = queue[idx - 1], queue[idx]
        return state.replace_queue(game_state, queue)
    return game_state


def _resolve_kangaroo(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    queue = list(game_state.zones.queue)
    idx = queue.index(card)
    if idx == 0:
        return game_state
    hop = min(2, idx)
    if action.params:
        hop = min(max(action.params[0], 0), hop)
    queue.pop(idx)
    queue.insert(idx - hop, card)
    return state.replace_queue(game_state, queue)


def _resolve_monkey(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    queue = list(game_state.zones.queue)
    monkey_count = sum(1 for c in queue if c.species == "monkey")
    if monkey_count < 2:
        return game_state

    survivors: List[state.Card] = []
    for c in queue:
        if c.species in {"hippo", "crocodile"}:
            game_state = state.push_to_zone(game_state, rules.ZONE_THATS_IT, c)
        else:
            survivors.append(c)

    monkeys = [c for c in survivors if c.species == "monkey"]
    others = [c for c in survivors if c.species != "monkey"]

    mates = [m for m in monkeys if m is not card]
    mates.reverse()

    new_queue = [card] + mates + others
    return state.replace_queue(game_state, new_queue)


def _resolve_parrot(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    if not action.params:
        raise ValueError("Parrot requires a target index parameter")

    target_index = action.params[0]
    queue = game_state.zones.queue
    if not (0 <= target_index < len(queue)):
        raise ValueError("Parrot target index out of range")

    game_state, target = state.remove_queue_card(game_state, target_index)
    game_state = state.push_to_zone(game_state, rules.ZONE_THATS_IT, target)
    queue_after = game_state.zones.queue
    try:
        parrot_index = queue_after.index(card)
    except ValueError:  # pragma: no cover - defensive safeguard
        return game_state
    game_state, _ = state.remove_queue_card(game_state, parrot_index)
    queue_without_actor = game_state.zones.queue
    insert_index = 0
    for existing in queue_without_actor:
        if existing.species == "parrot":
            insert_index += 1
        else:
            break
    game_state = state.insert_queue(game_state, insert_index, card)
    return game_state


def _resolve_seal(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    queue = tuple(reversed(game_state.zones.queue))
    return state.replace_queue(game_state, queue)


def _resolve_chameleon(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    if not action.params:
        raise ValueError("Chameleon requires the index of a species to imitate")

    queue = game_state.zones.queue
    card_index = queue.index(card)
    target_index = action.params[0]
    if not (0 <= target_index < len(queue)):
        raise ValueError("Chameleon target index out of range")

    target_card = queue[target_index]
    if target_card is card:
        raise ValueError("Chameleon must imitate another card")

    target_species = target_card.species
    handler = _HANDLERS.get(target_species, _noop_handler)

    fake_card = state.Card(owner=card.owner, species=target_species, entered_turn=card.entered_turn)
    temp_queue = tuple(fake_card if c is card else c for c in queue)
    temp_state = dc_replace(game_state, zones=dc_replace(game_state.zones, queue=temp_queue))

    copied_action = actions.Action(hand_index=action.hand_index, params=action.params[1:])
    temp_state = handler(temp_state, fake_card, copied_action)

    temp_state = _swap_card_reference(temp_state, fake_card, card)

    if target_species == "parrot" and len(action.params) >= 2:
        queue_after = temp_state.zones.queue
        try:
            upgraded_index = queue_after.index(card)
        except ValueError:  # pragma: no cover - defensive safeguard
            return temp_state

        temp_state, _ = state.remove_queue_card(temp_state, upgraded_index)
        insert_index = min(action.params[1], len(temp_state.zones.queue))
        temp_state = state.insert_queue(temp_state, insert_index, card)
    return temp_state


def _resolve_skunk(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    queue = list(game_state.zones.queue)
    strengths = sorted({c.strength for c in queue if c.species != "skunk"}, reverse=True)
    if not strengths:
        return game_state

    top_strengths = strengths[:2]
    remaining: List[state.Card] = []
    for c in queue:
        if c.species == "skunk":
            remaining.append(c)
        elif c.strength in top_strengths:
            game_state = state.push_to_zone(game_state, rules.ZONE_THATS_IT, c)
        else:
            remaining.append(c)

    skunks = [c for c in remaining if c.species == "skunk"]
    others = [c for c in remaining if c.species != "skunk"]
    new_queue = skunks + others
    return state.replace_queue(game_state, new_queue)


def _noop_handler(game_state: state.State, card: state.Card, action: actions.Action) -> state.State:
    return game_state


def _swap_card_reference(game_state: state.State, old: state.Card, new: state.Card) -> state.State:
    def _swap_tuple(cards: Tuple[state.Card, ...]) -> Tuple[state.Card, ...]:
        return tuple(new if c is old else c for c in cards)

    zones = game_state.zones
    swapped_zones = dc_replace(
        zones,
        queue=_swap_tuple(zones.queue),
        beasty_bar=_swap_tuple(zones.beasty_bar),
        thats_it=_swap_tuple(zones.thats_it),
    )
    return dc_replace(game_state, zones=swapped_zones)


def _recurring_hippo(game_state: state.State, index: int) -> Tuple[state.State, int, str]:
    queue = game_state.zones.queue
    hippo = queue[index]
    target = index
    blocker: Optional[state.Card] = None
    while target > 0:
        ahead = queue[target - 1]
        if ahead.species == "zebra" or ahead.strength >= hippo.strength:
            blocker = ahead
            break
        target -= 1
    if target == index:
        if index == 0:
            message = f"{_card_label(hippo)} already leads the line."
        elif blocker is not None:
            message = f"{_card_label(hippo)} is blocked by {_card_label(blocker)}."
        else:
            message = f"{_card_label(hippo)} holds position."
        return game_state, index + 1, message

    passed = queue[target:index]
    labels = ", ".join(_card_label(card) for card in passed)
    if labels:
        message = f"{_card_label(hippo)} charges past {labels}."
    else:
        message = f"{_card_label(hippo)} steps forward."

    game_state, _ = state.remove_queue_card(game_state, index)
    game_state = state.insert_queue(game_state, target, hippo)
    return game_state, target + 1, message


def _recurring_crocodile(game_state: state.State, index: int) -> Tuple[state.State, int, str]:
    queue = game_state.zones.queue
    crocodile = queue[index]
    scan = index - 1
    eaten: List[state.Card] = []
    blocker: Optional[state.Card] = None
    blockers = {"zebra"}
    while scan >= 0:
        queue = game_state.zones.queue
        ahead = queue[scan]
        if ahead.species in blockers or ahead.strength >= crocodile.strength:
            blocker = ahead
            break
        game_state, removed = state.remove_queue_card(game_state, scan)
        game_state = state.push_to_zone(game_state, rules.ZONE_THATS_IT, removed)
        eaten.append(removed)
        index -= 1
        scan -= 1
    if eaten:
        meal = ", ".join(_card_label(card) for card in eaten)
        message = f"{_card_label(crocodile)} eats {meal}."
        if blocker is not None:
            message += f" Stops before {_card_label(blocker)}."
    else:
        if blocker is not None:
            message = f"{_card_label(crocodile)} is blocked by {_card_label(blocker)}."
        else:
            message = f"{_card_label(crocodile)} already leads the line."
    return game_state, index + 1, message


def _recurring_giraffe(game_state: state.State, index: int) -> Tuple[state.State, int, str]:
    queue = game_state.zones.queue
    giraffe = queue[index]
    if giraffe.entered_turn == game_state.turn:
        message = f"{_card_label(giraffe)} waits until next turn."
        return game_state, index + 1, message
    if index == 0:
        message = f"{_card_label(giraffe)} already leads the line."
        return game_state, index + 1, message
    ahead = queue[index - 1]
    if ahead.strength >= giraffe.strength:
        message = f"{_card_label(giraffe)} is blocked by {_card_label(ahead)}."
        return game_state, index + 1, message

    game_state, _ = state.remove_queue_card(game_state, index)
    game_state = state.insert_queue(game_state, index - 1, giraffe)
    message = f"{_card_label(giraffe)} threads ahead of {_card_label(ahead)}."
    return game_state, index, message


_HANDLERS: Dict[str, Handler] = {
    "lion": _resolve_lion,
    "snake": _resolve_snake,
    "giraffe": _resolve_giraffe,
    "kangaroo": _resolve_kangaroo,
    "monkey": _resolve_monkey,
    "parrot": _resolve_parrot,
    "seal": _resolve_seal,
    "chameleon": _resolve_chameleon,
    "skunk": _resolve_skunk,
}


__all__ = [
    "resolve_play",
    "process_recurring",
    "process_recurring_with_trace",
]
