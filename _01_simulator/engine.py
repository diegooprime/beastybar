"""Rule engine entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import actions, cards, formatting, rules, state

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class TurnStep:
    """Describes a single phase of turn resolution."""

    name: str
    events: tuple[str, ...]


def legal_actions(game_state: state.State, player: int) -> Iterable[actions.Action]:
    """Return the legal play options for a player in the given state."""

    _validate_player_index(player)
    if player != game_state.active_player:
        return

    player_state = game_state.players[player]
    queue = game_state.zones.queue
    for idx, card in enumerate(player_state.hand):
        species = card.species
        if species == "kangaroo":
            max_hop = min(2, len(queue))
            if max_hop == 0:
                yield actions.Action(hand_index=idx)
            else:
                for hop in range(1, max_hop + 1):
                    yield actions.Action(hand_index=idx, params=(hop,))
        elif species == "parrot":
            for target in range(len(queue)):
                yield actions.Action(hand_index=idx, params=(target,))
        elif species == "chameleon":
            if not queue:
                continue
            for target in range(len(queue)):
                target_card = queue[target]
                if target_card is card or target_card.species == "chameleon":
                    continue
                # Allow additional params for the copied species to consume.
                for extra in _chameleon_params(target_card, len(queue)):
                    yield actions.Action(hand_index=idx, params=(target, *extra))
        else:
            yield actions.Action(hand_index=idx)


def step(game_state: state.State, action: actions.Action) -> state.State:
    """Advance the game by applying an action for the active player."""
    game_state, _ = step_with_trace(game_state, action)
    return game_state


def step_with_trace(game_state: state.State, action: actions.Action) -> tuple[state.State, tuple[TurnStep, ...]]:
    """Advance the game and capture the canonical five-phase resolution trace."""

    if is_terminal(game_state):
        raise ValueError("Cannot step; game already finished")

    player = game_state.active_player
    _validate_action(game_state, player, action)

    steps: list[TurnStep] = []

    game_state, card = state.remove_hand_card(game_state, player, action.hand_index)
    object.__setattr__(card, "entered_turn", game_state.turn)

    game_state = state.append_queue(game_state, card)
    steps.append(TurnStep(name="play", events=(f"Played {formatting.card_label(card)} to the queue.",)))

    before_resolve = game_state
    after_resolve = cards.resolve_play(game_state, card, action)
    resolve_events = _resolve_events(card, action, before_resolve, after_resolve)
    steps.append(TurnStep(name="resolve", events=resolve_events))
    game_state = after_resolve

    game_state, recurring_events = cards.process_recurring_with_trace(game_state)
    if not recurring_events:
        recurring_events = ("No recurring animals act.",)
    steps.append(TurnStep(name="recurring", events=recurring_events))

    game_state, check_events = _apply_five_card_check_with_trace(game_state)
    steps.append(TurnStep(name="five-animal check", events=check_events))

    game_state, _, draw_events = _draw_card_with_trace(game_state, player)
    steps.append(TurnStep(name="draw", events=draw_events))

    next_player = game_state.next_player()
    game_state = state.set_active_player(game_state, next_player, advance_turn=True)
    return game_state, tuple(steps)


def is_terminal(game_state: state.State) -> bool:
    """Return True if the game has concluded."""

    if game_state.turn == 0:
        return False

    remaining_cards = len(game_state.zones.queue)
    for player_state in game_state.players:
        remaining_cards += len(player_state.hand) + len(player_state.deck)
    if remaining_cards < rules.MAX_QUEUE_LENGTH:
        return True

    return all(not (player_state.hand or player_state.deck) for player_state in game_state.players)


def score(game_state: state.State) -> list[int]:
    """Return final scores for each player."""

    scores = [0 for _ in range(rules.PLAYER_COUNT)]
    for card in game_state.zones.beasty_bar:
        scores[card.owner] += card.points
    return scores


def _resolve_events(
    card: state.Card,
    action: actions.Action,
    before: state.State,
    after: state.State,
) -> tuple[str, ...]:
    unchanged = (
        before.zones.queue == after.zones.queue
        and before.zones.beasty_bar == after.zones.beasty_bar
        and before.zones.thats_it == after.zones.thats_it
    )

    species_messages = {
        "lion": "Lion roars to the front, scattering monkeys.",
        "snake": "Snake orders the queue by strength, keeping ties in place.",
        "giraffe": "Giraffe leans forward through weaker animals.",
        "kangaroo": "Kangaroo hops ahead based on the chosen distance.",
        "monkey": "Monkey pair drives out the heavyweights.",
        "parrot": "Parrot shoos a target to THAT'S IT.",
        "seal": "Seal flips Heaven's Gate and the bounce.",
        "chameleon": "Chameleon imitates the chosen ability.",
        "skunk": "Skunk expels the top strength bands.",
    }

    if card.species == "chameleon" and action.params:
        target_index = action.params[0]
        queue = before.zones.queue
        if 0 <= target_index < len(queue):
            target = queue[target_index]
            message = f"{formatting.card_label(card)} imitates {formatting.card_label(target)} on-play."
            if unchanged:
                return (message,)
            return (message,)

    if unchanged:
        return (f"{formatting.card_label(card)} has no immediate effect.",)

    message = species_messages.get(card.species, f"{formatting.card_label(card)} resolves its on-play ability.")
    return (message,)


def _draw_card_with_trace(
    game_state: state.State, player: int
) -> tuple[state.State, state.Card | None, tuple[str, ...]]:
    game_state, card = state.draw_card(game_state, player)
    if card is None:
        return game_state, None, ("Deck empty; no card drawn.",)
    return game_state, card, (f"Drew {formatting.card_label(card)}.",)


def _apply_five_card_check(game_state: state.State) -> state.State:
    game_state, _ = _apply_five_card_check_with_trace(game_state)
    return game_state


def _apply_five_card_check_with_trace(game_state: state.State) -> tuple[state.State, tuple[str, ...]]:
    queue = game_state.zones.queue
    if len(queue) != rules.MAX_QUEUE_LENGTH:
        return game_state, ("Queue below capacity; no animals move.",)

    entering = queue[:2]
    bounced = queue[-1]
    remaining = queue[2:-1]

    game_state = state.replace_queue(game_state, remaining)
    for card in entering:
        game_state = state.push_to_zone(game_state, rules.ZONE_BEASTY_BAR, card)
    game_state = state.push_to_zone(game_state, rules.ZONE_THATS_IT, bounced)

    messages: list[str] = []
    if entering:
        messages.append(f"Heaven's Gate admits {formatting.card_list(entering)}.")
    messages.append(f"THAT'S IT receives {formatting.card_label(bounced)}.")
    return game_state, tuple(messages)


def _validate_player_index(player: int) -> None:
    if not (0 <= player < rules.PLAYER_COUNT):
        raise ValueError("Player index out of range")


def _validate_action(game_state: state.State, player: int, action: actions.Action) -> None:
    player_state = game_state.players[player]
    if not (0 <= action.hand_index < len(player_state.hand)):
        raise ValueError("Hand index out of range for player")

    card = player_state.hand[action.hand_index]
    queue = game_state.zones.queue
    species = card.species

    if species == "kangaroo":
        max_hop = min(2, len(queue))
        if max_hop == 0:
            if action.params:
                raise ValueError("Kangaroo cannot hop without cards ahead")
            return
        if not action.params:
            return
        if len(action.params) != 1:
            raise ValueError("Kangaroo requires a hop distance parameter")
        hop = action.params[0]
        if not (1 <= hop <= max_hop):
            raise ValueError("Kangaroo hop distance out of range")
    elif species == "parrot":
        if len(action.params) != 1:
            raise ValueError("Parrot requires exactly one target parameter")
        target = action.params[0]
        if not (0 <= target < len(queue)):
            raise ValueError("Parrot target index out of range")
    elif species == "chameleon":
        if not action.params:
            raise ValueError("Chameleon requires target index to copy")
        target_index = action.params[0]
        if not (0 <= target_index < len(queue)):
            raise ValueError("Chameleon target index out of range")
        target_card = queue[target_index]
        if target_card is card:
            raise ValueError("Chameleon must copy another card")
        if target_card.species == "chameleon":
            raise ValueError("Chameleon cannot copy another chameleon")

        extra_params = action.params[1:]
        _validate_chameleon_params(target_card, extra_params, queue)
    else:
        if action.params:
            raise ValueError("This card does not accept parameters")


def _chameleon_params(target_card: state.Card, queue_len: int) -> Iterable[tuple[int, ...]]:
    species = target_card.species
    if species == "parrot":
        for target in range(queue_len):
            yield (target,)
        return
    if species == "kangaroo":
        max_hop = min(2, queue_len)
        if max_hop == 0:
            yield ()
        else:
            for hop in range(1, max_hop + 1):
                yield (hop,)
        return
    yield ()


def _validate_chameleon_params(target_card: state.Card, params: tuple[int, ...], queue: tuple[state.Card, ...]) -> None:
    species = target_card.species
    if species == "parrot":
        if len(params) != 1:
            raise ValueError("Chameleon-as-parrot requires one target parameter")
        if not (0 <= params[0] < len(queue)):
            raise ValueError("Chameleon-as-parrot target out of range")
    elif species == "kangaroo":
        if not params:
            return
        if len(params) != 1:
            raise ValueError("Chameleon-as-kangaroo expects a hop distance parameter")
        hop = params[0]
        max_hop = min(2, len(queue))
        if not (1 <= hop <= max_hop):
            raise ValueError("Chameleon-as-kangaroo hop distance out of range")
    else:
        if params:
            raise ValueError(f"Chameleon-as-{species} should not have extra parameters")


__all__ = [
    "TurnStep",
    "is_terminal",
    "legal_actions",
    "score",
    "step",
    "step_with_trace",
]
