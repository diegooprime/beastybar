"""Rule engine entry points."""
from __future__ import annotations

from typing import Iterable, List

from . import actions, cards, rules, state


def legal_actions(game_state: state.State, player: int) -> Iterable[actions.Action]:
    """Return the legal play options for a player in the given state."""

    _validate_player_index(player)
    if player != game_state.active_player:
        return []

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
                    yield actions.Action(hand_index=idx, params=(target,) + extra)
        else:
            yield actions.Action(hand_index=idx)


def step(game_state: state.State, action: actions.Action) -> state.State:
    """Advance the game by applying an action for the active player."""

    if is_terminal(game_state):
        raise ValueError("Cannot step; game already finished")

    player = game_state.active_player
    _validate_action(game_state, player, action)

    game_state, card = state.remove_hand_card(game_state, player, action.hand_index)
    game_state = state.append_queue(game_state, card)

    game_state = cards.resolve_play(game_state, card, action)
    game_state = cards.process_recurring(game_state)
    game_state = _apply_five_card_check(game_state)

    game_state, _ = state.draw_card(game_state, player)
    next_player = game_state.next_player()
    game_state = state.set_active_player(game_state, next_player, advance_turn=True)
    return game_state


def is_terminal(game_state: state.State) -> bool:
    """Return True if the game has concluded."""

    if game_state.turn == 0:
        return False

    remaining_cards = len(game_state.zones.queue)
    for player_state in game_state.players:
        remaining_cards += len(player_state.hand) + len(player_state.deck)
    if remaining_cards < rules.MAX_QUEUE_LENGTH:
        return True

    for player_state in game_state.players:
        if player_state.hand or player_state.deck:
            return False
    return True


def score(game_state: state.State) -> List[int]:
    """Return final scores for each player."""

    scores = [0 for _ in range(rules.PLAYER_COUNT)]
    for card in game_state.zones.beasty_bar:
        scores[card.owner] += card.points
    return scores


def _apply_five_card_check(game_state: state.State) -> state.State:
    queue = game_state.zones.queue
    if len(queue) != rules.MAX_QUEUE_LENGTH:
        return game_state

    entering = queue[:2]
    bounced = queue[-1]
    remaining = queue[2:-1]

    game_state = state.replace_queue(game_state, remaining)
    for card in entering:
        game_state = state.push_to_zone(game_state, rules.ZONE_BEASTY_BAR, card)
    game_state = state.push_to_zone(game_state, rules.ZONE_THATS_IT, bounced)
    return game_state


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


def _chameleon_params(target_card: state.Card, queue_len: int):
    species = target_card.species
    if species == "parrot":
        for target in range(queue_len):
            yield (target,)
    else:
        yield ()


def _validate_chameleon_params(target_card: state.Card, params: tuple[int, ...], queue: tuple[state.Card, ...]) -> None:
    species = target_card.species
    if species == "parrot":
        if len(params) != 1:
            raise ValueError("Chameleon-as-parrot requires one target parameter")
        if not (0 <= params[0] < len(queue)):
            raise ValueError("Chameleon-as-parrot target out of range")
    else:
        if params:
            raise ValueError("Chameleon-as-%s should not have extra parameters" % species)


__all__ = [
    "legal_actions",
    "step",
    "is_terminal",
    "score",
]
