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
    return [actions.Action(hand_index=i) for i in range(len(player_state.hand))]


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


__all__ = [
    "legal_actions",
    "step",
    "is_terminal",
    "score",
]
