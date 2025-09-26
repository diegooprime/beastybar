"""Immutable game state representations and helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

from . import rules

CardTuple = Tuple["Card", ...]


@dataclass(frozen=True)
class Card:
    """Represents a single animal card owned by a player."""

    owner: int
    species: str

    def __post_init__(self) -> None:
        if self.species not in rules.SPECIES:
            raise ValueError(f"Unknown species: {self.species}")
        if not (0 <= self.owner < rules.PLAYER_COUNT):
            raise ValueError("Owner index out of range")

    @property
    def strength(self) -> int:
        return rules.SPECIES[self.species].strength

    @property
    def points(self) -> int:
        return rules.SPECIES[self.species].points


@dataclass(frozen=True)
class PlayerState:
    """Immutable view of a player's resources."""

    deck: CardTuple
    hand: CardTuple


@dataclass(frozen=True)
class Zones:
    """Shared zones on the table."""

    queue: CardTuple = ()
    beasty_bar: CardTuple = ()
    bounced: CardTuple = ()
    thats_it: CardTuple = ()


@dataclass(frozen=True)
class State:
    """Full immutable snapshot of the game."""

    seed: int
    turn: int
    active_player: int
    players: Tuple[PlayerState, ...]
    zones: Zones

    def __post_init__(self) -> None:
        if len(self.players) != rules.PLAYER_COUNT:
            raise ValueError("State must track two players")
        if not (0 <= self.active_player < rules.PLAYER_COUNT):
            raise ValueError("Active player index out of range")

    def next_player(self) -> int:
        return (self.active_player + 1) % rules.PLAYER_COUNT


def initial_state(seed: int, starting_player: int = 0) -> State:
    """Create a freshly shuffled game state."""

    if not (0 <= starting_player < rules.PLAYER_COUNT):
        raise ValueError("Invalid starting player index")

    rng = random.Random(seed)
    players: list[PlayerState] = []
    for owner in range(rules.PLAYER_COUNT):
        cards = [Card(owner=owner, species=species) for species in rules.BASE_DECK]
        rng.shuffle(cards)
        hand = tuple(cards[: rules.HAND_SIZE])
        deck = tuple(cards[rules.HAND_SIZE :])
        players.append(PlayerState(deck=deck, hand=hand))

    return State(
        seed=seed,
        turn=0,
        active_player=starting_player,
        players=tuple(players),
        zones=Zones(),
    )


def draw_card(game_state: State, player: int) -> Tuple[State, Optional[Card]]:
    """Draw the top card of a player's deck into their hand."""

    player_state = game_state.players[player]
    if not player_state.deck:
        return game_state, None

    card = player_state.deck[0]
    new_deck = player_state.deck[1:]
    new_hand = player_state.hand + (card,)
    updated_player = replace(player_state, deck=new_deck, hand=new_hand)
    return _replace_player(game_state, player, updated_player), card


def remove_hand_card(game_state: State, player: int, index: int) -> Tuple[State, Card]:
    """Remove the indexed card from a player's hand."""

    player_state = game_state.players[player]
    if not (0 <= index < len(player_state.hand)):
        raise IndexError("Hand index out of range")

    card = player_state.hand[index]
    new_hand = player_state.hand[:index] + player_state.hand[index + 1 :]
    updated_player = replace(player_state, hand=new_hand)
    return _replace_player(game_state, player, updated_player), card


def append_queue(game_state: State, card: Card) -> State:
    """Add a card to the back of the queue."""

    queue = game_state.zones.queue
    if len(queue) >= rules.MAX_QUEUE_LENGTH:
        raise ValueError("Queue is at maximum capacity")
    new_queue = queue + (card,)
    return _replace_zones(game_state, queue=new_queue)


def insert_queue(game_state: State, index: int, card: Card) -> State:
    """Insert a card into the queue at a specific position."""

    queue = game_state.zones.queue
    if len(queue) >= rules.MAX_QUEUE_LENGTH:
        raise ValueError("Queue is at maximum capacity")
    if index < 0:
        index += len(queue) + 1
    if not (0 <= index <= len(queue)):
        raise IndexError("Queue insertion index out of range")

    new_queue = queue[:index] + (card,) + queue[index:]
    return _replace_zones(game_state, queue=new_queue)


def remove_queue_card(game_state: State, index: int) -> Tuple[State, Card]:
    """Remove and return a card from the queue."""

    queue = game_state.zones.queue
    if not (0 <= index < len(queue)):
        raise IndexError("Queue index out of range")

    card = queue[index]
    new_queue = queue[:index] + queue[index + 1 :]
    return _replace_zones(game_state, queue=new_queue), card


def push_to_zone(game_state: State, zone: str, card: Card) -> State:
    """Send a card to one of the named zones."""

    if zone not in _ZONE_NAMES:
        raise ValueError(f"Unknown zone: {zone}")
    current = getattr(game_state.zones, zone)
    new_zone = current + (card,)
    return _replace_zones(game_state, **{zone: new_zone})


def replace_queue(game_state: State, new_queue: Sequence[Card]) -> State:
    """Replace the full queue with a new ordered sequence."""

    if len(new_queue) > rules.MAX_QUEUE_LENGTH:
        raise ValueError("Queue length exceeds maximum")
    return _replace_zones(game_state, queue=tuple(new_queue))


def set_active_player(game_state: State, player: int, *, advance_turn: bool = False) -> State:
    """Return a state with the active player changed."""

    if not (0 <= player < rules.PLAYER_COUNT):
        raise ValueError("Active player index out of range")
    turn = game_state.turn + 1 if advance_turn else game_state.turn
    return replace(game_state, active_player=player, turn=turn)


_ZONE_NAMES = {"queue", "beasty_bar", "bounced", "thats_it"}


def _replace_player(game_state: State, index: int, new_player_state: PlayerState) -> State:
    players = list(game_state.players)
    players[index] = new_player_state
    return replace(game_state, players=tuple(players))


def _replace_zones(game_state: State, **updates) -> State:
    zones = replace(game_state.zones, **updates)
    return replace(game_state, zones=zones)


__all__ = [
    "Card",
    "CardTuple",
    "PlayerState",
    "Zones",
    "State",
    "initial_state",
    "draw_card",
    "remove_hand_card",
    "append_queue",
    "insert_queue",
    "remove_queue_card",
    "replace_queue",
    "push_to_zone",
    "set_active_player",
]
