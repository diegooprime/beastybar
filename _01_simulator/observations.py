"""Observation encoding utilities for the Beasty Bar simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from . import rules, state

CardEncoding = Tuple[int, int, int, int, int]

# Card encoding order: present flag, owner, species id, strength, points.

_SPECIES_INDEX = {
    species: index for index, species in enumerate(sorted(rules.SPECIES.keys()))
}
_INDEX_TO_SPECIES = tuple(sorted(rules.SPECIES.keys()))
_UNKNOWN_ID = _SPECIES_INDEX["unknown"]
_TOTAL_CARD_SLOTS = rules.DECK_SIZE * rules.PLAYER_COUNT

_PAD_CARD: CardEncoding = (0, -1, _UNKNOWN_ID, 0, 0)


@dataclass(frozen=True)
class Observation:
    """Fixed-size encoding of a game state from one player's perspective."""

    queue: Tuple[CardEncoding, ...]
    beasty_bar: Tuple[CardEncoding, ...]
    thats_it: Tuple[CardEncoding, ...]
    hand: Tuple[CardEncoding, ...]
    deck_counts: Tuple[int, ...]
    hand_counts: Tuple[int, ...]
    active_player: int
    perspective: int
    turn: int

    def as_dict(self) -> dict[str, object]:
        """Return the observation as a dict for JSON/np serialization."""

        return {
            "queue": self.queue,
            "beasty_bar": self.beasty_bar,
            "thats_it": self.thats_it,
            "hand": self.hand,
            "deck_counts": self.deck_counts,
            "hand_counts": self.hand_counts,
            "active_player": self.active_player,
            "perspective": self.perspective,
            "turn": self.turn,
        }


def build_observation(
    game_state: state.State,
    perspective: int,
    *,
    mask_hidden: bool = True,
) -> Observation:
    """Encode public zones and the perspective hand into fixed-size tensors.

    Args:
        game_state: Source immutable state snapshot.
        perspective: Player index the observation is built for.
        mask_hidden: When true, opponent hands/decks are masked via
            :func:`state.mask_state_for_player` to prevent leaking hidden cards.

    Returns:
        Observation dataclass with card feature tuples and turn context.
    """

    if not (0 <= perspective < rules.PLAYER_COUNT):
        raise ValueError("Perspective player index out of range")

    visible_state = (
        state.mask_state_for_player(game_state, perspective)
        if mask_hidden
        else game_state
    )

    queue = _encode_zone(visible_state.zones.queue, rules.MAX_QUEUE_LENGTH)
    beasty_bar = _encode_zone(visible_state.zones.beasty_bar, _TOTAL_CARD_SLOTS)
    thats_it = _encode_zone(visible_state.zones.thats_it, _TOTAL_CARD_SLOTS)
    hand = _encode_zone(visible_state.players[perspective].hand, rules.HAND_SIZE)

    deck_counts = tuple(len(player.deck) for player in visible_state.players)
    hand_counts = tuple(len(player.hand) for player in visible_state.players)

    return Observation(
        queue=queue,
        beasty_bar=beasty_bar,
        thats_it=thats_it,
        hand=hand,
        deck_counts=deck_counts,
        hand_counts=hand_counts,
        active_player=visible_state.active_player,
        perspective=perspective,
        turn=visible_state.turn,
    )


def species_index(species: str) -> int:
    """Return the stable integer index for a species name."""

    try:
        return _SPECIES_INDEX[species]
    except KeyError as exc:
        raise ValueError(f"Unknown species: {species}") from exc


def species_name(index: int) -> str:
    """Reverse lookup the species string for an encoded id."""

    if not (0 <= index < len(_INDEX_TO_SPECIES)):
        raise ValueError("Species index out of range")
    return _INDEX_TO_SPECIES[index]


def _encode_zone(cards: Sequence[state.Card], max_len: int) -> Tuple[CardEncoding, ...]:
    if max_len <= 0:
        raise ValueError("Zone length must be positive")

    encoded: list[CardEncoding] = []
    for card in cards:
        encoded.append(
            (
                1,
                card.owner,
                species_index(card.species),
                card.strength,
                card.points,
            )
        )

    if len(encoded) > max_len:
        raise ValueError("Zone contains more cards than the maximum length")

    encoded.extend(_PAD_CARD for _ in range(max_len - len(encoded)))
    return tuple(encoded)


__all__ = [
    "CardEncoding",
    "Observation",
    "build_observation",
    "species_index",
    "species_name",
]
