"""Core rule constants and species metadata for the Beasty Bar simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Zone identifiers used across the engine and UI layers.
ZONE_QUEUE = "queue"
ZONE_BEASTY_BAR = "beasty_bar"
ZONE_THATS_IT = "thats_it"

MAX_QUEUE_LENGTH = 5
HAND_SIZE = 4
DECK_SIZE = 12
PLAYER_COUNT = 2


@dataclass(frozen=True)
class Species:
    """Defines static properties for an animal species."""

    name: str
    strength: int
    points: int
    recurring: bool = False
    permanent: bool = False


SPECIES: Dict[str, Species] = {
    "lion": Species("lion", strength=12, points=2),
    "hippo": Species("hippo", strength=11, points=2, recurring=True),
    "crocodile": Species("crocodile", strength=10, points=3, recurring=True),
    "snake": Species("snake", strength=9, points=2),
    "giraffe": Species("giraffe", strength=8, points=3, recurring=True),
    "zebra": Species("zebra", strength=7, points=4, permanent=True),
    "seal": Species("seal", strength=6, points=2),
    "chameleon": Species("chameleon", strength=5, points=3),
    "monkey": Species("monkey", strength=4, points=3),
    "kangaroo": Species("kangaroo", strength=3, points=4),
    "parrot": Species("parrot", strength=2, points=4),
    "skunk": Species("skunk", strength=1, points=4),
}

BASE_DECK: Tuple[str, ...] = tuple(SPECIES.keys())

if len(BASE_DECK) != DECK_SIZE:
    raise ValueError("Base deck definition must contain exactly 12 cards")

__all__ = [
    "Species",
    "SPECIES",
    "BASE_DECK",
    "ZONE_QUEUE",
    "ZONE_BEASTY_BAR",
    "ZONE_THATS_IT",
    "MAX_QUEUE_LENGTH",
    "HAND_SIZE",
    "DECK_SIZE",
    "PLAYER_COUNT",
]
