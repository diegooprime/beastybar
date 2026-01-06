"""Action definitions for the Beasty Bar engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """Represents a player's decision to play a hand card.

    `hand_index` identifies the card in the current player's hand. Optional
    `params` can encode additional choices (e.g., parrot target indices). The
    structure is intentionally lightweight so that species-specific handlers can
    interpret the payload as needed.
    """

    hand_index: int
    params: tuple[int, ...] = ()


__all__ = ["Action"]
