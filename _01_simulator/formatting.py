"""Shared formatting utilities for game state display."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from . import state


def card_label(card: state.Card) -> str:
    """
    Return a human-readable label for a card.

    Args:
        card: The card to format

    Returns:
        A string like "Kangaroo (P0)" for the kangaroo card owned by player 0
    """
    species = card.species.replace("_", " ").title()
    return f"{species} (P{card.owner})"


def card_list(cards: Iterable[state.Card]) -> str:
    """
    Format a collection of cards as a comma-separated list.

    Args:
        cards: An iterable of cards to format

    Returns:
        A string like "Lion (P0), Monkey (P1), Zebra (P0)"
    """
    return ", ".join(card_label(card) for card in cards)


__all__ = ["card_label", "card_list"]
