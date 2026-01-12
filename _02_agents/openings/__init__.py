"""Opening book module for Beasty Bar.

This module provides pre-computed optimal openings using high-simulation MCTS,
enabling fast lookup at inference time for early game positions.

Usage:
    >>> from _02_agents.openings import OpeningBook, OpeningBookGenerator
    >>>
    >>> # Generate a new book
    >>> generator = OpeningBookGenerator(network, num_simulations=1600)
    >>> book = OpeningBook()
    >>> generator.generate(book, num_seeds=10000, depth=3)
    >>> book.save("opening_book.json")
    >>>
    >>> # Use at inference time
    >>> book = OpeningBook.load("opening_book.json")
    >>> agent = OpeningBookAgent(book, fallback_agent)
"""

from .book import (
    BookStats,
    OpeningBook,
    OpeningBookAgent,
    OpeningBookGenerator,
    OpeningEntry,
    generate_opening_book_cli,
)

__all__ = [
    "BookStats",
    "OpeningBook",
    "OpeningBookAgent",
    "OpeningBookGenerator",
    "OpeningEntry",
    "generate_opening_book_cli",
]
