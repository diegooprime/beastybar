"""Perfect information solver for Beasty Bar.

This module provides exact game-theoretic evaluation when opponent's hand is known.
"""

from .perfect_info import (
    PerfectInfoSolver,
    SearchResult,
    TranspositionTable,
    TTEntry,
    TTFlag,
    alpha_beta,
    analyze_position,
    evaluate_position,
    evaluate_terminal,
    generate_training_data,
    hash_state,
    is_position_solved,
    iterative_deepening,
    order_moves,
)

__all__ = [
    "PerfectInfoSolver",
    "SearchResult",
    "TTEntry",
    "TTFlag",
    "TranspositionTable",
    "alpha_beta",
    "analyze_position",
    "evaluate_position",
    "evaluate_terminal",
    "generate_training_data",
    "hash_state",
    "is_position_solved",
    "iterative_deepening",
    "order_moves",
]
