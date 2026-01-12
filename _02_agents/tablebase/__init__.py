"""Endgame tablebase for Beasty Bar.

Provides perfect play in positions with few remaining cards through
retrograde analysis from terminal positions.
"""

from __future__ import annotations

from .endgame import (
    EndgameTablebase,
    GameTheoreticValue,
    IncrementalTablebaseBuilder,
    TablebaseAgent,
    TablebaseConfig,
    TablebaseEntry,
    TablebaseGenerator,
    generate_tablebase,
    load_tablebase,
)

__all__ = [
    "EndgameTablebase",
    "GameTheoreticValue",
    "IncrementalTablebaseBuilder",
    "TablebaseAgent",
    "TablebaseConfig",
    "TablebaseEntry",
    "TablebaseGenerator",
    "generate_tablebase",
    "load_tablebase",
]
