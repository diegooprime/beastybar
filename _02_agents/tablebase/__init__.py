"""Endgame tablebase for Beasty Bar.

Provides perfect play in positions with few remaining cards through
retrograde analysis from terminal positions.

Current: 1M positions (≤4 cards per player) in data/endgame_4card_final.tb

TODO: Compute more positions
- Generate 5-card tablebase (~10M positions)
- Generate 6-card tablebase (~100M positions)
- See docs/TABLEBASE_AWS_PLAN.md for AWS generation plan

Modules:
- endgame: Forward minimax search with alpha-beta pruning
- enumerate: Systematic position enumeration
- storage: Memory-mapped storage for large tablebases
- retrograde: Backward analysis from terminal positions
- parallel: Multi-process generation for high-core-count machines
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
from .enumerate import (
    CanonicalKey,
    EnumerationConfig,
    PositionEnumerator,
    PositionIndexer,
    count_positions_estimate,
)
from .parallel import (
    ParallelConfig,
    ParallelStats,
    ParallelTablebaseGenerator,
    generate_parallel_tablebase,
)
from .retrograde import (
    CanonicalAction,
    RetrogradeConfig,
    RetrogradeStats,
    RetrogradeTablebase,
    generate_retrograde_tablebase,
)
from .storage import (
    MMapTablebase,
    MMapTablebaseConfig,
    SharedArrayTablebase,
    StoredValue,
    pack_entry,
    unpack_entry,
)

__all__ = [
    # endgame
    "EndgameTablebase",
    "GameTheoreticValue",
    "IncrementalTablebaseBuilder",
    "TablebaseAgent",
    "TablebaseConfig",
    "TablebaseEntry",
    "TablebaseGenerator",
    "generate_tablebase",
    "load_tablebase",
    # enumerate
    "CanonicalKey",
    "EnumerationConfig",
    "PositionEnumerator",
    "PositionIndexer",
    "count_positions_estimate",
    # storage
    "MMapTablebase",
    "MMapTablebaseConfig",
    "SharedArrayTablebase",
    "StoredValue",
    "pack_entry",
    "unpack_entry",
    # retrograde
    "CanonicalAction",
    "RetrogradeConfig",
    "RetrogradeStats",
    "RetrogradeTablebase",
    "generate_retrograde_tablebase",
    # parallel
    "ParallelConfig",
    "ParallelStats",
    "ParallelTablebaseGenerator",
    "generate_parallel_tablebase",
]
