"""Endgame tablebase for Beasty Bar.

Provides perfect play in positions with few remaining cards through
retrograde analysis from terminal positions.

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

from .storage import (
    MMapTablebase,
    MMapTablebaseConfig,
    SharedArrayTablebase,
    StoredValue,
    pack_entry,
    unpack_entry,
)

from .retrograde import (
    CanonicalAction,
    RetrogradeConfig,
    RetrogradeStats,
    RetrogradeTablebase,
    generate_retrograde_tablebase,
)

from .parallel import (
    ParallelConfig,
    ParallelStats,
    ParallelTablebaseGenerator,
    generate_parallel_tablebase,
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
