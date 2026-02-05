"""Cython-accelerated game simulation module.

This module provides GIL-releasing implementations of the core game simulator
for multi-threaded execution. Cython extension is required.

Usage:
    from _01_simulator._cython import (
        GameStateArray,
        step_batch_parallel,
        encode_observations_parallel,
        get_legal_masks_parallel,
    )
"""

from __future__ import annotations

# Import Cython implementation directly - no fallback
from ._cython_core import (
    PY_ACTION_DIM as ACTION_DIM,
)
from ._cython_core import (
    PY_OBSERVATION_DIM as OBSERVATION_DIM,
)
from ._cython_core import (
    # Constants (have PY_ prefix in Cython module)
    PY_SPECIES_CHAMELEON as SPECIES_CHAMELEON,
)
from ._cython_core import (
    PY_SPECIES_CROCODILE as SPECIES_CROCODILE,
)
from ._cython_core import (
    PY_SPECIES_GIRAFFE as SPECIES_GIRAFFE,
)
from ._cython_core import (
    PY_SPECIES_HIPPO as SPECIES_HIPPO,
)
from ._cython_core import (
    PY_SPECIES_KANGAROO as SPECIES_KANGAROO,
)
from ._cython_core import (
    PY_SPECIES_LION as SPECIES_LION,
)
from ._cython_core import (
    PY_SPECIES_MONKEY as SPECIES_MONKEY,
)
from ._cython_core import (
    PY_SPECIES_PARROT as SPECIES_PARROT,
)
from ._cython_core import (
    PY_SPECIES_SEAL as SPECIES_SEAL,
)
from ._cython_core import (
    PY_SPECIES_SKUNK as SPECIES_SKUNK,
)
from ._cython_core import (
    PY_SPECIES_SNAKE as SPECIES_SNAKE,
)
from ._cython_core import (
    PY_SPECIES_UNKNOWN as SPECIES_UNKNOWN,
)
from ._cython_core import (
    PY_SPECIES_ZEBRA as SPECIES_ZEBRA,
)
from ._cython_core import (
    # Types
    GameStateArray,
    # Batch functions (main API)
    encode_observations_parallel,
    # Single-game functions (for testing/debugging)
    encode_single_observation,
    get_legal_masks_parallel,
    get_single_legal_mask,
    get_single_scores,
    is_terminal_batch,
    python_state_to_c,
    step_batch_parallel,
    step_single,
)


def is_cython_available() -> bool:
    """Check if Cython extension is available. Always returns True."""
    return True


__all__ = [
    # Constants
    "ACTION_DIM",
    "OBSERVATION_DIM",
    "SPECIES_CHAMELEON",
    "SPECIES_CROCODILE",
    "SPECIES_GIRAFFE",
    "SPECIES_HIPPO",
    "SPECIES_KANGAROO",
    "SPECIES_LION",
    "SPECIES_MONKEY",
    "SPECIES_PARROT",
    "SPECIES_SEAL",
    "SPECIES_SKUNK",
    "SPECIES_SNAKE",
    "SPECIES_UNKNOWN",
    "SPECIES_ZEBRA",
    # Types
    "GameStateArray",
    # Batch functions
    "encode_observations_parallel",
    # Single-game functions
    "encode_single_observation",
    "get_legal_masks_parallel",
    "get_single_legal_mask",
    "get_single_scores",
    # Availability check
    "is_cython_available",
    "is_terminal_batch",
    "python_state_to_c",
    "step_batch_parallel",
    "step_single",
]
