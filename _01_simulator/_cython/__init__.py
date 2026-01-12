"""Cython-accelerated game simulation module.

This module provides GIL-releasing implementations of the core game simulator
for multi-threaded execution. Falls back to pure Python if Cython is unavailable.

Usage:
    from _01_simulator._cython import (
        GameStateArray,
        step_batch_parallel,
        encode_observations_parallel,
        get_legal_masks_parallel,
        is_cython_available,
    )
"""

from __future__ import annotations

# Try to import Cython implementation, fall back to Python
try:
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

    _CYTHON_AVAILABLE = True
except ImportError:
    _CYTHON_AVAILABLE = False

    # Provide stubs that raise informative errors
    def _unavailable(*args: object, **kwargs: object) -> object:
        raise RuntimeError(
            "Cython extension not available. Build with: "
            "./scripts/build_cython.sh"
        )

    step_batch_parallel = _unavailable
    encode_observations_parallel = _unavailable
    get_legal_masks_parallel = _unavailable
    is_terminal_batch = _unavailable
    python_state_to_c = _unavailable
    encode_single_observation = _unavailable
    get_single_legal_mask = _unavailable
    get_single_scores = _unavailable
    step_single = _unavailable

    # Constants still available for reference
    SPECIES_CHAMELEON = 0
    SPECIES_CROCODILE = 1
    SPECIES_GIRAFFE = 2
    SPECIES_HIPPO = 3
    SPECIES_KANGAROO = 4
    SPECIES_LION = 5
    SPECIES_MONKEY = 6
    SPECIES_PARROT = 7
    SPECIES_SEAL = 8
    SPECIES_SKUNK = 9
    SPECIES_SNAKE = 10
    SPECIES_UNKNOWN = 11
    SPECIES_ZEBRA = 12

    GameStateArray = None  # type: ignore[assignment, misc]
    ACTION_DIM = 124
    OBSERVATION_DIM = 988


def is_cython_available() -> bool:
    """Check if Cython extension is available."""
    return _CYTHON_AVAILABLE


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
