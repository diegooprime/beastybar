"""Beasty Bar package exposing the main domain areas."""

# Absolute imports allow this module to be imported directly when the repository
# root is on PYTHONPATH (e.g., during pytest collection) without requiring
# package-relative context.
from _01_simulator import *  # noqa: F403
from _04_ui import *  # noqa: F403
from _05_other import *  # noqa: F403

__all__ = [
    "_01_simulator",
    "_04_ui",
    "_05_other",
]
