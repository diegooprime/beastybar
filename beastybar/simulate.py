"""Batch simulation entry point."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from . import engine, state


@dataclass
class SimulationConfig:
    seed: int
    games: int = 1
    agent_a: Optional[str] = None
    agent_b: Optional[str] = None


def run(config: SimulationConfig) -> Iterable[state.State]:
    """Yield final states for the configured simulations."""

    raise NotImplementedError
