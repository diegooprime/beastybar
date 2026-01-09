"""Opponent pool for diverse self-play training.

Provides opponent diversity to prevent self-play collapse by mixing:
- Current network (60%): Standard self-play signal
- Past checkpoints (20%): Prevents catastrophic forgetting
- Random agent (10%): Baseline calibration
- Heuristic agent (10%): Quality baseline
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import torch
    from torch import nn
    from _02_agents.base import Agent

logger = logging.getLogger(__name__)


class OpponentType(Enum):
    """Types of opponents in the pool."""
    CURRENT = auto()
    CHECKPOINT = auto()
    RANDOM = auto()
    HEURISTIC = auto()


class CheckpointEntry(NamedTuple):
    """Network checkpoint stored in the pool."""
    state_dict: dict[str, Any]
    iteration: int
    win_rate: float | None = None


@dataclass
class SampledOpponent:
    """Result of sampling an opponent from the pool."""
    opponent_type: OpponentType
    agent: Agent | None = None
    network_state: dict[str, Any] | None = None
    iteration: int | None = None

    @property
    def name(self) -> str:
        """Descriptive name for logging."""
        match self.opponent_type:
            case OpponentType.CURRENT: return "current"
            case OpponentType.CHECKPOINT: return f"checkpoint_{self.iteration}"
            case OpponentType.RANDOM: return "random"
            case OpponentType.HEURISTIC: return "heuristic"


@dataclass
class OpponentConfig:
    """Configuration for opponent sampling distribution."""
    current_weight: float = 0.6
    checkpoint_weight: float = 0.2
    random_weight: float = 0.1
    heuristic_weight: float = 0.1
    max_checkpoints: int = 10

    def __post_init__(self) -> None:
        total = self.current_weight + self.checkpoint_weight + self.random_weight + self.heuristic_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Opponent weights must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpponentConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class OpponentPool:
    """Pool of diverse opponents for self-play training."""

    def __init__(self, config: OpponentConfig | None = None, seed: int | None = None) -> None:
        self.config = config or OpponentConfig()
        self._rng = random.Random(seed)
        self.checkpoints: list[CheckpointEntry] = []
        self._random_agent: Agent | None = None
        self._heuristic_agent: Agent | None = None
        self._sample_counts: dict[OpponentType, int] = {t: 0 for t in OpponentType}

        logger.info(
            f"OpponentPool: current={self.config.current_weight:.0%}, "
            f"checkpoint={self.config.checkpoint_weight:.0%}, "
            f"random={self.config.random_weight:.0%}, "
            f"heuristic={self.config.heuristic_weight:.0%}"
        )

    @property
    def random_agent(self) -> Agent:
        if self._random_agent is None:
            from _02_agents.random_agent import RandomAgent
            self._random_agent = RandomAgent()
        return self._random_agent

    @property
    def heuristic_agent(self) -> Agent:
        if self._heuristic_agent is None:
            from _02_agents.heuristic import HeuristicAgent
            self._heuristic_agent = HeuristicAgent()
        return self._heuristic_agent

    def add_checkpoint(self, state_dict: dict[str, Any], iteration: int, win_rate: float | None = None) -> None:
        """Add a network checkpoint to the pool."""
        self.checkpoints.append(CheckpointEntry(copy.deepcopy(state_dict), iteration, win_rate))
        if len(self.checkpoints) > self.config.max_checkpoints:
            removed = self.checkpoints.pop(0)
            logger.debug(f"Removed oldest checkpoint (iter {removed.iteration})")
        logger.info(f"Added checkpoint at iteration {iteration} (pool: {len(self.checkpoints)}/{self.config.max_checkpoints})")

    def sample_opponent(self) -> SampledOpponent:
        """Sample an opponent according to configured distribution."""
        weights = [
            self.config.current_weight + (self.config.checkpoint_weight if not self.checkpoints else 0),
            self.config.checkpoint_weight if self.checkpoints else 0,
            self.config.random_weight,
            self.config.heuristic_weight,
        ]
        opponent_type = self._rng.choices(list(OpponentType), weights=weights)[0]
        self._sample_counts[opponent_type] += 1

        match opponent_type:
            case OpponentType.CURRENT:
                return SampledOpponent(OpponentType.CURRENT)
            case OpponentType.CHECKPOINT:
                cp = self._rng.choice(self.checkpoints)
                return SampledOpponent(OpponentType.CHECKPOINT, network_state=cp.state_dict, iteration=cp.iteration)
            case OpponentType.RANDOM:
                return SampledOpponent(OpponentType.RANDOM, agent=self.random_agent)
            case OpponentType.HEURISTIC:
                return SampledOpponent(OpponentType.HEURISTIC, agent=self.heuristic_agent)

    def get_statistics(self) -> dict[str, Any]:
        """Get sampling statistics."""
        total = sum(self._sample_counts.values())
        if total == 0:
            return {"total_samples": 0, "checkpoint_pool_size": len(self.checkpoints)}
        return {
            "total_samples": total,
            "checkpoint_pool_size": len(self.checkpoints),
            "sample_counts": {t.name.lower(): c for t, c in self._sample_counts.items()},
            "sample_percentages": {t.name.lower(): c / total for t, c in self._sample_counts.items()},
        }


def create_opponent_network(
    sampled: SampledOpponent,
    network_class: type[nn.Module],
    network_config: Any,
    device: torch.device,
) -> nn.Module | None:
    """Create network instance from a checkpoint opponent."""
    if sampled.opponent_type != OpponentType.CHECKPOINT or sampled.network_state is None:
        return None
    network = network_class(network_config)
    network.load_state_dict(sampled.network_state)
    network = network.to(device)
    network.eval()
    return network


__all__ = [
    "CheckpointEntry",
    "OpponentConfig",
    "OpponentPool",
    "OpponentType",
    "SampledOpponent",
    "create_opponent_network",
]
