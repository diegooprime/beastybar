"""Opponent pool for diverse self-play training.

Provides opponent diversity to prevent self-play collapse by mixing:
- Current network (60%): Standard self-play signal
- Past checkpoints (20%): Prevents catastrophic forgetting
- Random agent (10%): Baseline calibration
- Heuristic agent (10%): Quality baseline
- MCTS agents (0%): Strong search-based opponents (optional)
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import torch
    from torch import nn

    from _02_agents.base import Agent
    from _02_agents.mcts.agent import MCTSAgent
    from _02_agents.neural.network import BeastyBarNetwork

logger = logging.getLogger(__name__)


class OpponentType(Enum):
    """Types of opponents in the pool."""
    CURRENT = auto()
    CHECKPOINT = auto()
    RANDOM = auto()
    HEURISTIC = auto()
    MCTS = auto()


class CheckpointEntry(NamedTuple):
    """Network checkpoint stored in the pool."""
    state_dict: dict[str, Any]
    iteration: int
    win_rate: float | None = None


@dataclass
class MCTSOpponentConfig:
    """Configuration for a single MCTS opponent."""
    c_puct: float = 1.5
    num_simulations: int = 200
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.3
    temperature: float = 1.0
    name: str = ""  # Auto-generated if empty

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"mcts_c{self.c_puct}_n{self.num_simulations}"


@dataclass
class SampledOpponent:
    """Result of sampling an opponent from the pool."""
    opponent_type: OpponentType
    agent: Agent | None = None
    network_state: dict[str, Any] | None = None
    iteration: int | None = None
    mcts_config_name: str | None = None

    @property
    def name(self) -> str:
        """Descriptive name for logging."""
        match self.opponent_type:
            case OpponentType.CURRENT:
                return "current"
            case OpponentType.CHECKPOINT:
                return f"checkpoint_{self.iteration}"
            case OpponentType.RANDOM:
                return "random"
            case OpponentType.HEURISTIC:
                return "heuristic"
            case OpponentType.MCTS:
                return self.mcts_config_name or "mcts"


@dataclass
class OpponentConfig:
    """Configuration for opponent sampling distribution."""
    current_weight: float = 0.6
    checkpoint_weight: float = 0.2
    random_weight: float = 0.1
    heuristic_weight: float = 0.1
    mcts_weight: float = 0.0  # Default 0 for backward compatibility
    max_checkpoints: int = 10
    mcts_configs: list[MCTSOpponentConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        total = (
            self.current_weight
            + self.checkpoint_weight
            + self.random_weight
            + self.heuristic_weight
            + self.mcts_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Opponent weights must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if f == "mcts_configs":
                result[f] = [
                    {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
                    for cfg in val
                ]
            else:
                result[f] = val
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpponentConfig:
        filtered = {}
        for k, v in data.items():
            if k not in cls.__dataclass_fields__:
                continue
            if k == "mcts_configs":
                filtered[k] = [MCTSOpponentConfig(**cfg) for cfg in v]
            else:
                filtered[k] = v
        return cls(**filtered)


class OpponentPool:
    """Pool of diverse opponents for self-play training."""

    def __init__(self, config: OpponentConfig | None = None, seed: int | None = None) -> None:
        self.config = config or OpponentConfig()
        self._rng = random.Random(seed)
        self.checkpoints: list[CheckpointEntry] = []
        self._random_agent: Agent | None = None
        self._heuristic_agent: Agent | None = None
        self._heuristic_variants: list[Agent] | None = None
        self._mcts_agents: dict[str, MCTSAgent] = {}
        self._mcts_network: BeastyBarNetwork | None = None
        self._sample_counts: dict[OpponentType, int] = dict.fromkeys(OpponentType, 0)

        log_parts = [
            f"current={self.config.current_weight:.0%}",
            f"checkpoint={self.config.checkpoint_weight:.0%}",
            f"random={self.config.random_weight:.0%}",
            f"heuristic={self.config.heuristic_weight:.0%}",
        ]
        if self.config.mcts_weight > 0:
            log_parts.append(f"mcts={self.config.mcts_weight:.0%}")
        logger.info(f"OpponentPool: {', '.join(log_parts)}")

    @property
    def random_agent(self) -> Agent:
        if self._random_agent is None:
            from _02_agents.random_agent import RandomAgent
            self._random_agent = RandomAgent()
        return self._random_agent

    @property
    def heuristic_agent(self) -> Agent:
        """Sample a random heuristic variant."""
        if self._heuristic_variants is None:
            from _02_agents.heuristic import create_heuristic_variants
            self._heuristic_variants = create_heuristic_variants()
        # Randomly sample from variants each time
        return self._rng.choice(self._heuristic_variants)

    @property
    def mcts_agents(self) -> dict[str, MCTSAgent]:
        """Lazy-load MCTS agents from configs on first access."""
        if not self._mcts_agents and self.config.mcts_configs:
            if self._mcts_network is None:
                raise RuntimeError(
                    "MCTS network not set. Call set_mcts_network() before sampling MCTS opponents."
                )
            from _02_agents.mcts.agent import MCTSAgent
            for cfg in self.config.mcts_configs:
                agent = MCTSAgent(
                    network=self._mcts_network,
                    num_simulations=cfg.num_simulations,
                    c_puct=cfg.c_puct,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    dirichlet_epsilon=cfg.dirichlet_epsilon,
                    temperature=cfg.temperature,
                    name=cfg.name,
                )
                self._mcts_agents[cfg.name] = agent
                logger.debug(f"Created MCTS agent: {cfg.name}")
        return self._mcts_agents

    def set_mcts_network(self, network: BeastyBarNetwork) -> None:
        """Set the network for MCTS evaluation.

        This must be called before sampling MCTS opponents.
        If MCTS agents were already created, they are recreated with the new network.

        Args:
            network: Policy-value network for MCTS evaluation
        """
        self._mcts_network = network
        # Clear existing agents so they get recreated with new network
        self._mcts_agents = {}
        logger.debug("MCTS network set for opponent pool")

    def add_checkpoint(self, state_dict: dict[str, Any], iteration: int, win_rate: float | None = None) -> None:
        """Add a network checkpoint to the pool."""
        self.checkpoints.append(CheckpointEntry(copy.deepcopy(state_dict), iteration, win_rate))
        if len(self.checkpoints) > self.config.max_checkpoints:
            removed = self.checkpoints.pop(0)
            logger.debug(f"Removed oldest checkpoint (iter {removed.iteration})")
        logger.info(f"Added checkpoint at iteration {iteration} (pool: {len(self.checkpoints)}/{self.config.max_checkpoints})")

    def sample_opponent(self) -> SampledOpponent:
        """Sample an opponent according to configured distribution."""
        # Redistribute checkpoint weight to current if no checkpoints available
        checkpoint_redistrib = self.config.checkpoint_weight if not self.checkpoints else 0
        # Redistribute MCTS weight to current if no MCTS configs available
        mcts_redistrib = self.config.mcts_weight if not self.config.mcts_configs else 0

        weights = [
            self.config.current_weight + checkpoint_redistrib + mcts_redistrib,
            self.config.checkpoint_weight if self.checkpoints else 0,
            self.config.random_weight,
            self.config.heuristic_weight,
            self.config.mcts_weight if self.config.mcts_configs else 0,
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
            case OpponentType.MCTS:
                agents = self.mcts_agents
                config_name = self._rng.choice(list(agents.keys()))
                return SampledOpponent(
                    OpponentType.MCTS,
                    agent=agents[config_name],
                    mcts_config_name=config_name,
                )

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


def create_default_mcts_configs() -> list[MCTSOpponentConfig]:
    """Create a diverse set of default MCTS opponent configurations.

    Returns 6 configs with varying characteristics:
    - Exploitation-heavy: Low c_puct for greedy search
    - Balanced: Standard MCTS parameters
    - Exploration-heavy: High c_puct for broad search
    - Fast/shallow: Fewer simulations for speed
    - Deep search: More simulations for strength
    - High noise: More Dirichlet noise for diversity
    """
    return [
        MCTSOpponentConfig(
            c_puct=0.5,
            num_simulations=200,
            name="mcts_exploit",
        ),
        MCTSOpponentConfig(
            c_puct=1.5,
            num_simulations=200,
            name="mcts_balanced",
        ),
        MCTSOpponentConfig(
            c_puct=3.0,
            num_simulations=200,
            name="mcts_explore",
        ),
        MCTSOpponentConfig(
            c_puct=1.5,
            num_simulations=50,
            name="mcts_fast",
        ),
        MCTSOpponentConfig(
            c_puct=1.5,
            num_simulations=500,
            name="mcts_deep",
        ),
        MCTSOpponentConfig(
            c_puct=1.5,
            num_simulations=200,
            dirichlet_epsilon=0.5,
            name="mcts_noisy",
        ),
    ]


class AdaptiveOpponentPool(OpponentPool):
    """Opponent pool with win-rate-based adaptive weighting.

    Adjusts opponent sampling weights based on evaluation win rates:
    opponents we lose to get higher weight, opponents we beat consistently
    get lower weight. This focuses training on weaknesses.
    """

    MIN_WEIGHT = 0.05  # Never drop below 5%

    def __init__(self, config: OpponentConfig | None = None, seed: int | None = None) -> None:
        super().__init__(config, seed)
        # Adaptive weights for each opponent type (normalized, sum to 1.0)
        # Initialize from config weights
        self._adaptive_weights: dict[OpponentType, float] = {
            OpponentType.CURRENT: self.config.current_weight,
            OpponentType.CHECKPOINT: self.config.checkpoint_weight,
            OpponentType.RANDOM: self.config.random_weight,
            OpponentType.HEURISTIC: self.config.heuristic_weight,
            OpponentType.MCTS: self.config.mcts_weight,
        }
        logger.info("AdaptiveOpponentPool initialized with win-rate-based weighting")

    def update_weights_from_win_rates(self, win_rates: dict[str, float]) -> None:
        """Recompute opponent weights based on recent evaluation win rates.

        The weighting formula inverts win rates: opponents we lose to get
        higher weight, opponents we beat consistently get lower weight.
        A minimum weight floor ensures no opponent is ever completely dropped.

        Args:
            win_rates: Map of opponent name to win rate (0.0 to 1.0).
                       Keys should match OpponentType names (lowercase) or
                       specific opponent names like "checkpoint_100".
        """
        if not win_rates:
            logger.warning("Empty win_rates dict, weights unchanged")
            return

        # Map opponent names to OpponentType
        name_to_type: dict[str, OpponentType] = {
            "current": OpponentType.CURRENT,
            "random": OpponentType.RANDOM,
            "heuristic": OpponentType.HEURISTIC,
        }
        # Checkpoint entries get aggregated
        # MCTS entries get aggregated

        # Calculate raw weights: invert win rates (lose more = higher weight)
        raw_weights: dict[OpponentType, float] = {}
        checkpoint_win_rates: list[float] = []
        mcts_win_rates: list[float] = []

        for opponent_name, win_rate in win_rates.items():
            opponent_lower = opponent_name.lower()

            if opponent_lower in name_to_type:
                opponent_type = name_to_type[opponent_lower]
                raw_weights[opponent_type] = max(1.0 - win_rate, self.MIN_WEIGHT)
            elif opponent_lower.startswith("checkpoint"):
                checkpoint_win_rates.append(win_rate)
            elif opponent_lower.startswith("mcts"):
                mcts_win_rates.append(win_rate)
            else:
                logger.debug(f"Unknown opponent name: {opponent_name}, skipping")

        # Aggregate checkpoint win rates (average)
        if checkpoint_win_rates:
            avg_checkpoint_wr = sum(checkpoint_win_rates) / len(checkpoint_win_rates)
            raw_weights[OpponentType.CHECKPOINT] = max(1.0 - avg_checkpoint_wr, self.MIN_WEIGHT)
        elif self.config.checkpoint_weight > 0:
            # Keep original weight if no checkpoint data
            raw_weights[OpponentType.CHECKPOINT] = self.config.checkpoint_weight

        # Aggregate MCTS win rates (average)
        if mcts_win_rates:
            avg_mcts_wr = sum(mcts_win_rates) / len(mcts_win_rates)
            raw_weights[OpponentType.MCTS] = max(1.0 - avg_mcts_wr, self.MIN_WEIGHT)
        elif self.config.mcts_weight > 0:
            # Keep original weight if no MCTS data
            raw_weights[OpponentType.MCTS] = self.config.mcts_weight

        # Fill in any missing opponent types with minimum weight
        for opponent_type in OpponentType:
            if opponent_type not in raw_weights:
                # Use config weight if available, otherwise minimum
                config_weight = getattr(
                    self.config,
                    f"{opponent_type.name.lower()}_weight",
                    self.MIN_WEIGHT
                )
                raw_weights[opponent_type] = max(config_weight, self.MIN_WEIGHT)

        # Normalize to sum to 1.0
        total = sum(raw_weights.values())
        if total > 0:
            normalized = {k: v / total for k, v in raw_weights.items()}
            self._adaptive_weights = normalized

            # Log the update
            weight_str = ", ".join(
                f"{t.name.lower()}={w:.1%}"
                for t, w in sorted(normalized.items(), key=lambda x: x[0].name)
            )
            logger.info(f"AdaptiveOpponentPool weights updated: {weight_str}")
        else:
            logger.warning("Total raw weights is 0, keeping previous weights")

    def sample_opponent(self) -> SampledOpponent:
        """Sample an opponent according to adaptive distribution."""
        # Redistribute unavailable weights to current
        checkpoint_weight = self._adaptive_weights[OpponentType.CHECKPOINT]
        mcts_weight = self._adaptive_weights[OpponentType.MCTS]

        checkpoint_redistrib = checkpoint_weight if not self.checkpoints else 0
        mcts_redistrib = mcts_weight if not self.config.mcts_configs else 0

        weights = [
            self._adaptive_weights[OpponentType.CURRENT] + checkpoint_redistrib + mcts_redistrib,
            checkpoint_weight if self.checkpoints else 0,
            self._adaptive_weights[OpponentType.RANDOM],
            self._adaptive_weights[OpponentType.HEURISTIC],
            mcts_weight if self.config.mcts_configs else 0,
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
            case OpponentType.MCTS:
                agents = self.mcts_agents
                config_name = self._rng.choice(list(agents.keys()))
                return SampledOpponent(
                    OpponentType.MCTS,
                    agent=agents[config_name],
                    mcts_config_name=config_name,
                )

    def get_adaptive_weights(self) -> dict[str, float]:
        """Get current adaptive weights for logging/debugging."""
        return {t.name.lower(): w for t, w in self._adaptive_weights.items()}

    def get_statistics(self) -> dict[str, Any]:
        """Get sampling statistics including adaptive weights."""
        stats = super().get_statistics()
        stats["adaptive_weights"] = self.get_adaptive_weights()
        return stats


__all__ = [
    "AdaptiveOpponentPool",
    "CheckpointEntry",
    "MCTSOpponentConfig",
    "OpponentConfig",
    "OpponentPool",
    "OpponentType",
    "SampledOpponent",
    "create_default_mcts_configs",
    "create_opponent_network",
]
