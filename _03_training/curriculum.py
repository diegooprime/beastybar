"""Curriculum learning for Beasty Bar neural network training.

This module provides:
- Species subset games for gradual complexity increase
- Opponent curriculum for progressive difficulty
- Curriculum scheduling with advancement criteria
- State persistence for training resumption

The curriculum approach allows the neural network to learn basic gameplay
patterns before encountering the full complexity of all species and
sophisticated opponents.

Example:
    from _03_training.curriculum import (
        CurriculumScheduler,
        CurriculumConfig,
        create_curriculum_game,
        sample_opponent,
    )

    # Create curriculum scheduler
    config = CurriculumConfig(advance_threshold=0.70)
    scheduler = CurriculumScheduler(config)

    # Create game with current curriculum level species
    state = create_curriculum_game(
        seed=42,
        species_whitelist=scheduler.get_species_whitelist(),
    )

    # Sample opponent based on current stage
    opponent = sample_opponent(
        mix=OPPONENT_STAGES[scheduler.opponent_stage],
        network=current_network,
    )
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from _01_simulator import rules
from _01_simulator.state import Card, PlayerState, State, Zones

if TYPE_CHECKING:
    import torch.nn as nn

    from _02_agents.base import Agent

logger = logging.getLogger(__name__)


# ============================================================================
# Species Curriculum Levels
# ============================================================================

CURRICULUM_LEVELS: dict[int, list[str]] = {
    1: ["lion", "hippo", "crocodile", "snake"],  # Simple abilities
    2: ["lion", "hippo", "crocodile", "snake", "giraffe", "zebra", "seal"],  # +blocking/movement
    3: [
        "lion",
        "hippo",
        "crocodile",
        "snake",
        "giraffe",
        "zebra",
        "seal",
        "chameleon",
        "monkey",
    ],  # +conditional
    4: [
        "lion",
        "hippo",
        "crocodile",
        "snake",
        "giraffe",
        "zebra",
        "seal",
        "chameleon",
        "monkey",
        "kangaroo",
        "parrot",
        "skunk",
    ],  # Full game
}

# Minimum species count required for a valid game
MIN_SPECIES_FOR_GAME = 4


def validate_species_whitelist(species_whitelist: list[str]) -> None:
    """Validate that a species whitelist is valid for gameplay.

    Args:
        species_whitelist: List of species names to validate.

    Raises:
        ValueError: If whitelist is invalid (too few species or unknown species).
    """
    if len(species_whitelist) < MIN_SPECIES_FOR_GAME:
        raise ValueError(
            f"Species whitelist must contain at least {MIN_SPECIES_FOR_GAME} species, "
            f"got {len(species_whitelist)}"
        )

    for species in species_whitelist:
        if species not in rules.SPECIES:
            raise ValueError(f"Unknown species: {species}")
        if species == "unknown":
            raise ValueError("Cannot include 'unknown' species in whitelist")


# ============================================================================
# Species-Restricted Game Initialization
# ============================================================================


def create_curriculum_game(
    seed: int,
    species_whitelist: list[str],
    starting_player: int = 0,
) -> State:
    """Create a game state with restricted species set.

    Generates a new game where each player's deck contains only
    species from the whitelist. This enables curriculum learning
    by starting with simpler species combinations.

    Args:
        seed: Random seed for deck shuffling and game initialization.
        species_whitelist: List of species to include in both players' decks.
            Must contain at least MIN_SPECIES_FOR_GAME species.
        starting_player: Index of the player who moves first (0 or 1).

    Returns:
        Initial game state with restricted species decks.

    Raises:
        ValueError: If species_whitelist is invalid or starting_player out of range.

    Example:
        >>> state = create_curriculum_game(
        ...     seed=42,
        ...     species_whitelist=["lion", "hippo", "crocodile", "snake"],
        ... )
        >>> # Each player now has only these 4 species in their deck/hand
    """
    validate_species_whitelist(species_whitelist)

    if not (0 <= starting_player < rules.PLAYER_COUNT):
        raise ValueError(f"Invalid starting player index: {starting_player}")

    rng = random.Random(seed)
    players: list[PlayerState] = []

    for owner in range(rules.PLAYER_COUNT):
        # Create cards only for whitelisted species
        cards = [Card(owner=owner, species=species) for species in species_whitelist]
        rng.shuffle(cards)

        # Split into hand and deck
        hand_size = min(rules.HAND_SIZE, len(cards))
        hand = tuple(cards[:hand_size])
        deck = tuple(cards[hand_size:])

        players.append(PlayerState(deck=deck, hand=hand))

    return State(
        seed=seed,
        turn=0,
        active_player=starting_player,
        players=tuple(players),
        zones=Zones(),
    )


def get_curriculum_level_for_species_count(count: int) -> int:
    """Determine appropriate curriculum level based on desired species count.

    Args:
        count: Desired number of species.

    Returns:
        Curriculum level that provides at least that many species.
    """
    for level in sorted(CURRICULUM_LEVELS.keys()):
        if len(CURRICULUM_LEVELS[level]) >= count:
            return level
    return max(CURRICULUM_LEVELS.keys())


# ============================================================================
# Curriculum Scheduler
# ============================================================================


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning scheduler.

    Attributes:
        initial_level: Starting curriculum level (1-4 for species).
        advance_threshold: Win rate required to advance to next level.
        min_games_per_level: Minimum games required before advancement check.
        max_games_per_level: Maximum games at a level before forced advancement.
        initial_opponent_stage: Starting opponent difficulty stage (1-5).
        opponent_advance_threshold: Win rate to advance opponent stage.
        opponent_min_games: Minimum games before opponent stage advancement.
    """

    initial_level: int = 1
    advance_threshold: float = 0.70
    min_games_per_level: int = 500
    max_games_per_level: int = 5000
    initial_opponent_stage: int = 1
    opponent_advance_threshold: float = 0.65
    opponent_min_games: int = 300

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.initial_level < 1 or self.initial_level > len(CURRICULUM_LEVELS):
            raise ValueError(
                f"initial_level must be between 1 and {len(CURRICULUM_LEVELS)}, "
                f"got {self.initial_level}"
            )
        if not 0.0 < self.advance_threshold <= 1.0:
            raise ValueError(
                f"advance_threshold must be in (0, 1], got {self.advance_threshold}"
            )
        if self.min_games_per_level <= 0:
            raise ValueError(
                f"min_games_per_level must be positive, got {self.min_games_per_level}"
            )
        if self.max_games_per_level < self.min_games_per_level:
            raise ValueError(
                f"max_games_per_level ({self.max_games_per_level}) must be >= "
                f"min_games_per_level ({self.min_games_per_level})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumConfig:
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CurriculumState:
    """Internal state tracking for curriculum scheduler.

    Attributes:
        current_level: Current species curriculum level (1-4).
        games_at_level: Number of games played at current level.
        wins_at_level: Number of wins at current level.
        opponent_stage: Current opponent difficulty stage (1-5).
        games_at_stage: Number of games at current opponent stage.
        wins_at_stage: Number of wins at current opponent stage.
        level_history: List of (level, games, win_rate) for completed levels.
        stage_history: List of (stage, games, win_rate) for completed stages.
    """

    current_level: int = 1
    games_at_level: int = 0
    wins_at_level: int = 0
    opponent_stage: int = 1
    games_at_stage: int = 0
    wins_at_stage: int = 0
    level_history: list[tuple[int, int, float]] = field(default_factory=list)
    stage_history: list[tuple[int, int, float]] = field(default_factory=list)


class CurriculumScheduler:
    """Scheduler for curriculum learning progression.

    Manages both species curriculum (which species are available) and
    opponent curriculum (how difficult opponents are). Tracks performance
    and decides when to advance to harder settings.

    Attributes:
        config: Curriculum configuration parameters.

    Example:
        >>> config = CurriculumConfig(advance_threshold=0.70)
        >>> scheduler = CurriculumScheduler(config)
        >>>
        >>> # Record game results
        >>> scheduler.record_game(win=True)
        >>>
        >>> # Check if should advance
        >>> if scheduler.should_advance(scheduler.win_rate, scheduler.games_at_level):
        ...     scheduler.advance()
    """

    def __init__(self, config: CurriculumConfig | None = None) -> None:
        """Initialize curriculum scheduler.

        Args:
            config: Configuration for curriculum. Uses defaults if None.
        """
        self._config = config or CurriculumConfig()
        self._state = CurriculumState(
            current_level=self._config.initial_level,
            opponent_stage=self._config.initial_opponent_stage,
        )

    @property
    def config(self) -> CurriculumConfig:
        """Return curriculum configuration."""
        return self._config

    @property
    def current_level(self) -> int:
        """Current species curriculum level (1-4)."""
        return self._state.current_level

    @property
    def max_level(self) -> int:
        """Maximum species curriculum level."""
        return len(CURRICULUM_LEVELS)

    @property
    def games_at_level(self) -> int:
        """Number of games played at current level."""
        return self._state.games_at_level

    @property
    def wins_at_level(self) -> int:
        """Number of wins at current level."""
        return self._state.wins_at_level

    @property
    def win_rate(self) -> float:
        """Current win rate at this level (0.0-1.0)."""
        if self._state.games_at_level == 0:
            return 0.0
        return self._state.wins_at_level / self._state.games_at_level

    @property
    def opponent_stage(self) -> int:
        """Current opponent difficulty stage (1-5)."""
        return self._state.opponent_stage

    @property
    def max_opponent_stage(self) -> int:
        """Maximum opponent difficulty stage."""
        return len(OPPONENT_STAGES)

    @property
    def games_at_stage(self) -> int:
        """Number of games at current opponent stage."""
        return self._state.games_at_stage

    @property
    def stage_win_rate(self) -> float:
        """Current win rate at this opponent stage."""
        if self._state.games_at_stage == 0:
            return 0.0
        return self._state.wins_at_stage / self._state.games_at_stage

    @property
    def level_history(self) -> list[tuple[int, int, float]]:
        """History of completed levels: (level, games, win_rate)."""
        return self._state.level_history.copy()

    @property
    def stage_history(self) -> list[tuple[int, int, float]]:
        """History of completed stages: (stage, games, win_rate)."""
        return self._state.stage_history.copy()

    def get_species_whitelist(self) -> list[str]:
        """Get species whitelist for current curriculum level.

        Returns:
            List of species names available at current level.
        """
        return CURRICULUM_LEVELS[self._state.current_level].copy()

    def should_advance(self, win_rate: float, games_at_level: int) -> bool:
        """Determine if ready to advance to next species level.

        Advancement occurs when:
        1. Minimum games have been played at this level AND
        2. Win rate exceeds threshold OR max games reached

        Args:
            win_rate: Current win rate (0.0-1.0).
            games_at_level: Number of games played at current level.

        Returns:
            True if should advance to next level.
        """
        # Already at max level
        if self._state.current_level >= self.max_level:
            return False

        # Not enough games yet
        if games_at_level < self._config.min_games_per_level:
            return False

        # Force advance if at max games
        if games_at_level >= self._config.max_games_per_level:
            logger.info(
                f"Force advancing from level {self._state.current_level} "
                f"after {games_at_level} games (max reached)"
            )
            return True

        # Advance if win rate exceeds threshold
        return win_rate >= self._config.advance_threshold

    def should_advance_opponent(self) -> bool:
        """Determine if ready to advance opponent difficulty.

        Returns:
            True if should advance to next opponent stage.
        """
        # Already at max stage
        if self._state.opponent_stage >= self.max_opponent_stage:
            return False

        # Not enough games
        if self._state.games_at_stage < self._config.opponent_min_games:
            return False

        # Advance if win rate exceeds threshold
        return self.stage_win_rate >= self._config.opponent_advance_threshold

    def advance(self) -> bool:
        """Advance to next species curriculum level.

        Records current level statistics to history and resets counters.

        Returns:
            True if advanced, False if already at max level.
        """
        if self._state.current_level >= self.max_level:
            logger.warning(f"Already at max level {self.max_level}, cannot advance")
            return False

        # Record current level stats
        self._state.level_history.append(
            (self._state.current_level, self._state.games_at_level, self.win_rate)
        )

        old_level = self._state.current_level
        self._state.current_level += 1
        self._state.games_at_level = 0
        self._state.wins_at_level = 0

        logger.info(
            f"Advanced species curriculum: level {old_level} -> {self._state.current_level} "
            f"(now using {len(self.get_species_whitelist())} species)"
        )

        return True

    def advance_opponent(self) -> bool:
        """Advance to next opponent difficulty stage.

        Returns:
            True if advanced, False if already at max stage.
        """
        if self._state.opponent_stage >= self.max_opponent_stage:
            logger.warning(f"Already at max opponent stage {self.max_opponent_stage}")
            return False

        # Record current stage stats
        self._state.stage_history.append(
            (self._state.opponent_stage, self._state.games_at_stage, self.stage_win_rate)
        )

        old_stage = self._state.opponent_stage
        self._state.opponent_stage += 1
        self._state.games_at_stage = 0
        self._state.wins_at_stage = 0

        logger.info(f"Advanced opponent stage: {old_stage} -> {self._state.opponent_stage}")

        return True

    def record_game(self, win: bool) -> None:
        """Record a completed game result.

        Updates counters for both species level and opponent stage.
        Automatically checks and performs advancement if criteria met.

        Args:
            win: Whether the training agent won the game.
        """
        # Update level counters
        self._state.games_at_level += 1
        if win:
            self._state.wins_at_level += 1

        # Update stage counters
        self._state.games_at_stage += 1
        if win:
            self._state.wins_at_stage += 1

        # Check for automatic advancement
        if self.should_advance(self.win_rate, self.games_at_level):
            self.advance()

        if self.should_advance_opponent():
            self.advance_opponent()

    def reset_level_stats(self) -> None:
        """Reset statistics for current level (without advancing)."""
        self._state.games_at_level = 0
        self._state.wins_at_level = 0

    def reset_stage_stats(self) -> None:
        """Reset statistics for current opponent stage."""
        self._state.games_at_stage = 0
        self._state.wins_at_stage = 0

    def set_level(self, level: int) -> None:
        """Manually set the curriculum level.

        Args:
            level: Target level (1-4).

        Raises:
            ValueError: If level is out of range.
        """
        if level < 1 or level > self.max_level:
            raise ValueError(f"Level must be between 1 and {self.max_level}, got {level}")

        self._state.current_level = level
        self.reset_level_stats()
        logger.info(f"Manually set curriculum level to {level}")

    def set_opponent_stage(self, stage: int) -> None:
        """Manually set the opponent stage.

        Args:
            stage: Target stage (1-5).

        Raises:
            ValueError: If stage is out of range.
        """
        if stage < 1 or stage > self.max_opponent_stage:
            raise ValueError(
                f"Stage must be between 1 and {self.max_opponent_stage}, got {stage}"
            )

        self._state.opponent_stage = stage
        self.reset_stage_stats()
        logger.info(f"Manually set opponent stage to {stage}")

    def get_state_dict(self) -> dict[str, Any]:
        """Get scheduler state as dictionary for serialization.

        Returns:
            Dictionary containing all scheduler state.
        """
        return {
            "config": self._config.to_dict(),
            "state": {
                "current_level": self._state.current_level,
                "games_at_level": self._state.games_at_level,
                "wins_at_level": self._state.wins_at_level,
                "opponent_stage": self._state.opponent_stage,
                "games_at_stage": self._state.games_at_stage,
                "wins_at_stage": self._state.wins_at_stage,
                "level_history": self._state.level_history,
                "stage_history": self._state.stage_history,
            },
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> CurriculumScheduler:
        """Create scheduler from state dictionary.

        Args:
            data: Dictionary from get_state_dict().

        Returns:
            CurriculumScheduler with restored state.
        """
        config = CurriculumConfig.from_dict(data["config"])
        scheduler = cls(config)

        state_data = data["state"]
        scheduler._state = CurriculumState(
            current_level=state_data["current_level"],
            games_at_level=state_data["games_at_level"],
            wins_at_level=state_data["wins_at_level"],
            opponent_stage=state_data["opponent_stage"],
            games_at_stage=state_data["games_at_stage"],
            wins_at_stage=state_data["wins_at_stage"],
            level_history=[tuple(x) for x in state_data["level_history"]],
            stage_history=[tuple(x) for x in state_data["stage_history"]],
        )

        return scheduler

    def summary(self) -> str:
        """Get human-readable summary of curriculum state.

        Returns:
            Multi-line summary string.
        """
        species = self.get_species_whitelist()
        lines = [
            "Curriculum Status:",
            f"  Species Level: {self.current_level}/{self.max_level} "
            f"({len(species)} species: {', '.join(species[:3])}...)",
            f"  Level Progress: {self.games_at_level} games, "
            f"{self.win_rate:.1%} win rate "
            f"(threshold: {self._config.advance_threshold:.0%})",
            f"  Opponent Stage: {self.opponent_stage}/{self.max_opponent_stage}",
            f"  Stage Progress: {self.games_at_stage} games, "
            f"{self.stage_win_rate:.1%} win rate",
        ]
        return "\n".join(lines)


# ============================================================================
# Curriculum State Persistence
# ============================================================================


def save_curriculum_state(scheduler: CurriculumScheduler, path: str) -> None:
    """Save curriculum scheduler state to file.

    Args:
        scheduler: Scheduler to save.
        path: Path to save file (JSON format).
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = scheduler.get_state_dict()

    with open(save_path, "w") as f:
        json.dump(state_dict, f, indent=2)

    logger.info(f"Saved curriculum state to {save_path}")


def load_curriculum_state(path: str) -> CurriculumScheduler:
    """Load curriculum scheduler state from file.

    Args:
        path: Path to saved state file.

    Returns:
        CurriculumScheduler with restored state.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    load_path = Path(path)

    if not load_path.exists():
        raise FileNotFoundError(f"Curriculum state file not found: {load_path}")

    with open(load_path) as f:
        state_dict = json.load(f)

    scheduler = CurriculumScheduler.from_state_dict(state_dict)

    logger.info(
        f"Loaded curriculum state from {load_path}: "
        f"level={scheduler.current_level}, stage={scheduler.opponent_stage}"
    )

    return scheduler


# ============================================================================
# Opponent Curriculum
# ============================================================================


@dataclass(frozen=True)
class OpponentMix:
    """Configuration for mixing different opponent types.

    All ratios should sum to 1.0 for proper probability distribution.

    Attributes:
        self_play_ratio: Probability of self-play opponent.
        random_ratio: Probability of random agent opponent.
        heuristic_ratio: Probability of heuristic agent opponent.
        mcts_ratio: Probability of MCTS agent opponent.
        historical_ratio: Probability of historical checkpoint opponent.
    """

    self_play_ratio: float = 1.0
    random_ratio: float = 0.0
    heuristic_ratio: float = 0.0
    mcts_ratio: float = 0.0
    historical_ratio: float = 0.0

    def __post_init__(self) -> None:
        """Validate that ratios are valid probabilities."""
        ratios = [
            self.self_play_ratio,
            self.random_ratio,
            self.heuristic_ratio,
            self.mcts_ratio,
            self.historical_ratio,
        ]

        for ratio in ratios:
            if ratio < 0 or ratio > 1:
                raise ValueError(f"All ratios must be in [0, 1], got {ratio}")

        total = sum(ratios)
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "self_play_ratio": self.self_play_ratio,
            "random_ratio": self.random_ratio,
            "heuristic_ratio": self.heuristic_ratio,
            "mcts_ratio": self.mcts_ratio,
            "historical_ratio": self.historical_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> OpponentMix:
        """Create from dictionary."""
        return cls(**data)


# Predefined opponent stages for progressive difficulty
OPPONENT_STAGES: dict[int, OpponentMix] = {
    1: OpponentMix(self_play_ratio=1.0),  # Pure self-play
    2: OpponentMix(self_play_ratio=0.7, random_ratio=0.3),
    3: OpponentMix(self_play_ratio=0.6, heuristic_ratio=0.4),
    4: OpponentMix(self_play_ratio=0.5, heuristic_ratio=0.3, mcts_ratio=0.2),
    5: OpponentMix(self_play_ratio=0.4, historical_ratio=0.6),  # League
}


def sample_opponent(
    mix: OpponentMix,
    network: nn.Module,
    historical_checkpoints: list[str] | None = None,
    mcts_iterations: int = 500,
    device: str | None = None,
) -> Agent:
    """Sample an opponent according to the mix ratios.

    Randomly selects an opponent type based on the probability distribution
    defined by the OpponentMix, then creates and returns the corresponding
    agent instance.

    Args:
        mix: Opponent type probability distribution.
        network: Current neural network for self-play and neural opponents.
        historical_checkpoints: List of checkpoint paths for historical opponents.
            Required if historical_ratio > 0.
        mcts_iterations: Number of iterations for MCTS opponent.
        device: Device for neural network inference.

    Returns:
        Agent instance of the sampled type.

    Raises:
        ValueError: If historical_ratio > 0 but no checkpoints provided.
    """
    from _02_agents.heuristic import HeuristicAgent
    from _02_agents.mcts import MCTSAgent
    from _02_agents.neural.agent import NeuralAgent
    from _02_agents.random_agent import RandomAgent

    # Build cumulative distribution
    thresholds = [
        mix.self_play_ratio,
        mix.self_play_ratio + mix.random_ratio,
        mix.self_play_ratio + mix.random_ratio + mix.heuristic_ratio,
        mix.self_play_ratio + mix.random_ratio + mix.heuristic_ratio + mix.mcts_ratio,
        1.0,  # historical
    ]

    sample = np.random.random()

    if sample < thresholds[0]:
        # Self-play: wrap current network in NeuralAgent
        import torch

        device_torch = torch.device(device) if device else None
        return NeuralAgent(network, device=device_torch, mode="stochastic")

    elif sample < thresholds[1]:
        # Random agent
        return RandomAgent()

    elif sample < thresholds[2]:
        # Heuristic agent
        return HeuristicAgent()

    elif sample < thresholds[3]:
        # MCTS agent
        return MCTSAgent(iterations=mcts_iterations)

    else:
        # Historical checkpoint
        if not historical_checkpoints:
            # Fall back to self-play if no checkpoints available
            logger.warning(
                "Historical opponent requested but no checkpoints available. "
                "Falling back to self-play."
            )
            import torch

            device_torch = torch.device(device) if device else None
            return NeuralAgent(network, device=device_torch, mode="stochastic")

        # Sample from historical checkpoints
        from _02_agents.neural.agent import load_neural_agent

        checkpoint_path = np.random.choice(historical_checkpoints)
        return load_neural_agent(checkpoint_path, mode="stochastic")


def get_opponent_type_name(mix: OpponentMix) -> str:
    """Get a descriptive name for the opponent mix.

    Args:
        mix: Opponent mix configuration.

    Returns:
        Human-readable description of the mix.
    """
    parts = []
    if mix.self_play_ratio > 0:
        parts.append(f"self-play:{mix.self_play_ratio:.0%}")
    if mix.random_ratio > 0:
        parts.append(f"random:{mix.random_ratio:.0%}")
    if mix.heuristic_ratio > 0:
        parts.append(f"heuristic:{mix.heuristic_ratio:.0%}")
    if mix.mcts_ratio > 0:
        parts.append(f"mcts:{mix.mcts_ratio:.0%}")
    if mix.historical_ratio > 0:
        parts.append(f"historical:{mix.historical_ratio:.0%}")

    return " + ".join(parts)


# ============================================================================
# Historical Checkpoint Management
# ============================================================================


@dataclass
class CheckpointInfo:
    """Metadata for a historical checkpoint.

    Attributes:
        path: Path to checkpoint file.
        iteration: Training iteration when checkpoint was saved.
        win_rate: Win rate achieved by this checkpoint.
        timestamp: Unix timestamp when added to pool.
    """

    path: str
    iteration: int
    win_rate: float
    timestamp: float = 0.0


class HistoricalPool:
    """Manages a pool of historical checkpoints for league training.

    Maintains a fixed-size pool of past model checkpoints that can be
    sampled as opponents during training. Implements prioritized sampling
    based on checkpoint quality (win rate).

    Attributes:
        max_size: Maximum number of checkpoints to retain.
    """

    def __init__(self, max_size: int = 10) -> None:
        """Initialize historical pool.

        Args:
            max_size: Maximum number of checkpoints to keep.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self._max_size = max_size
        self._checkpoints: list[CheckpointInfo] = []

    @property
    def max_size(self) -> int:
        """Maximum pool size."""
        return self._max_size

    def __len__(self) -> int:
        """Number of checkpoints in pool."""
        return len(self._checkpoints)

    def add_checkpoint(
        self,
        path: str,
        iteration: int,
        win_rate: float,
    ) -> None:
        """Add a checkpoint to the pool.

        If pool is full, removes the lowest win-rate checkpoint.

        Args:
            path: Path to checkpoint file.
            iteration: Training iteration number.
            win_rate: Win rate achieved by this checkpoint.
        """
        import time

        info = CheckpointInfo(
            path=path,
            iteration=iteration,
            win_rate=win_rate,
            timestamp=time.time(),
        )

        # Check if we already have this checkpoint
        for existing in self._checkpoints:
            if existing.path == path:
                logger.debug(f"Checkpoint {path} already in pool, updating")
                self._checkpoints.remove(existing)
                break

        self._checkpoints.append(info)

        # Evict if over capacity (remove lowest win rate)
        while len(self._checkpoints) > self._max_size:
            # Sort by win rate and remove lowest
            self._checkpoints.sort(key=lambda x: x.win_rate)
            removed = self._checkpoints.pop(0)
            logger.debug(
                f"Evicted checkpoint {removed.path} with win_rate={removed.win_rate:.2%}"
            )

        logger.info(
            f"Added checkpoint to pool: {path} (iter={iteration}, "
            f"win_rate={win_rate:.2%}), pool size={len(self._checkpoints)}"
        )

    def sample_checkpoint(self, strategy: str = "uniform") -> str | None:
        """Sample a checkpoint from the pool.

        Args:
            strategy: Sampling strategy:
                - "uniform": Equal probability for all checkpoints.
                - "weighted": Higher win rate = higher probability.
                - "recent": More recent checkpoints preferred.

        Returns:
            Path to sampled checkpoint, or None if pool is empty.
        """
        if not self._checkpoints:
            return None

        if strategy == "uniform":
            return np.random.choice([c.path for c in self._checkpoints])

        elif strategy == "weighted":
            # Weight by win rate (with floor to avoid zero weights)
            win_rates = np.array([max(c.win_rate, 0.1) for c in self._checkpoints])
            probs = win_rates / win_rates.sum()
            idx = np.random.choice(len(self._checkpoints), p=probs)
            return self._checkpoints[idx].path

        elif strategy == "recent":
            # Weight by recency (more recent = higher weight)
            timestamps = np.array([c.timestamp for c in self._checkpoints])
            # Normalize to [0, 1] and add small constant
            if timestamps.max() > timestamps.min():
                weights = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
            else:
                weights = np.ones_like(timestamps)
            weights = weights + 0.1  # Avoid zero weights
            probs = weights / weights.sum()
            idx = np.random.choice(len(self._checkpoints), p=probs)
            return self._checkpoints[idx].path

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def get_checkpoints(self) -> list[str]:
        """Get all checkpoint paths in the pool.

        Returns:
            List of checkpoint paths, sorted by iteration (oldest first).
        """
        sorted_checkpoints = sorted(self._checkpoints, key=lambda x: x.iteration)
        return [c.path for c in sorted_checkpoints]

    def get_checkpoint_info(self) -> list[CheckpointInfo]:
        """Get detailed info for all checkpoints.

        Returns:
            List of CheckpointInfo objects.
        """
        return self._checkpoints.copy()

    def clear(self) -> None:
        """Remove all checkpoints from pool."""
        self._checkpoints.clear()
        logger.info("Cleared historical checkpoint pool")

    def get_state_dict(self) -> dict[str, Any]:
        """Get pool state for serialization.

        Returns:
            Dictionary with pool configuration and checkpoints.
        """
        return {
            "max_size": self._max_size,
            "checkpoints": [
                {
                    "path": c.path,
                    "iteration": c.iteration,
                    "win_rate": c.win_rate,
                    "timestamp": c.timestamp,
                }
                for c in self._checkpoints
            ],
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> HistoricalPool:
        """Restore pool from state dictionary.

        Args:
            data: Dictionary from get_state_dict().

        Returns:
            HistoricalPool with restored state.
        """
        pool = cls(max_size=data["max_size"])
        for c_data in data["checkpoints"]:
            pool._checkpoints.append(
                CheckpointInfo(
                    path=c_data["path"],
                    iteration=c_data["iteration"],
                    win_rate=c_data["win_rate"],
                    timestamp=c_data.get("timestamp", 0.0),
                )
            )
        return pool


def save_historical_pool(pool: HistoricalPool, path: str) -> None:
    """Save historical pool state to file.

    Args:
        pool: Pool to save.
        path: Path to save file (JSON format).
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(pool.get_state_dict(), f, indent=2)

    logger.info(f"Saved historical pool ({len(pool)} checkpoints) to {save_path}")


def load_historical_pool(path: str) -> HistoricalPool:
    """Load historical pool from file.

    Args:
        path: Path to saved pool file.

    Returns:
        HistoricalPool with restored state.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    load_path = Path(path)

    if not load_path.exists():
        raise FileNotFoundError(f"Historical pool file not found: {load_path}")

    with open(load_path) as f:
        data = json.load(f)

    pool = HistoricalPool.from_state_dict(data)
    logger.info(f"Loaded historical pool ({len(pool)} checkpoints) from {load_path}")

    return pool


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "CURRICULUM_LEVELS",
    "MIN_SPECIES_FOR_GAME",
    "OPPONENT_STAGES",
    "CheckpointInfo",
    "CurriculumConfig",
    "CurriculumScheduler",
    "CurriculumState",
    "HistoricalPool",
    "OpponentMix",
    "create_curriculum_game",
    "get_curriculum_level_for_species_count",
    "get_opponent_type_name",
    "load_curriculum_state",
    "load_historical_pool",
    "sample_opponent",
    "save_curriculum_state",
    "save_historical_pool",
    "validate_species_whitelist",
]
