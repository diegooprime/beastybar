"""Per-opponent win rate tracking and adaptive sampling weights.

Tracks game results against individual opponents and computes sampling weights
based on learning signal - opponents with ~50% win rate provide maximum learning.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class OpponentStats:
    """Statistics for a single opponent."""

    opponent_id: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_updated_iteration: int = 0

    @property
    def games(self) -> int:
        """Total number of games played against this opponent."""
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Return win rate, default 0.5 if no games."""
        if self.games == 0:
            return 0.5
        return self.wins / self.games

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "opponent_id": self.opponent_id,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "last_updated_iteration": self.last_updated_iteration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpponentStats:
        """Deserialize from dictionary."""
        return cls(
            opponent_id=data["opponent_id"],
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            draws=data.get("draws", 0),
            last_updated_iteration=data.get("last_updated_iteration", 0),
        )


@dataclass
class _WindowedStats:
    """Internal stats with sliding window support."""

    opponent_id: str
    results: deque[Literal["win", "loss", "draw"]] = field(default_factory=deque)
    last_updated_iteration: int = 0

    def add_result(self, result: Literal["win", "loss", "draw"], max_size: int) -> None:
        """Add a result, enforcing window size."""
        self.results.append(result)
        while len(self.results) > max_size:
            self.results.popleft()

    def to_opponent_stats(self) -> OpponentStats:
        """Convert to OpponentStats."""
        wins = sum(1 for r in self.results if r == "win")
        losses = sum(1 for r in self.results if r == "loss")
        draws = sum(1 for r in self.results if r == "draw")
        return OpponentStats(
            opponent_id=self.opponent_id,
            wins=wins,
            losses=losses,
            draws=draws,
            last_updated_iteration=self.last_updated_iteration,
        )


def _learning_signal(win_rate: float) -> float:
    """Bell curve centered at 0.5 for maximum learning near 50%.

    Args:
        win_rate: Win rate between 0 and 1.

    Returns:
        Learning signal value, highest at win_rate=0.5.
    """
    return math.exp(-((win_rate - 0.5) ** 2) / 0.08)


class OpponentStatsTracker:
    """Track per-opponent win rates and compute adaptive sampling weights.

    Maintains statistics for each opponent and computes sampling weights
    that prioritize opponents where the agent has ~50% win rate, as these
    provide the strongest learning signal.
    """

    def __init__(self, window_size: int = 500) -> None:
        """Initialize the tracker.

        Args:
            window_size: Maximum number of games to track per opponent.
                Uses sliding window to focus on recent performance.
        """
        self._window_size = window_size
        self._stats: dict[str, _WindowedStats] = {}
        logger.debug(f"OpponentStatsTracker initialized with window_size={window_size}")

    def update(
        self,
        opponent_id: str,
        result: Literal["win", "loss", "draw"],
        iteration: int,
    ) -> None:
        """Record a single game result.

        Args:
            opponent_id: Identifier for the opponent.
            result: Game outcome from the agent's perspective.
            iteration: Current training iteration.
        """
        if opponent_id not in self._stats:
            self._stats[opponent_id] = _WindowedStats(opponent_id=opponent_id)

        stats = self._stats[opponent_id]
        stats.add_result(result, self._window_size)
        stats.last_updated_iteration = iteration

    def update_batch(
        self,
        results: list[tuple[str, Literal["win", "loss", "draw"]]],
        iteration: int,
    ) -> None:
        """Record multiple game results efficiently.

        Args:
            results: List of (opponent_id, result) tuples.
            iteration: Current training iteration.
        """
        for opponent_id, result in results:
            self.update(opponent_id, result, iteration)

    def get_stats(self, opponent_id: str) -> OpponentStats:
        """Get stats for a specific opponent.

        Args:
            opponent_id: Identifier for the opponent.

        Returns:
            OpponentStats for the specified opponent (empty stats if unknown).
        """
        if opponent_id not in self._stats:
            return OpponentStats(opponent_id=opponent_id)
        return self._stats[opponent_id].to_opponent_stats()

    def get_all_stats(self) -> dict[str, OpponentStats]:
        """Get all opponent stats.

        Returns:
            Dictionary mapping opponent_id to OpponentStats.
        """
        return {oid: ws.to_opponent_stats() for oid, ws in self._stats.items()}

    def compute_learning_weights(self, exploration_rate: float = 0.1) -> dict[str, float]:
        """Compute sampling weights based on win rates.

        Uses bell curve centered at 50% win rate - opponents where we have
        50% win rate provide maximum learning signal.

        Args:
            exploration_rate: Minimum weight floor for exploration (0-1).
                Ensures all opponents have some chance of being sampled.

        Returns:
            Dict mapping opponent_id to sampling weight (normalized to sum to 1).
        """
        if not self._stats:
            return {}

        # Compute raw learning signals
        raw_weights: dict[str, float] = {}
        for opponent_id, windowed_stats in self._stats.items():
            stats = windowed_stats.to_opponent_stats()
            signal = _learning_signal(stats.win_rate)
            # Apply exploration floor
            raw_weights[opponent_id] = max(signal, exploration_rate)

        # Normalize to sum to 1
        total = sum(raw_weights.values())
        if total == 0:
            # Uniform distribution if all weights are zero
            n = len(raw_weights)
            return dict.fromkeys(raw_weights, 1.0 / n)

        return {oid: w / total for oid, w in raw_weights.items()}

    def get_summary(self) -> dict[str, Any]:
        """Get summary dict for logging.

        Returns:
            Summary statistics including total games, per-opponent stats,
            and computed learning weights.
        """
        all_stats = self.get_all_stats()
        weights = self.compute_learning_weights()

        total_games = sum(s.games for s in all_stats.values())
        total_wins = sum(s.wins for s in all_stats.values())

        summary: dict[str, Any] = {
            "total_opponents": len(all_stats),
            "total_games": total_games,
            "overall_win_rate": total_wins / total_games if total_games > 0 else 0.5,
            "window_size": self._window_size,
            "opponents": {},
        }

        for opponent_id, stats in all_stats.items():
            summary["opponents"][opponent_id] = {
                "games": stats.games,
                "win_rate": stats.win_rate,
                "record": f"{stats.wins}-{stats.losses}-{stats.draws}",
                "learning_weight": weights.get(opponent_id, 0.0),
                "last_updated": stats.last_updated_iteration,
            }

        return summary

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpointing.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "window_size": self._window_size,
            "stats": {
                opponent_id: {
                    "opponent_id": opponent_id,
                    "results": list(ws.results),
                    "last_updated_iteration": ws.last_updated_iteration,
                }
                for opponent_id, ws in self._stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpponentStatsTracker:
        """Deserialize from checkpoint.

        Args:
            data: Dictionary from to_dict().

        Returns:
            Reconstructed OpponentStatsTracker instance.
        """
        tracker = cls(window_size=data.get("window_size", 500))

        for opponent_id, stats_data in data.get("stats", {}).items():
            windowed = _WindowedStats(
                opponent_id=opponent_id,
                results=deque(stats_data.get("results", [])),
                last_updated_iteration=stats_data.get("last_updated_iteration", 0),
            )
            tracker._stats[opponent_id] = windowed

        logger.debug(
            f"OpponentStatsTracker restored with {len(tracker._stats)} opponents"
        )
        return tracker


__all__ = [
    "OpponentStats",
    "OpponentStatsTracker",
]
