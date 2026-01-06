"""ELO rating system implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class EloRating:
    """ELO rating calculator."""

    k_factor: float = 32.0
    initial_rating: float = 1500.0

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float,
    ) -> tuple[float, float]:
        """Update ratings based on match result.

        Args:
            rating_a: Current rating of player A.
            rating_b: Current rating of player B.
            score_a: Actual score for player A (1.0 win, 0.5 draw, 0.0 loss).

        Returns:
            Tuple of (new_rating_a, new_rating_b).
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        score_b = 1.0 - score_a

        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        return new_rating_a, new_rating_b


@dataclass
class PlayerStats:
    """Statistics for a single player/agent."""

    name: str
    rating: float = 1500.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_points_scored: int = 0
    total_points_allowed: int = 0
    rating_history: list[float] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games

    @property
    def avg_margin(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.total_points_scored - self.total_points_allowed) / self.games

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rating": self.rating,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "total_points_scored": self.total_points_scored,
            "total_points_allowed": self.total_points_allowed,
            "games": self.games,
            "win_rate": self.win_rate,
            "avg_margin": self.avg_margin,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlayerStats:
        return cls(
            name=data["name"],
            rating=data.get("rating", 1500.0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            draws=data.get("draws", 0),
            total_points_scored=data.get("total_points_scored", 0),
            total_points_allowed=data.get("total_points_allowed", 0),
        )


class Leaderboard:
    """Maintains ELO ratings and statistics for multiple agents."""

    def __init__(
        self,
        elo: EloRating | None = None,
        initial_rating: float = 1500.0,
    ):
        self._elo = elo or EloRating(initial_rating=initial_rating)
        self._players: dict[str, PlayerStats] = {}

    def register(self, name: str) -> None:
        """Register a new player with initial rating."""
        if name not in self._players:
            self._players[name] = PlayerStats(
                name=name,
                rating=self._elo.initial_rating,
            )

    def get_rating(self, name: str) -> float:
        """Get current rating for a player."""
        if name not in self._players:
            self.register(name)
        return self._players[name].rating

    def get_stats(self, name: str) -> PlayerStats:
        """Get full statistics for a player."""
        if name not in self._players:
            self.register(name)
        return self._players[name]

    def record_match(
        self,
        player_a: str,
        player_b: str,
        score_a: int,
        score_b: int,
    ) -> tuple[float, float]:
        """Record a match result and update ratings.

        Args:
            player_a: Name of first player.
            player_b: Name of second player.
            score_a: Points scored by player A.
            score_b: Points scored by player B.

        Returns:
            Tuple of (new_rating_a, new_rating_b).
        """
        self.register(player_a)
        self.register(player_b)

        stats_a = self._players[player_a]
        stats_b = self._players[player_b]

        # Determine match outcome
        if score_a > score_b:
            result_a = 1.0
            stats_a.wins += 1
            stats_b.losses += 1
        elif score_a < score_b:
            result_a = 0.0
            stats_a.losses += 1
            stats_b.wins += 1
        else:
            result_a = 0.5
            stats_a.draws += 1
            stats_b.draws += 1

        # Update points
        stats_a.total_points_scored += score_a
        stats_a.total_points_allowed += score_b
        stats_b.total_points_scored += score_b
        stats_b.total_points_allowed += score_a

        # Update ELO ratings
        new_a, new_b = self._elo.update_ratings(stats_a.rating, stats_b.rating, result_a)
        stats_a.rating = new_a
        stats_b.rating = new_b

        # Track history
        stats_a.rating_history.append(new_a)
        stats_b.rating_history.append(new_b)

        return new_a, new_b

    def rankings(self) -> list[PlayerStats]:
        """Get all players sorted by rating (descending)."""
        return sorted(self._players.values(), key=lambda p: -p.rating)

    def summary(self) -> str:
        """Generate a formatted leaderboard summary."""
        lines = [
            "=" * 80,
            f"{'Rank':<6}{'Name':<25}{'Rating':<10}{'W-L-D':<15}{'Win%':<10}{'Margin':<10}",
            "-" * 80,
        ]

        for rank, player in enumerate(self.rankings(), 1):
            record = f"{player.wins}-{player.losses}-{player.draws}"
            lines.append(
                f"{rank:<6}{player.name:<25}{player.rating:<10.1f}"
                f"{record:<15}{player.win_rate * 100:<10.1f}{player.avg_margin:<+10.2f}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save leaderboard to JSON file."""
        data = {
            "players": {name: stats.to_dict() for name, stats in self._players.items()},
            "elo_config": {
                "k_factor": self._elo.k_factor,
                "initial_rating": self._elo.initial_rating,
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> Leaderboard:
        """Load leaderboard from JSON file."""
        data = json.loads(path.read_text())
        elo_config = data.get("elo_config", {})
        leaderboard = cls(
            elo=EloRating(
                k_factor=elo_config.get("k_factor", 32.0),
                initial_rating=elo_config.get("initial_rating", 1500.0),
            )
        )
        for name, stats_data in data.get("players", {}).items():
            leaderboard._players[name] = PlayerStats.from_dict(stats_data)
        return leaderboard


__all__ = ["EloRating", "Leaderboard", "PlayerStats"]
