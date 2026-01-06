"""Tournament runner for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from _01_simulator import engine, simulate

from .elo import Leaderboard

if TYPE_CHECKING:
    from _01_simulator.simulate import AgentFn


@dataclass
class MatchResult:
    """Result of a single match."""

    player_a: str
    player_b: str
    score_a: int
    score_b: int
    seed: int
    winner: str | None = None

    def __post_init__(self) -> None:
        if self.score_a > self.score_b:
            self.winner = self.player_a
        elif self.score_b > self.score_a:
            self.winner = self.player_b
        else:
            self.winner = None


@dataclass
class TournamentConfig:
    """Configuration for tournament runs."""

    games_per_matchup: int = 100
    alternate_starting: bool = True
    seed: int = 42
    verbose: bool = True


@dataclass
class AgentEntry:
    """An agent registered in the tournament."""

    name: str
    agent: AgentFn


class Tournament:
    """Runs tournaments between multiple agents."""

    def __init__(self, config: TournamentConfig | None = None):
        self._config = config or TournamentConfig()
        self._agents: dict[str, AgentEntry] = {}
        self._leaderboard = Leaderboard()
        self._results: list[MatchResult] = []

    def register(self, name: str, agent: AgentFn) -> None:
        """Register an agent for the tournament."""
        self._agents[name] = AgentEntry(name=name, agent=agent)
        self._leaderboard.register(name)

    def head_to_head(
        self,
        name_a: str,
        name_b: str,
        games: int | None = None,
    ) -> tuple[int, int, int]:
        """Run head-to-head matches between two agents.

        Args:
            name_a: Name of first agent.
            name_b: Name of second agent.
            games: Number of games (overrides config if provided).

        Returns:
            Tuple of (wins_a, wins_b, draws).
        """
        if name_a not in self._agents:
            raise ValueError(f"Agent '{name_a}' not registered")
        if name_b not in self._agents:
            raise ValueError(f"Agent '{name_b}' not registered")

        agent_a = self._agents[name_a].agent
        agent_b = self._agents[name_b].agent
        num_games = games or self._config.games_per_matchup

        wins_a, wins_b, draws = 0, 0, 0
        seed = self._config.seed

        for i in range(num_games):
            # Alternate starting player
            if self._config.alternate_starting and i % 2 == 1:
                current_a, current_b = agent_b, agent_a
                swapped = True
            else:
                current_a, current_b = agent_a, agent_b
                swapped = False

            # Run single game
            config = simulate.SimulationConfig(
                seed=seed + i,
                games=1,
                agent_a=current_a,
                agent_b=current_b,
            )

            for final_state in simulate.run(config):
                scores = engine.score(final_state)

                if swapped:
                    score_a, score_b = scores[1], scores[0]
                else:
                    score_a, score_b = scores[0], scores[1]

                # Record result
                result = MatchResult(
                    player_a=name_a,
                    player_b=name_b,
                    score_a=score_a,
                    score_b=score_b,
                    seed=seed + i,
                )
                self._results.append(result)

                # Update leaderboard
                self._leaderboard.record_match(name_a, name_b, score_a, score_b)

                # Track wins/draws
                if score_a > score_b:
                    wins_a += 1
                elif score_b > score_a:
                    wins_b += 1
                else:
                    draws += 1

            if self._config.verbose and (i + 1) % 10 == 0:
                print(f"  {name_a} vs {name_b}: {i + 1}/{num_games} games ({wins_a}-{wins_b}-{draws})")

        return wins_a, wins_b, draws

    def round_robin(self) -> dict[tuple[str, str], tuple[int, int, int]]:
        """Run round-robin tournament between all registered agents.

        Returns:
            Dictionary mapping (agent_a, agent_b) to (wins_a, wins_b, draws).
        """
        results: dict[tuple[str, str], tuple[int, int, int]] = {}
        agents = list(self._agents.keys())

        total_matchups = len(agents) * (len(agents) - 1) // 2
        current_matchup = 0

        for i, name_a in enumerate(agents):
            for name_b in agents[i + 1 :]:
                current_matchup += 1
                if self._config.verbose:
                    print(f"\nMatchup {current_matchup}/{total_matchups}: {name_a} vs {name_b}")

                wins_a, wins_b, draws = self.head_to_head(name_a, name_b)
                results[(name_a, name_b)] = (wins_a, wins_b, draws)

                if self._config.verbose:
                    print(f"  Final: {name_a} {wins_a} - {wins_b} {name_b} (draws: {draws})")

        return results

    @property
    def leaderboard(self) -> Leaderboard:
        """Get the current leaderboard."""
        return self._leaderboard

    @property
    def results(self) -> list[MatchResult]:
        """Get all match results."""
        return self._results

    def summary(self) -> str:
        """Generate tournament summary."""
        return self._leaderboard.summary()


def benchmark_agent(
    agent: AgentFn,
    name: str,
    opponents: dict[str, AgentFn],
    games_per_opponent: int = 50,
    seed: int = 42,
) -> Leaderboard:
    """Benchmark a single agent against multiple opponents.

    Args:
        agent: The agent to benchmark.
        name: Name for the agent.
        opponents: Dictionary of opponent name -> agent.
        games_per_opponent: Games to play against each opponent.
        seed: Random seed.

    Returns:
        Leaderboard with results.
    """
    config = TournamentConfig(
        games_per_matchup=games_per_opponent,
        seed=seed,
        verbose=True,
    )

    tournament = Tournament(config)
    tournament.register(name, agent)
    for opp_name, opp_agent in opponents.items():
        tournament.register(opp_name, opp_agent)

    print(f"\nBenchmarking {name} against {len(opponents)} opponents...")
    print("=" * 60)

    for opp_name in opponents:
        tournament.head_to_head(name, opp_name)

    print("\n" + tournament.summary())
    return tournament.leaderboard


__all__ = ["MatchResult", "Tournament", "TournamentConfig", "benchmark_agent"]
