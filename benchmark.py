#!/usr/bin/env python3
"""Benchmark script for Beasty Bar AI agents.

Run with: uv run python benchmark.py

This will run a tournament between all available agents and display ELO rankings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _02_agents import HeuristicAgent, MCTSAgent, RandomAgent
from _03_training import Leaderboard, Tournament, TournamentConfig


def create_agents(include_slow: bool = False) -> dict:
    """Create the standard set of agents for benchmarking."""
    agents = {
        "Random": RandomAgent(seed=42),
        "Heuristic": HeuristicAgent(seed=42),
        "MCTS-500": MCTSAgent(iterations=500, determinizations=8, seed=42),
    }

    if include_slow:
        agents["MCTS-1000"] = MCTSAgent(iterations=1000, determinizations=10, seed=42)
        agents["MCTS-2000"] = MCTSAgent(iterations=2000, determinizations=12, seed=42)

    return agents


def run_tournament(
    games_per_matchup: int = 50,
    seed: int = 42,
    include_slow: bool = False,
    output_file: Path | None = None,
) -> Leaderboard:
    """Run a full round-robin tournament."""
    print("=" * 70)
    print("BEASTY BAR AI BENCHMARK")
    print("=" * 70)

    agents = create_agents(include_slow)
    print(f"\nAgents: {', '.join(agents.keys())}")
    print(f"Games per matchup: {games_per_matchup}")
    print(f"Seed: {seed}")

    config = TournamentConfig(
        games_per_matchup=games_per_matchup,
        alternate_starting=True,
        seed=seed,
        verbose=True,
    )

    tournament = Tournament(config)
    for name, agent in agents.items():
        tournament.register(name, agent)

    print("\n" + "-" * 70)
    print("RUNNING TOURNAMENT")
    print("-" * 70)

    tournament.round_robin()

    print("\n" + "=" * 70)
    print("FINAL ELO RANKINGS")
    print("=" * 70)
    print(tournament.summary())

    if output_file:
        tournament.leaderboard.save(output_file)
        print(f"\nResults saved to: {output_file}")

    return tournament.leaderboard


def single_match(
    agent_a_name: str,
    agent_b_name: str,
    games: int = 100,
    seed: int = 42,
) -> None:
    """Run a head-to-head match between two agents."""
    agents = create_agents(include_slow=True)

    if agent_a_name not in agents:
        print(f"Unknown agent: {agent_a_name}")
        print(f"Available: {', '.join(agents.keys())}")
        return

    if agent_b_name not in agents:
        print(f"Unknown agent: {agent_b_name}")
        print(f"Available: {', '.join(agents.keys())}")
        return

    print(f"\n{agent_a_name} vs {agent_b_name} ({games} games)")
    print("-" * 50)

    config = TournamentConfig(
        games_per_matchup=games,
        alternate_starting=True,
        seed=seed,
        verbose=True,
    )

    tournament = Tournament(config)
    tournament.register(agent_a_name, agents[agent_a_name])
    tournament.register(agent_b_name, agents[agent_b_name])

    wins_a, wins_b, draws = tournament.head_to_head(agent_a_name, agent_b_name)

    print(f"\nFinal: {agent_a_name} {wins_a} - {wins_b} {agent_b_name} (draws: {draws})")
    print("\nELO Ratings:")
    for stats in tournament.leaderboard.rankings():
        print(f"  {stats.name}: {stats.rating:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Beasty Bar AI Benchmark")
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Games per matchup (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slower MCTS variants (1000, 2000 iterations)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--match",
        nargs=2,
        metavar=("AGENT_A", "AGENT_B"),
        help="Run single match between two agents",
    )

    args = parser.parse_args()

    if args.match:
        single_match(
            args.match[0],
            args.match[1],
            games=args.games,
            seed=args.seed,
        )
    else:
        run_tournament(
            games_per_matchup=args.games,
            seed=args.seed,
            include_slow=args.include_slow,
            output_file=args.output,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
