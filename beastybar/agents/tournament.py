"""Tournament utilities for bulk agent evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .. import actions, simulate, state
from . import baselines, diego, frontrunner, greedy, killer
from .base import Agent


@dataclass
class SeriesConfig:
    """Configuration describing a head-to-head agent series."""

    games: int
    seed: int
    agent_a: Agent
    agent_b: Agent
    alternate_start: bool = True


@dataclass
class GameRecord:
    """Telemetry for a single simulated game."""

    index: int
    seed: int
    starting_player: int
    turns: int
    scores: tuple[int, int]
    winner: int | None


@dataclass
class SeriesSummary:
    """Aggregated statistics for a series of games."""

    games: int
    wins: tuple[int, int]
    ties: int
    average_scores: tuple[float, float]
    average_turns: float


@dataclass
class SeriesResult:
    """Verbose result containing both summary statistics and per-game data."""

    summary: SeriesSummary
    records: list[GameRecord]


def play_series(config: SeriesConfig) -> SeriesResult:
    """Play a block of games and collect telemetry."""

    if config.games < 1:
        raise ValueError("games must be at least 1")

    records: list[GameRecord] = []
    agents: Sequence[Agent] = (config.agent_a, config.agent_b)

    for offset in range(config.games):
        game_seed = config.seed + offset
        starting_player = (offset % 2) if config.alternate_start else 0
        current = simulate.new_game(game_seed, starting_player=starting_player)

        for agent in agents:
            agent.start_game(current)

        turns = 0
        while not simulate.is_terminal(current):
            player = current.active_player
            legal = simulate.legal_actions(current, player)
            action = agents[player].select_action(current, legal)
            current = simulate.apply(current, action)
            turns += 1

        for agent in agents:
            agent.end_game(current)

        scores = simulate.score(current)
        winner = _winner(scores)
        records.append(
            GameRecord(
                index=offset,
                seed=game_seed,
                starting_player=starting_player,
                turns=turns,
                scores=scores,
                winner=winner,
            )
        )

    summary = summarize(records)
    return SeriesResult(summary=summary, records=records)


def summarize(records: Iterable[GameRecord]) -> SeriesSummary:
    records = list(records)
    total_games = len(records)
    if total_games == 0:
        raise ValueError("No records provided for summary")

    wins = [0, 0]
    ties = 0
    total_scores = [0, 0]
    total_turns = 0

    for record in records:
        if record.winner is None:
            ties += 1
        else:
            wins[record.winner] += 1
        total_scores[0] += record.scores[0]
        total_scores[1] += record.scores[1]
        total_turns += record.turns

    avg_scores = (total_scores[0] / total_games, total_scores[1] / total_games)
    avg_turns = total_turns / total_games

    return SeriesSummary(
        games=total_games,
        wins=(wins[0], wins[1]),
        ties=ties,
        average_scores=avg_scores,
        average_turns=avg_turns,
    )


def export_csv(path: Path, records: Iterable[GameRecord]) -> None:
    """Write per-game telemetry to CSV."""

    rows = list(records)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["game", "seed", "starting_player", "turns", "score_a", "score_b", "winner"])
        for record in rows:
            writer.writerow(
                [
                    record.index,
                    record.seed,
                    record.starting_player,
                    record.turns,
                    record.scores[0],
                    record.scores[1],
                    record.winner if record.winner is not None else "tie",
                ]
            )


def export_json(path: Path, records: Iterable[GameRecord]) -> None:
    """Write per-game telemetry to JSON."""

    data = [
        {
            "game": record.index,
            "seed": record.seed,
            "startingPlayer": record.starting_player,
            "turns": record.turns,
            "scores": list(record.scores),
            "winner": record.winner,
        }
        for record in records
    ]
    path.write_text(json.dumps(data, indent=2))


def _winner(scores: Sequence[int]) -> int | None:
    if scores[0] == scores[1]:
        return None
    return 0 if scores[0] > scores[1] else 1


def _agent_from_name(name: str) -> Agent:
    lookup = {
        "first": baselines.FirstLegalAgent,
        "random": baselines.RandomAgent,
        "greedy": greedy.GreedyAgent,
        "frontrunner": frontrunner.FrontRunnerAgent,
        "diego": diego.DiegoAgent,
        "killer": killer.KillerAgent,
    }
    try:
        factory = lookup[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent '{name}'") from exc
    return factory()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent tournaments and export telemetry")
    parser.add_argument(
        "agent_a",
        help="Name of the first agent (first, random, greedy, frontrunner, diego, killer)",
    )
    parser.add_argument(
        "agent_b",
        help="Name of the second agent (first, random, greedy, frontrunner, diego, killer)",
    )
    parser.add_argument("--games", type=int, default=1000, help="Number of games to play")
    parser.add_argument("--seed", type=int, default=2025, help="Base seed for the series")
    parser.add_argument("--no-alternate-start", action="store_true", help="Disable alternating starting player")
    parser.add_argument("--csv", type=Path, help="Path to write per-game telemetry as CSV")
    parser.add_argument("--json", type=Path, help="Path to write per-game telemetry as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> SeriesResult:
    args = _parse_args(argv)
    agent_a = _agent_from_name(args.agent_a)
    agent_b = _agent_from_name(args.agent_b)

    config = SeriesConfig(
        games=args.games,
        seed=args.seed,
        agent_a=agent_a,
        agent_b=agent_b,
        alternate_start=not args.no_alternate_start,
    )

    result = play_series(config)

    summary = result.summary
    print(
        f"Games: {summary.games}\n"
        f"Wins: A={summary.wins[0]} B={summary.wins[1]} Ties={summary.ties}\n"
        f"Average scores: A={summary.average_scores[0]:.2f} B={summary.average_scores[1]:.2f}\n"
        f"Average turns: {summary.average_turns:.2f}"
    )

    if args.csv:
        export_csv(args.csv, result.records)
        print(f"Wrote CSV telemetry to {args.csv}")
    if args.json:
        export_json(args.json, result.records)
        print(f"Wrote JSON telemetry to {args.json}")

    return result


if __name__ == "__main__":  # pragma: no cover - CLI support
    main()
