"""Tournament utilities for bulk agent evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence, Tuple

from simulator import actions, simulate, state
from agents import DiegoAgent, FirstLegalAgent, GreedyAgent, RandomAgent
from agents.base import Agent


@dataclass
class SeriesConfig:
    """Configuration describing a head-to-head agent series."""

    games: int
    seed: int
    agent_a: Agent
    agent_b: Agent
    alternate_start: bool = True
    collect_actions: bool = False
    early_stop: EarlyStopConfig | None = None


@dataclass
class EarlyStopConfig:
    """Parameters governing early termination based on Wilson confidence interval."""

    confidence_level: float = 0.95
    min_games: int = 10
    threshold: float = 0.5
    ties_as_half: bool = True
    _z_value: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be between 0 and 1")
        if self.min_games < 1:
            raise ValueError("min_games must be at least 1")
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must be between 0 and 1")
        object.__setattr__(
            self,
            "_z_value",
            NormalDist().inv_cdf(0.5 + self.confidence_level / 2.0),
        )


@dataclass(frozen=True)
class ActionRecord:
    """Telemetry captured for a single action within a game."""

    turn: int
    player: int
    was_pass: bool
    hand_index: int
    params: tuple[int, ...]
    species: str
    strength: int
    points: int
    hand_before: tuple[str, ...]
    legal_options: tuple[tuple[int, str, tuple[int, ...]], ...]
    queue_before: tuple[tuple[int, str], ...]
    queue_after: tuple[tuple[int, str], ...]
    beasty_bar_after: tuple[tuple[int, str], ...]
    thats_it_after: tuple[tuple[int, str], ...]


@dataclass
class GameRecord:
    """Telemetry for a single simulated game."""

    index: int
    seed: int
    starting_player: int
    turns: int
    scores: tuple[int, int]
    winner: int | None
    actions: tuple[ActionRecord, ...] | None = None


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


def _wilson_score_interval(successes: float, total: int, z_value: float) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("total must be positive for Wilson interval")

    if successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")

    denom = 1.0 + (z_value * z_value) / total
    adjusted = successes + 0.5 * z_value * z_value
    center = (adjusted / total) / denom
    variance = (successes * (total - successes)) / total + 0.25 * z_value * z_value
    margin = z_value * math.sqrt(variance) / (total * denom)

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def _should_stop_early(
    *,
    wins_a: int,
    wins_b: int,
    ties: int,
    config: EarlyStopConfig,
) -> bool:
    """Return True when the Wilson CI indicates a decisive result."""

    games = wins_a + wins_b + ties
    if games < config.min_games:
        return False

    if games == 0:
        return False

    if config.ties_as_half:
        successes_a = wins_a + 0.5 * ties
    else:
        successes_a = wins_a

    if successes_a < 0:
        successes_a = 0.0
    if successes_a > games:
        successes_a = float(games)

    lower_a, upper_a = _wilson_score_interval(successes_a, games, config._z_value)

    if lower_a > config.threshold:
        return True
    if upper_a < config.threshold:
        return True

    return False


@dataclass
class LeaderboardEntry:
    """Aggregated ranking information for an agent."""

    name: str
    rating: float
    wins: int
    losses: int
    ties: int
    games: int
    win_rate: float
    average_score: float


def play_series(config: SeriesConfig) -> SeriesResult:
    """Play a block of games and collect telemetry."""

    if config.games < 1:
        raise ValueError("games must be at least 1")

    records: list[GameRecord] = []
    wins = [0, 0]
    ties = 0
    agents: Sequence[Agent] = (config.agent_a, config.agent_b)

    for offset in range(config.games):
        game_seed = config.seed + offset
        starting_player = (offset % 2) if config.alternate_start else 0
        current = simulate.new_game(game_seed, starting_player=starting_player)
        action_events: list[ActionRecord] | None = [] if config.collect_actions else None

        for agent in agents:
            agent.start_game(current)

        turns = 0
        while not simulate.is_terminal(current):
            player = current.active_player
            legal = simulate.legal_actions(current, player)
            if not legal:
                if action_events is not None:
                    action_events.append(_capture_pass(current, player))
                current = state.set_active_player(current, current.next_player(), advance_turn=True)
                turns += 1
                continue
            action = agents[player].select_action(current, legal)
            next_state = simulate.apply(current, action)
            if action_events is not None:
                action_events.append(
                    _capture_action(
                        before=current,
                        after=next_state,
                        player=player,
                        action=action,
                        legal=legal,
                    )
                )
            current = next_state
            turns += 1

        for agent in agents:
            agent.end_game(current)

        scores = simulate.score(current)
        winner = _winner(scores)
        if winner is None:
            ties += 1
        else:
            wins[winner] += 1
        records.append(
            GameRecord(
                index=offset,
                seed=game_seed,
                starting_player=starting_player,
                turns=turns,
                scores=scores,
                winner=winner,
                actions=tuple(action_events) if action_events is not None else None,
            )
        )

        if config.early_stop and _should_stop_early(
            wins_a=wins[0],
            wins_b=wins[1],
            ties=ties,
            config=config.early_stop,
        ):
            break

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

    data = [_game_record_to_dict(record) for record in records]
    path.write_text(json.dumps(data, indent=2))


def _winner(scores: Sequence[int]) -> int | None:
    if scores[0] == scores[1]:
        return None
    return 0 if scores[0] > scores[1] else 1


def _capture_action(
    *,
    before: state.State,
    after: state.State,
    player: int,
    action: actions.Action,
    legal: Sequence[actions.Action],
) -> ActionRecord:
    player_state = before.players[player]
    card = player_state.hand[action.hand_index]
    return ActionRecord(
        turn=before.turn,
        player=player,
        was_pass=False,
        hand_index=action.hand_index,
        params=action.params,
        species=card.species,
        strength=card.strength,
        points=card.points,
        hand_before=_snapshot_hand(player_state.hand),
        legal_options=_snapshot_legal(player_state.hand, legal),
        queue_before=_snapshot_zone(before.zones.queue),
        queue_after=_snapshot_zone(after.zones.queue),
        beasty_bar_after=_snapshot_zone(after.zones.beasty_bar),
        thats_it_after=_snapshot_zone(after.zones.thats_it),
    )


def _snapshot_zone(cards: Sequence[state.Card]) -> tuple[tuple[int, str], ...]:
    return tuple((card.owner, card.species) for card in cards)


def _snapshot_hand(cards: Sequence[state.Card]) -> tuple[str, ...]:
    return tuple(card.species for card in cards)


def _snapshot_legal(
    hand: Sequence[state.Card], legal: Sequence[actions.Action]
) -> tuple[tuple[int, str, tuple[int, ...]], ...]:
    options: list[tuple[int, str, tuple[int, ...]]] = []
    for option in legal:
        card = hand[option.hand_index]
        options.append((option.hand_index, card.species, option.params))
    return tuple(options)


def _capture_pass(game_state: state.State, player: int) -> ActionRecord:
    player_state = game_state.players[player]
    return ActionRecord(
        turn=game_state.turn,
        player=player,
        was_pass=True,
        hand_index=-1,
        params=(),
        species="",
        strength=0,
        points=0,
        hand_before=_snapshot_hand(player_state.hand),
        legal_options=(),
        queue_before=_snapshot_zone(game_state.zones.queue),
        queue_after=_snapshot_zone(game_state.zones.queue),
        beasty_bar_after=_snapshot_zone(game_state.zones.beasty_bar),
        thats_it_after=_snapshot_zone(game_state.zones.thats_it),
    )


def _game_record_to_dict(record: GameRecord) -> dict:
    payload = {
        "game": record.index,
        "seed": record.seed,
        "startingPlayer": record.starting_player,
        "turns": record.turns,
        "scores": list(record.scores),
        "winner": record.winner,
    }
    if record.actions is not None:
        payload["actions"] = [_action_record_to_dict(action) for action in record.actions]
    return payload


def _action_record_to_dict(action: ActionRecord) -> dict:
    return {
        "turn": action.turn,
        "player": action.player,
        "pass": action.was_pass,
        "handIndex": action.hand_index,
        "params": list(action.params),
        "species": action.species,
        "strength": action.strength,
        "points": action.points,
        "handBefore": list(action.hand_before),
        "legalOptions": [
            {
                "handIndex": option[0],
                "species": option[1],
                "params": list(option[2]),
            }
            for option in action.legal_options
        ],
        "queueBefore": [
            {"owner": owner, "species": species} for owner, species in action.queue_before
        ],
        "queueAfter": [
            {"owner": owner, "species": species} for owner, species in action.queue_after
        ],
        "beastyBarAfter": [
            {"owner": owner, "species": species} for owner, species in action.beasty_bar_after
        ],
        "thatsItAfter": [
            {"owner": owner, "species": species} for owner, species in action.thats_it_after
        ],
    }


_AGENT_LOOKUP = {
    "first": FirstLegalAgent,
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "diego": DiegoAgent,
}

_DEFAULT_ELO_BASE = 1500.0
_DEFAULT_ELO_K = 32.0


def _agent_from_name(name: str) -> Agent:
    try:
        factory = _AGENT_LOOKUP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent '{name}'") from exc
    return factory()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent tournaments and export telemetry")
    parser.add_argument(
        "agent_a",
        nargs="?",
        help="Name of the first agent (first, random, greedy, diego)",
    )
    parser.add_argument(
        "agent_b",
        nargs="?",
        help="Name of the second agent (first, random, greedy, diego)",
    )
    parser.add_argument("--games", type=int, default=1000, help="Number of games to play")
    parser.add_argument("--seed", type=int, default=2025, help="Base seed for the series")
    parser.add_argument("--no-alternate-start", action="store_true", help="Disable alternating starting player")
    parser.add_argument("--csv", type=Path, help="Path to write per-game telemetry as CSV")
    parser.add_argument("--json", type=Path, help="Path to write per-game telemetry as JSON")
    parser.add_argument("--log-actions", action="store_true", help="Record per-action telemetry")
    parser.add_argument(
        "--round-robin",
        action="store_true",
        help="Run a round robin across all built-in agents",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory to write JSON telemetry for each matchup when using round robin",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop the series early when a Wilson confidence interval shows a decisive winner",
    )
    parser.add_argument(
        "--early-stop-min-games",
        type=int,
        default=None,
        help="Minimum games required before performing early stop checks",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=None,
        help="Win-rate threshold considered decisive for early stopping (default: 0.5)",
    )
    parser.add_argument(
        "--early-stop-confidence",
        type=float,
        default=None,
        help="Confidence level to use for the Wilson interval (default: 0.95)",
    )
    parser.add_argument(
        "--elo-base",
        type=float,
        default=_DEFAULT_ELO_BASE,
        help="Starting rating for Elo leaderboard (default: 1500)",
    )
    parser.add_argument(
        "--elo-k",
        type=float,
        default=_DEFAULT_ELO_K,
        help="K-factor used for Elo updates (default: 32)",
    )
    parser.add_argument(
        "--leaderboard-json",
        type=Path,
        help="Optional path to export the Elo leaderboard as JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> SeriesResult:
    args = _parse_args(argv)
    log_actions = args.log_actions or args.round_robin

    early_stop_config: EarlyStopConfig | None = None
    if (
        args.early_stop
        or args.early_stop_min_games is not None
        or args.early_stop_threshold is not None
        or args.early_stop_confidence is not None
    ):
        kwargs = {}
        if args.early_stop_min_games is not None:
            kwargs["min_games"] = args.early_stop_min_games
        if args.early_stop_threshold is not None:
            kwargs["threshold"] = args.early_stop_threshold
        if args.early_stop_confidence is not None:
            kwargs["confidence_level"] = args.early_stop_confidence
        early_stop_config = EarlyStopConfig(**kwargs)

    if args.round_robin:
        results = _run_round_robin(
            games=args.games,
            seed=args.seed,
            alternate_start=not args.no_alternate_start,
            collect_actions=log_actions,
            log_dir=args.log_dir,
            elo_base=args.elo_base,
            elo_k=args.elo_k,
            leaderboard_path=args.leaderboard_json,
            early_stop=early_stop_config,
        )
        return results

    if not args.agent_a or not args.agent_b:
        raise SystemExit("agent_a and agent_b are required unless --round-robin is used")

    agent_a = _agent_from_name(args.agent_a)
    agent_b = _agent_from_name(args.agent_b)

    config = SeriesConfig(
        games=args.games,
        seed=args.seed,
        agent_a=agent_a,
        agent_b=agent_b,
        alternate_start=not args.no_alternate_start,
        collect_actions=log_actions,
        early_stop=early_stop_config,
    )

    result = play_series(config)

    _print_summary(result.summary, label_a=args.agent_a, label_b=args.agent_b)

    if args.csv:
        export_csv(args.csv, result.records)
        print(f"Wrote CSV telemetry to {args.csv}")
    if args.json:
        export_json(args.json, result.records)
        print(f"Wrote JSON telemetry to {args.json}")

    return result


def _run_round_robin(
    *,
    games: int,
    seed: int,
    alternate_start: bool,
    collect_actions: bool,
    log_dir: Path | None,
    elo_base: float,
    elo_k: float,
    leaderboard_path: Path | None,
    early_stop: EarlyStopConfig | None,
) -> SeriesResult:
    agent_names = sorted(_AGENT_LOOKUP.keys())
    pairings = list(itertools.combinations(agent_names, 2))
    if not pairings:
        raise SystemExit("No agent pairings available for round robin")

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    aggregated_records: list[GameRecord] = []
    combined_wins = {name: 0 for name in agent_names}
    combined_scores = {name: 0.0 for name in agent_names}
    games_per_agent = {name: 0 for name in agent_names}
    combined_stats = {
        name: {"wins": 0, "losses": 0, "ties": 0, "games": 0}
        for name in agent_names
    }
    combined_turns = 0
    total_games = 0
    ratings = {name: float(elo_base) for name in agent_names}

    for idx, (name_a, name_b) in enumerate(pairings):
        agent_a = _agent_from_name(name_a)
        agent_b = _agent_from_name(name_b)
        series_seed = seed + idx * max(games, 1)
        config = SeriesConfig(
            games=games,
            seed=series_seed,
            agent_a=agent_a,
            agent_b=agent_b,
            alternate_start=alternate_start,
            collect_actions=collect_actions,
            early_stop=early_stop,
        )
        result = play_series(config)
        _print_summary(result.summary, label_a=name_a, label_b=name_b)

        if log_dir:
            out_path = log_dir / f"{name_a}_vs_{name_b}.json"
            export_json(out_path, result.records)
            print(f"Wrote JSON telemetry to {out_path}")

        aggregated_records.extend(result.records)

        wins_a, wins_b = result.summary.wins
        combined_wins[name_a] += wins_a
        combined_wins[name_b] += wins_b
        combined_scores[name_a] += result.summary.average_scores[0] * result.summary.games
        combined_scores[name_b] += result.summary.average_scores[1] * result.summary.games
        games_per_agent[name_a] += result.summary.games
        games_per_agent[name_b] += result.summary.games
        combined_stats[name_a]["wins"] += wins_a
        combined_stats[name_a]["losses"] += wins_b
        combined_stats[name_a]["ties"] += result.summary.ties
        combined_stats[name_a]["games"] += result.summary.games
        combined_stats[name_b]["wins"] += wins_b
        combined_stats[name_b]["losses"] += wins_a
        combined_stats[name_b]["ties"] += result.summary.ties
        combined_stats[name_b]["games"] += result.summary.games
        combined_turns += result.summary.average_turns * result.summary.games
        total_games += result.summary.games

        _apply_elo_update(
            ratings,
            name_a,
            name_b,
            wins_a=wins_a,
            wins_b=wins_b,
            ties=result.summary.ties,
            k_factor=elo_k,
        )

    overall_summary = summarize(aggregated_records)
    print("\n=== Aggregate Round Robin Summary ===")
    _print_summary(overall_summary, label_a="agent_a", label_b="agent_b")
    print("\nWins by agent:")
    for name in agent_names:
        print(f"  {name}: {combined_wins[name]}")

    print("\nAverage score by agent:")
    for name in agent_names:
        agent_games = games_per_agent[name]
        avg_score = combined_scores[name] / max(agent_games, 1)
        print(f"  {name}: {avg_score:.2f}")

    print(f"\nAverage turns across all games: {combined_turns / max(total_games, 1):.2f}")

    leaderboard = _build_leaderboard(
        agent_names,
        ratings,
        combined_stats,
        combined_scores,
        games_per_agent,
    )
    print("\n=== Elo Leaderboard ===")
    _print_leaderboard(leaderboard)

    if leaderboard_path:
        _export_leaderboard(leaderboard_path, leaderboard)
        print(f"Wrote Elo leaderboard to {leaderboard_path}")

    return SeriesResult(summary=overall_summary, records=aggregated_records)


def _print_summary(summary: SeriesSummary, *, label_a: str, label_b: str) -> None:
    print(
        f"Games: {summary.games}\n"
        f"Wins: {label_a}={summary.wins[0]} {label_b}={summary.wins[1]} Ties={summary.ties}\n"
        f"Average scores: {label_a}={summary.average_scores[0]:.2f} {label_b}={summary.average_scores[1]:.2f}\n"
        f"Average turns: {summary.average_turns:.2f}"
    )


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _apply_elo_update(
    ratings: dict[str, float],
    name_a: str,
    name_b: str,
    *,
    wins_a: int,
    wins_b: int,
    ties: int,
    k_factor: float,
) -> None:
    games = wins_a + wins_b + ties
    if games == 0:
        return

    score_a = wins_a + 0.5 * ties
    score_b = wins_b + 0.5 * ties

    rating_a = ratings[name_a]
    rating_b = ratings[name_b]

    expected_a = _expected_score(rating_a, rating_b) * games
    expected_b = games - expected_a

    ratings[name_a] = rating_a + k_factor * (score_a - expected_a)
    ratings[name_b] = rating_b + k_factor * (score_b - expected_b)


def _build_leaderboard(
    agent_names: Sequence[str],
    ratings: dict[str, float],
    stats: dict[str, dict[str, int]],
    combined_scores: dict[str, float],
    games_per_agent: dict[str, int],
) -> list[LeaderboardEntry]:
    entries: list[LeaderboardEntry] = []
    for name in agent_names:
        games = stats[name]["games"]
        wins = stats[name]["wins"]
        losses = stats[name]["losses"]
        ties = stats[name]["ties"]
        win_rate = wins / games if games else 0.0
        avg_score = combined_scores[name] / games if games else 0.0
        entries.append(
            LeaderboardEntry(
                name=name,
                rating=ratings[name],
                wins=wins,
                losses=losses,
                ties=ties,
                games=games,
                win_rate=win_rate,
                average_score=avg_score,
            )
        )

    entries.sort(key=lambda entry: entry.rating, reverse=True)
    return entries


def _print_leaderboard(leaderboard: Sequence[LeaderboardEntry]) -> None:
    if not leaderboard:
        print("No leaderboard entries")
        return

    print(f"{'Rank':>4s} {'Agent':12s} {'Rating':>8s} {'Record':>18s} {'WinRate':>8s} {'AvgScore':>9s}")
    for idx, entry in enumerate(leaderboard, start=1):
        record = f"{entry.wins}-{entry.losses}-{entry.ties}"
        print(
            f"{idx:>4d} {entry.name:12s} {entry.rating:8.1f} {record:>18s} "
            f"{entry.win_rate:8.3f} {entry.average_score:9.2f}"
        )


def _export_leaderboard(path: Path, leaderboard: Sequence[LeaderboardEntry]) -> None:
    payload = [
        {
            "agent": entry.name,
            "rating": entry.rating,
            "wins": entry.wins,
            "losses": entry.losses,
            "ties": entry.ties,
            "games": entry.games,
            "winRate": entry.win_rate,
            "averageScore": entry.average_score,
        }
        for entry in leaderboard
    ]
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI support
    main()
