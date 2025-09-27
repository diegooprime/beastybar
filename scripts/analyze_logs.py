"""Aggregate telemetry logs to surface high-performing action heuristics."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass
class ActionStats:
    count: int = 0
    wins: int = 0
    ties: int = 0
    total_score_diff: float = 0.0
    total_points: float = 0.0
    total_queue_before: float = 0.0
    total_legal_options: float = 0.0

    def update(
        self,
        *,
        score_diff: float,
        points: float,
        queue_len: int,
        legal_options: int,
    ) -> None:
        self.count += 1
        if score_diff > 0:
            self.wins += 1
        elif score_diff == 0:
            self.ties += 1
        self.total_score_diff += score_diff
        self.total_points += points
        self.total_queue_before += queue_len
        self.total_legal_options += legal_options

    def finalize(self) -> Mapping[str, float]:
        if self.count == 0:
            raise ValueError("Cannot finalize stats with zero count")
        avg_diff = self.total_score_diff / self.count
        return {
            "count": self.count,
            "winRate": self.wins / self.count,
            "tieRate": self.ties / self.count,
            "avgScoreDiff": avg_diff,
            "avgPoints": self.total_points / self.count,
            "avgQueueBefore": self.total_queue_before / self.count,
            "avgLegalOptions": self.total_legal_options / self.count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-action telemetry logs")
    parser.add_argument("log_dir", type=Path, help="Directory containing *_vs_*.json telemetry files")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Minimum number of samples required for a heuristic to be ranked",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the aggregated table as JSON",
    )
    return parser.parse_args()


def iter_log_files(log_dir: Path) -> Iterable[Path]:
    yield from sorted(log_dir.glob("*_vs_*.json"))


def analyze(
    log_dir: Path,
) -> tuple[dict[str, int], dict[tuple[str, str, str, str], ActionStats]]:
    totals_by_agent: dict[str, int] = defaultdict(int)
    stats_by_key: dict[tuple[str, str, str, str], ActionStats] = defaultdict(ActionStats)

    for path in iter_log_files(log_dir):
        agent_a, agent_b = _agents_from_filename(path)
        with path.open() as handle:
            games = json.load(handle)
        for game in games:
            scores = game["scores"]
            for action in game.get("actions", []):
                agent = agent_a if action["player"] == 0 else agent_b
                opponent = agent_b if action["player"] == 0 else agent_a
                score_diff = scores[action["player"]] - scores[1 - action["player"]]
                species = action["species"] if not action["pass"] else "pass"
                queue_bucket = _bucket_queue_len(len(action["queueBefore"]))
                front_owner = _front_owner_label(action, acting_player=action["player"])
                key = (agent, species, queue_bucket, front_owner)

                stats = stats_by_key[key]
                stats.update(
                    score_diff=score_diff,
                    points=action["points"],
                    queue_len=len(action["queueBefore"]),
                    legal_options=len(action["legalOptions"]),
                )
                stats_by_key[key] = stats

                totals_by_agent[agent] += 1
                # ensure opponent is tracked for total counts even if they pass less often
                totals_by_agent[opponent] += 0

    return totals_by_agent, stats_by_key


def rank_heuristics(
    totals_by_agent: Mapping[str, int],
    stats_by_key: Mapping[tuple[str, str, str, str], ActionStats],
    *,
    min_samples: int,
) -> list[dict[str, object]]:
    finalized: list[dict[str, object]] = []
    diffs: list[float] = []

    for (agent, species, queue_bucket, front_owner), stats in stats_by_key.items():
        if stats.count < min_samples:
            continue
        metrics = stats.finalize()
        metrics.update({
            "agent": agent,
            "species": species,
            "queueBucket": queue_bucket,
            "frontOwner": front_owner,
            "actionShare": stats.count / max(totals_by_agent.get(agent, 1), 1),
        })
        diffs.append(metrics["avgScoreDiff"])
        finalized.append(metrics)

    if not finalized:
        return []

    diff_min = min(diffs)
    diff_max = max(diffs)
    span = diff_max - diff_min
    for entry in finalized:
        if span == 0:
            normalized = 0.5
        else:
            normalized = (entry["avgScoreDiff"] - diff_min) / span
        score = 0.6 * entry["winRate"] + 0.4 * normalized
        entry["rankingScore"] = score

    finalized.sort(key=lambda item: item["rankingScore"], reverse=True)
    return finalized


def _agents_from_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "_vs_" not in stem:
        raise ValueError(f"Unexpected telemetry filename: {path.name}")
    left, right = stem.split("_vs_", maxsplit=1)
    return left, right


def _bucket_queue_len(length: int) -> str:
    if length == 0:
        return "empty"
    if length <= 2:
        return "short"
    if length <= 4:
        return "mid"
    return "full"


def _front_owner_label(action: Mapping[str, object], *, acting_player: int) -> str:
    queue = action["queueBefore"]
    if not queue:
        return "none"
    owner = queue[0]["owner"]
    if owner == acting_player:
        return "ally"
    return "enemy"


def main() -> None:
    args = parse_args()
    totals_by_agent, stats_by_key = analyze(args.log_dir)
    ranked = rank_heuristics(totals_by_agent, stats_by_key, min_samples=args.min_samples)

    if args.output:
        args.output.write_text(json.dumps(ranked, indent=2))

    top = ranked[:10]
    if not top:
        print("No heuristics met the sample threshold.")
        return

    print("Top heuristics (agent, species, context):")
    for idx, entry in enumerate(top, start=1):
        context = f"queue={entry['queueBucket']} front={entry['frontOwner']}"
        print(
            f"{idx:2d}. {entry['agent']:12s} plays {entry['species'] or 'pass':5s} | {context} | "
            f"rank={entry['rankingScore']:.3f} winRate={entry['winRate']:.3f} "
            f"avgDiff={entry['avgScoreDiff']:.2f} count={entry['count']} share={entry['actionShare']:.3f}"
        )


if __name__ == "__main__":
    main()
