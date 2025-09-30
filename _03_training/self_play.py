"""Self-play RL training entry point.

This CLI seeds run metadata, prepares artifact directories, and executes a basic
self-play PPO loop using the simulator. Checkpoints and metrics are written to
`_03_training/artifacts/<run_id>/` so tournaments and the UI can pick up the
latest policy."""
from __future__ import annotations

import argparse
import itertools
import json
import numbers
import os
import random
import statistics
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from _01_simulator import action_space
from _02_agents import DiegoAgent, FirstLegalAgent, GreedyAgent, RandomAgent, SelfPlayRLAgent
from _02_agents.base import Agent

from . import encoders, models, policy_loader, ppo, rollout, tournament

_DEFAULT_ARTIFACT_ROOT = Path("_03_training/artifacts")
_DEFAULT_OPPONENT_POOL = ["first", "random", "greedy", "diego"]
_DEFAULT_EVAL_GAMES = 200
_DEFAULT_EVAL_SEED = 4_096
_EVAL_ELO_BASE = 1500.0
_EVAL_ELO_K = 32.0
_ROLLING_METRIC_WINDOW = 50
_PROMOTION_WIN_RATE_THRESHOLD = 0.55
_PROMOTION_ELO_DELTA_THRESHOLD = 25.0

try:  # Optional Parquet support
    import pyarrow as _pa
    import pyarrow.parquet as _pq
except ModuleNotFoundError:  # pragma: no cover - pyarrow not required for JSON output
    _pa = None
    _pq = None


@dataclass
class TrainingConfig:
    """Resolved configuration for a self-play run."""

    phase: str
    seed: int
    opponents: list[str]
    total_steps: int
    eval_frequency: int
    artifact_root: Path
    run_id: str | None
    resume_from: Path | None
    notes: str | None
    rollout_steps: int
    reservoir_size: int
    eval_games: int
    eval_seed: int
    promotion_min_games: int
    promotion_min_win_rate: float
    promotion_min_elo_delta: float
    gamma: float
    gae_lambda: float
    margin_weight: float
    jitter_scale: float
    learning_rate: float
    ppo_epochs: int
    ppo_batch_size: int
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    device: str


@dataclass
class RunArtifacts:
    """Filesystem layout prepared for the run."""

    root: Path
    checkpoints: Path
    metrics: Path
    rollouts: Path
    eval: Path
    manifest: Path
    champion_manifest: Path


class RollingMetricAggregator:
    """Maintain rolling statistics and persist them for dashboards."""

    _IGNORED_KEYS = {"timestamp", "iteration", "steps", "episodes"}

    def __init__(
        self,
        *,
        directory: Path,
        window_size: int = _ROLLING_METRIC_WINDOW,
        json_name: str = "rolling_metrics.json",
        parquet_name: str = "rolling_metrics.parquet",
    ) -> None:
        self._directory = directory
        self._window_size = max(1, int(window_size))
        self._records: deque[dict[str, Any]] = deque(maxlen=self._window_size)
        self._history: list[dict[str, Any]] = []
        self._metric_keys: list[str] = []
        self._json_path = self._directory / json_name
        self._parquet_path = self._directory / parquet_name
        self._parquet_supported = _pa is not None and _pq is not None
        self._bootstrapped = False

    def bootstrap(self) -> None:
        """Seed the aggregator from existing step metrics."""

        if self._bootstrapped:
            return
        step_files = sorted(self._directory.glob("step_*.json"))
        for path in step_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupted metrics
                continue
            self.record(payload, persist=False)
        if self._history:
            self._write_outputs()
        else:
            self._remove_outputs()
        self._bootstrapped = True

    def record(self, payload: Mapping[str, Any], *, persist: bool = True) -> dict[str, Any]:
        entry = dict(payload)
        aggregated = self._ingest(entry)
        if persist:
            self._write_outputs()
        return aggregated

    def _ingest(self, entry: dict[str, Any]) -> dict[str, Any]:
        self._records.append(entry)
        self._update_metric_keys(entry)
        aggregated = self._aggregate(entry)
        self._history.append(aggregated)
        return aggregated

    def _update_metric_keys(self, entry: Mapping[str, Any]) -> None:
        for key, value in entry.items():
            if key in self._IGNORED_KEYS:
                continue
            if isinstance(value, numbers.Real) and key not in self._metric_keys:
                self._metric_keys.append(key)

    def _aggregate(self, latest: Mapping[str, Any]) -> dict[str, Any]:
        window_entries = list(self._records)
        aggregated: dict[str, Any] = {
            "timestamp": latest.get("timestamp"),
            "iteration": int(latest.get("iteration", 0)),
            "steps": int(latest.get("steps", 0)),
            "episodes": int(latest.get("episodes", 0)),
            "window": len(window_entries),
        }
        for key in self._metric_keys:
            values: list[float] = []
            for record in window_entries:
                value = record.get(key)
                if isinstance(value, numbers.Real):
                    values.append(float(value))
            if not values:
                continue
            aggregated[f"{key}_mean"] = statistics.fmean(values)
            aggregated[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)
        return aggregated

    def _write_outputs(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        self._json_path.write_text(json.dumps(self._history, indent=2) + "\n", encoding="utf-8")
        if self._parquet_supported and self._history:
            table = _pa.Table.from_pylist(self._history)
            _pq.write_table(table, self._parquet_path)

    def _remove_outputs(self) -> None:
        try:
            self._json_path.unlink(missing_ok=True)
        except TypeError:  # pragma: no cover - Python < 3.8 fallback
            if self._json_path.exists():
                self._json_path.unlink()
        if self._parquet_supported:
            try:
                self._parquet_path.unlink(missing_ok=True)
            except TypeError:  # pragma: no cover - Python < 3.8 fallback
                if self._parquet_path.exists():
                    self._parquet_path.unlink()


@dataclass
class EvaluationSummary:
    path: Path
    iteration: int
    step: int
    final_elo: float
    win_rate: float
    games: int
    elo_delta: float
    payload: dict[str, Any]

    @property
    def opponents(self) -> Sequence[Mapping[str, Any]]:
        return list(self.payload.get("opponents", []))

    @property
    def timestamp(self) -> str | None:
        return self.payload.get("timestamp")


@dataclass
class ChampionRecord:
    checkpoint: Path
    win_rate: float
    final_elo: float
    elo_delta: float
    games: int
    iteration: int
    step: int
    promoted_at: str

    def relative_checkpoint(self, root: Path) -> str:
        try:
            return str(self.checkpoint.relative_to(root))
        except ValueError:
            return str(self.checkpoint)


def _relative_path(target: Path, base: Path) -> str:
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(target.resolve())


def _load_champion_record(path: Path) -> ChampionRecord | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupted champion manifest
        return None

    checkpoint_str = payload.get("checkpoint")
    if not checkpoint_str:
        return None
    checkpoint_path = Path(checkpoint_str)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (path.parent / checkpoint_path).resolve()

    evaluation = payload.get("evaluation", {})
    win_rate = float(evaluation.get("winRate", 0.0))
    final_elo = float(evaluation.get("finalElo", _EVAL_ELO_BASE))
    games = int(evaluation.get("games", 0))
    iteration = int(evaluation.get("iteration", 0))
    step = int(evaluation.get("step", 0))
    elo_delta = float(evaluation.get("eloDelta", final_elo - _EVAL_ELO_BASE))
    promoted_at = str(payload.get("promotedAt", "")) or str(evaluation.get("timestamp", ""))

    return ChampionRecord(
        checkpoint=checkpoint_path,
        win_rate=win_rate,
        final_elo=final_elo,
        elo_delta=elo_delta,
        games=games,
        iteration=iteration,
        step=step,
        promoted_at=promoted_at,
    )


def _maybe_promote_checkpoint(
    *,
    checkpoint: Path,
    summary: EvaluationSummary,
    config: TrainingConfig,
    artifacts: RunArtifacts,
    champion: ChampionRecord | None,
) -> tuple[bool, ChampionRecord | None]:
    min_games = max(0, config.promotion_min_games)
    if summary.games < min_games:
        return False, champion
    if summary.win_rate < config.promotion_min_win_rate:
        return False, champion
    if summary.elo_delta < config.promotion_min_elo_delta:
        return False, champion

    resolved_checkpoint = checkpoint.resolve()
    if champion is not None:
        if resolved_checkpoint == champion.checkpoint:
            return False, champion
        if summary.final_elo <= champion.final_elo and summary.win_rate <= champion.win_rate:
            return False, champion

    promoted_at = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
    new_champion = ChampionRecord(
        checkpoint=resolved_checkpoint,
        win_rate=summary.win_rate,
        final_elo=summary.final_elo,
        elo_delta=summary.elo_delta,
        games=summary.games,
        iteration=summary.iteration,
        step=summary.step,
        promoted_at=promoted_at,
    )

    _persist_champion_manifest(
        champion_record=new_champion,
        summary=summary,
        artifacts=artifacts,
        config=config,
    )
    _update_run_manifest_champion(
        manifest_path=artifacts.manifest,
        champion=new_champion,
        summary=summary,
        config=config,
        champion_manifest=artifacts.champion_manifest,
    )
    return True, new_champion


def _persist_champion_manifest(
    *,
    champion_record: ChampionRecord,
    summary: EvaluationSummary,
    artifacts: RunArtifacts,
    config: TrainingConfig,
) -> None:
    run_root = artifacts.root
    summary_payload = summary.payload.copy()
    summary_payload["winRate"] = summary.win_rate
    summary_payload["games"] = summary.games
    summary_payload["finalElo"] = summary.final_elo
    summary_payload["eloDelta"] = summary.elo_delta
    summary_payload["iteration"] = summary.iteration
    summary_payload["step"] = summary.step
    summary_payload["summaryPath"] = _relative_path(summary.path, run_root)

    payload: dict[str, Any] = {
        "model": "_03_training.policy_loader:load_policy",
        "checkpoint": champion_record.relative_checkpoint(run_root),
        "factoryKwargs": {"device": "cpu"},
        "runId": config.run_id or "unknown",
        "promotedAt": champion_record.promoted_at,
        "evaluation": summary_payload,
        "promotionCriteria": {
            "minGames": config.promotion_min_games,
            "minWinRate": config.promotion_min_win_rate,
            "minEloDelta": config.promotion_min_elo_delta,
        },
        "opponents": config.opponents,
    }

    artifacts.champion_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _update_run_manifest_champion(
    *,
    manifest_path: Path,
    champion: ChampionRecord,
    summary: EvaluationSummary,
    config: TrainingConfig,
    champion_manifest: Path,
) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - manifest corruption
        return

    champion_entry = {
        "checkpoint": champion.relative_checkpoint(manifest_path.parent),
        "step": champion.step,
        "iteration": champion.iteration,
        "promotedAt": champion.promoted_at,
        "winRate": champion.win_rate,
        "games": champion.games,
        "finalElo": champion.final_elo,
        "eloDelta": champion.elo_delta,
        "manifest": _relative_path(champion_manifest, manifest_path.parent),
        "evaluation": {
            "path": _relative_path(summary.path, manifest_path.parent),
            "timestamp": summary.timestamp,
        },
        "criteria": {
            "minGames": config.promotion_min_games,
            "minWinRate": config.promotion_min_win_rate,
            "minEloDelta": config.promotion_min_elo_delta,
        },
        "opponents": config.opponents,
    }
    manifest["champion"] = champion_entry
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
@dataclass
class SelfPlayRunContext:
    """Summary of the seeded run."""

    config: TrainingConfig
    run_id: str
    git_sha: str
    created_at: str
    artifacts: RunArtifacts
    completed_steps: int
    episodes: int
    last_checkpoint: Path | None
    last_evaluation: Path | None


def main(argv: Sequence[str] | None = None) -> SelfPlayRunContext:
    args = _parse_args(argv)
    base_config = _load_config(args.config)
    merged_config = _merge_config(base_config, args)

    git_sha = _git_sha()
    run_id = merged_config.run_id or _default_run_id(
        phase=merged_config.phase,
        seed=merged_config.seed,
        git_sha=git_sha,
    )
    merged_config.run_id = run_id

    created_at = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()

    _seed_everything(merged_config.seed)
    artifacts = _prepare_artifacts(merged_config.artifact_root, run_id)
    _write_manifest(
        artifacts.manifest,
        config=merged_config,
        git_sha=git_sha,
        created_at=created_at,
        config_path=args.config,
        run_id=run_id,
    )

    _print_summary(run_id, merged_config, artifacts, git_sha)

    completed_steps, episodes, last_checkpoint, last_evaluation = _run_training(merged_config, artifacts)

    return SelfPlayRunContext(
        config=merged_config,
        run_id=run_id,
        git_sha=git_sha,
        created_at=created_at,
        artifacts=artifacts,
        completed_steps=completed_steps,
        episodes=episodes,
        last_checkpoint=last_checkpoint,
        last_evaluation=last_evaluation,
    )


@dataclass
class CheckpointEntry:
    """Metadata for a checkpoint stored in the opponent reservoir."""

    path: Path
    step: int
    factory: rollout.AgentFactory


class CheckpointReservoir:
    """Maintain a bounded set of checkpoint-based opponent factories."""

    def __init__(self, *, max_size: int, device: torch.device) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._device = device
        self._entries: list[CheckpointEntry] = []
        self._known_paths: set[Path] = set()

    def bootstrap_from_directory(self, directory: Path) -> None:
        """Load checkpoints from ``directory`` in sorted order."""

        if not directory.exists():
            return
        for checkpoint_path in sorted(directory.glob("*.pt")):
            self.add_checkpoint(checkpoint_path)

    def add_checkpoint(self, path: Path) -> None:
        """Load ``path`` and register it as a reservoir opponent."""

        resolved = path.resolve()
        if resolved in self._known_paths:
            return
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")

        model, _, payload = models.load_checkpoint(
            resolved,
            device=self._device,
            include_optimizer=False,
        )
        model.eval()
        factory = _checkpoint_agent_factory(model, self._device)
        step = int(payload.get("step", 0))

        entry = CheckpointEntry(path=resolved, step=step, factory=factory)
        self._entries.append(entry)
        self._known_paths.add(resolved)

        if len(self._entries) > self._max_size:
            removed = self._entries.pop(0)
            self._known_paths.discard(removed.path)

    @property
    def factories(self) -> list[rollout.AgentFactory]:
        return [entry.factory for entry in self._entries]

    @property
    def paths(self) -> list[Path]:
        return [entry.path for entry in self._entries]

    def __len__(self) -> int:
        return len(self._entries)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-play PPO training")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a JSON config file",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Phase tag used in generated run ids (default: p3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global seed for reproducibility (default: 2025)",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        help="Opponent name to include in the evaluation pool; repeat for multiple",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total learner decisions to train on (default: 1,000,000)",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=None,
        help="Save checkpoints every N learner steps (default: 50,000)",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Root directory for artifacts (default: _03_training/artifacts)",
    )
    parser.add_argument("--run-id", help="Override auto-generated run id")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional checkpoint file/directory to resume from",
    )
    parser.add_argument("--notes", help="Free-form notes stored in the manifest")
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Minimum learner decisions per rollout batch (default: 2048)",
    )
    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=None,
        help="Maximum checkpoints to retain as self-play opponents (default: 3)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=None,
        help="Games per opponent when evaluating checkpoints (default: 200)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Base seed used for evaluation tournaments (default: 4096)",
    )
    parser.add_argument(
        "--promotion-min-games",
        type=int,
        default=None,
        help="Minimum total games before a checkpoint can be promoted (default: opponents * eval_games)",
    )
    parser.add_argument(
        "--promotion-min-win-rate",
        type=float,
        default=None,
        help="Average win rate threshold for champion promotion (default: 0.55)",
    )
    parser.add_argument(
        "--promotion-min-elo-delta",
        type=float,
        default=None,
        help="Minimum Elo delta above baseline required for promotion (default: 25.0)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Adam learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=None,
        help="PPO epochs per rollout (default: 4)",
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=None,
        help="PPO minibatch size (default: 512)",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=None,
        help="PPO clipping coefficient (default: 0.2)",
    )
    parser.add_argument(
        "--value-coef",
        type=float,
        default=None,
        help="Value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Entropy bonus coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Gradient clipping norm (default: 0.5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help="GAE lambda (default: 0.95)",
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=None,
        help="Weight applied to margin reward shaping (default: 0.25)",
    )
    parser.add_argument(
        "--jitter-scale",
        type=float,
        default=None,
        help="Magnitude of deterministic reward jitter (default: 0.01)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run on (default: auto)",
    )
    return parser.parse_args(argv)


def _load_config(config_path: Path | None) -> Mapping[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    try:
        payload = config_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem failure
        raise SystemExit(f"Failed to read config file: {exc}") from exc
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON config: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SystemExit("Config file must contain a JSON object")
    return data


def _config_lookup(config: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in config:
            return config[key]
    return None


def _merge_config(base: Mapping[str, Any], args: argparse.Namespace) -> TrainingConfig:
    if args.opponent:
        opponents = [str(name) for name in args.opponent]
    else:
        base_opponents = _config_lookup(base, "opponents", "opponent_pool")
        if isinstance(base_opponents, Sequence) and not isinstance(base_opponents, (str, bytes)):
            opponents = [str(name) for name in base_opponents]
        else:
            opponents = list(_DEFAULT_OPPONENT_POOL)

    phase = str(args.phase if args.phase is not None else _config_lookup(base, "phase", "Phase") or "p3")

    seed_value = args.seed if args.seed is not None else _config_lookup(base, "seed", "Seed")
    seed = int(seed_value) if seed_value is not None else 2025

    total_steps_value = args.total_steps if args.total_steps is not None else _config_lookup(base, "total_steps", "totalSteps")
    total_steps = int(total_steps_value) if total_steps_value is not None else 1_000_000

    eval_freq_value = args.eval_frequency if args.eval_frequency is not None else _config_lookup(base, "eval_frequency", "evalFrequency")
    eval_frequency = int(eval_freq_value) if eval_freq_value is not None else 50_000

    artifact_root_value = args.artifact_root if args.artifact_root is not None else _config_lookup(base, "artifact_root", "artifactRoot")
    artifact_root = Path(artifact_root_value) if artifact_root_value is not None else _DEFAULT_ARTIFACT_ROOT

    run_id = args.run_id or _config_lookup(base, "run_id", "runId")

    resume_value = args.resume_from if args.resume_from is not None else _config_lookup(base, "resume_from", "resumeFrom")
    resume_from = Path(resume_value) if resume_value is not None else None

    notes_value = args.notes if args.notes is not None else _config_lookup(base, "notes")
    notes = str(notes_value) if notes_value is not None else None

    rollout_steps_value = args.rollout_steps if args.rollout_steps is not None else _config_lookup(base, "rollout_steps", "rolloutSteps")
    rollout_steps = int(rollout_steps_value) if rollout_steps_value is not None else 2048

    reservoir_size_value = args.reservoir_size if args.reservoir_size is not None else _config_lookup(base, "reservoir_size", "reservoirSize")
    reservoir_size = int(reservoir_size_value) if reservoir_size_value is not None else 3

    eval_games_value = args.eval_games if args.eval_games is not None else _config_lookup(base, "eval_games", "evalGames")
    eval_games = int(eval_games_value) if eval_games_value is not None else _DEFAULT_EVAL_GAMES

    eval_seed_value = args.eval_seed if args.eval_seed is not None else _config_lookup(base, "eval_seed", "evalSeed")
    eval_seed = int(eval_seed_value) if eval_seed_value is not None else _DEFAULT_EVAL_SEED

    promotion_games_value = args.promotion_min_games if args.promotion_min_games is not None else _config_lookup(base, "promotion_min_games", "promotionMinGames")
    if promotion_games_value is not None:
        promotion_min_games = max(0, int(promotion_games_value))
    else:
        promotion_min_games = max(0, len(opponents) * max(eval_games, 0))

    promotion_win_rate_value = (
        args.promotion_min_win_rate if args.promotion_min_win_rate is not None else _config_lookup(base, "promotion_min_win_rate", "promotionMinWinRate")
    )
    promotion_min_win_rate = (
        float(promotion_win_rate_value) if promotion_win_rate_value is not None else _PROMOTION_WIN_RATE_THRESHOLD
    )

    promotion_elo_delta_value = (
        args.promotion_min_elo_delta if args.promotion_min_elo_delta is not None else _config_lookup(base, "promotion_min_elo_delta", "promotionMinEloDelta")
    )
    promotion_min_elo_delta = (
        float(promotion_elo_delta_value) if promotion_elo_delta_value is not None else _PROMOTION_ELO_DELTA_THRESHOLD
    )

    learning_rate_value = args.learning_rate if args.learning_rate is not None else _config_lookup(base, "learning_rate", "learningRate")
    learning_rate = float(learning_rate_value) if learning_rate_value is not None else 3e-4

    ppo_epochs_value = args.ppo_epochs if args.ppo_epochs is not None else _config_lookup(base, "ppo_epochs", "ppoEpochs")
    ppo_epochs = int(ppo_epochs_value) if ppo_epochs_value is not None else 4

    ppo_batch_value = args.ppo_batch_size if args.ppo_batch_size is not None else _config_lookup(base, "ppo_batch_size", "ppoBatchSize")
    ppo_batch_size = int(ppo_batch_value) if ppo_batch_value is not None else 512

    clip_value = args.clip_coef if args.clip_coef is not None else _config_lookup(base, "clip_coef", "clipCoef")
    clip_coef = float(clip_value) if clip_value is not None else 0.2

    value_coef_value = args.value_coef if args.value_coef is not None else _config_lookup(base, "value_coef", "valueCoef")
    value_coef = float(value_coef_value) if value_coef_value is not None else 0.5

    entropy_coef_value = args.entropy_coef if args.entropy_coef is not None else _config_lookup(base, "entropy_coef", "entropyCoef")
    entropy_coef = float(entropy_coef_value) if entropy_coef_value is not None else 0.01

    grad_norm_value = args.max_grad_norm if args.max_grad_norm is not None else _config_lookup(base, "max_grad_norm", "maxGradNorm")
    max_grad_norm = float(grad_norm_value) if grad_norm_value is not None else 0.5

    gamma_value = args.gamma if args.gamma is not None else _config_lookup(base, "gamma")
    gamma = float(gamma_value) if gamma_value is not None else 0.99

    gae_lambda_value = args.gae_lambda if args.gae_lambda is not None else _config_lookup(base, "gae_lambda", "gaeLambda")
    gae_lambda = float(gae_lambda_value) if gae_lambda_value is not None else 0.95

    margin_value = args.margin_weight if args.margin_weight is not None else _config_lookup(base, "margin_weight", "marginWeight")
    margin_weight = float(margin_value) if margin_value is not None else 0.25

    jitter_value = args.jitter_scale if args.jitter_scale is not None else _config_lookup(base, "jitter_scale", "jitterScale")
    jitter_scale = float(jitter_value) if jitter_value is not None else 0.01

    device_value = args.device if args.device is not None else _config_lookup(base, "device")
    device = str(device_value) if device_value is not None else "auto"

    return TrainingConfig(
        phase=phase,
        seed=seed,
        opponents=opponents,
        total_steps=total_steps,
        eval_frequency=eval_frequency,
        artifact_root=artifact_root,
        run_id=run_id if run_id is None else str(run_id),
        resume_from=resume_from,
        notes=notes,
        rollout_steps=rollout_steps,
        reservoir_size=reservoir_size,
        eval_games=eval_games,
        eval_seed=eval_seed,
        promotion_min_games=promotion_min_games,
        promotion_min_win_rate=promotion_min_win_rate,
        promotion_min_elo_delta=promotion_min_elo_delta,
        gamma=gamma,
        gae_lambda=gae_lambda,
        margin_weight=margin_weight,
        jitter_scale=jitter_scale,
        learning_rate=learning_rate,
        ppo_epochs=ppo_epochs,
        ppo_batch_size=ppo_batch_size,
        clip_coef=clip_coef,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        device=device,
    )


def _run_training(config: TrainingConfig, artifacts: RunArtifacts) -> tuple[int, int, Path | None, Path | None]:
    device = _resolve_device(config.device)
    print(f"Using device: {device}")

    observation_size = encoders.observation_size()
    action_size = len(action_space.canonical_actions())
    policy_config = models.PolicyConfig(
        observation_size=observation_size,
        action_size=action_size,
    )

    global_steps = 0
    episodes = 0
    last_checkpoint = None
    last_evaluation: Path | None = None

    if config.resume_from is not None:
        checkpoint_path = _resolve_resume_checkpoint(config.resume_from)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model, optimizer_state, payload = models.load_checkpoint(
            checkpoint_path,
            device=device,
            include_optimizer=True,
        )
        if (
            model.config.observation_size != observation_size
            or model.config.action_size != action_size
        ):
            raise SystemExit(
                "Checkpoint architecture mismatch; ensure observation/action schemas align",
            )
        global_steps = int(payload.get("step", 0))
        metadata = payload.get("metadata", {}) or {}
        episodes = int(metadata.get("episodes", episodes))
        last_checkpoint = checkpoint_path
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    else:
        model = models.PolicyValueNet(policy_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    baseline_factories = _build_opponent_factories(config.opponents, config.seed)
    reservoir_device = _reservoir_device(device)
    reservoir = CheckpointReservoir(
        max_size=config.reservoir_size,
        device=reservoir_device,
    )
    reservoir.bootstrap_from_directory(artifacts.checkpoints)
    if last_checkpoint is not None:
        reservoir.add_checkpoint(last_checkpoint)
    rollout_config = rollout.RolloutConfig(
        min_steps=config.rollout_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        margin_weight=config.margin_weight,
        jitter_scale=config.jitter_scale,
    )
    ppo_config = ppo.PPOConfig(
        learning_rate=config.learning_rate,
        epochs=config.ppo_epochs,
        batch_size=config.ppo_batch_size,
        clip_coef=config.clip_coef,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
    )

    metrics_aggregator = RollingMetricAggregator(directory=artifacts.metrics, window_size=_ROLLING_METRIC_WINDOW)
    metrics_aggregator.bootstrap()
    champion = _load_champion_record(artifacts.champion_manifest)
    if champion is not None:
        print(
            "Loaded champion: "
            f"{champion.relative_checkpoint(artifacts.root)} "
            f"(elo={champion.final_elo:.1f}, win_rate={champion.win_rate:.3f})"
        )

    iteration = 0
    last_checkpoint_step = global_steps

    while global_steps < config.total_steps:
        iteration += 1
        model.eval()
        active_factories = list(baseline_factories) + reservoir.factories
        if not active_factories:
            raise SystemExit("No opponents available for rollouts; configure opponents or reservoir size > 0")
        batch = rollout.collect_rollouts(
            model=model,
            opponent_factories=active_factories,
            config=rollout_config,
            base_seed=config.seed + iteration,
            device=device,
        )
        if batch.steps == 0:
            print("Rollout produced zero learner decisions; retrying...")
            continue

        model.train()
        metrics = ppo.ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            config=ppo_config,
            device=device,
        )

        global_steps += batch.steps
        episodes += batch.episodes

        avg_reward = sum(batch.episode_rewards) / max(len(batch.episode_rewards), 1)
        mean_entropy = batch.entropies.mean().item()

        metrics_payload = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "iteration": iteration,
            "steps": global_steps,
            "episodes": episodes,
            "rolloutSteps": batch.steps,
            "batchEpisodes": batch.episodes,
            "avgEpisodeReward": avg_reward,
            "meanEntropy": mean_entropy,
            **metrics,
        }
        metrics_path = artifacts.metrics / f"step_{global_steps}.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
        metrics_aggregator.record(metrics_payload)

        print(
            f"[iter {iteration:03d}] steps={global_steps} episodes={episodes} "
            f"avg_reward={avg_reward:.3f} policy_loss={metrics['policy_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} entropy={metrics['entropy']:.4f}"
        )

        if global_steps - last_checkpoint_step >= config.eval_frequency or global_steps >= config.total_steps:
            checkpoint_path = artifacts.checkpoints / f"step_{global_steps}.pt"
            models.save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=global_steps,
                metadata={
                    "iteration": iteration,
                    "episodes": episodes,
                    "avgEpisodeReward": avg_reward,
                },
            )
            last_checkpoint = checkpoint_path
            last_checkpoint_step = global_steps
            print(f"Saved checkpoint: {checkpoint_path}")
            try:
                reservoir.add_checkpoint(checkpoint_path)
                print(f"Reservoir size={len(reservoir)} (added {checkpoint_path.name})")
            except FileNotFoundError:
                print(f"Warning: checkpoint missing for reservoir registration: {checkpoint_path}")

            evaluation = _run_evaluation_suite(
                checkpoint=checkpoint_path,
                config=config,
                artifacts=artifacts,
                iteration=iteration,
                step=global_steps,
                eval_device=reservoir_device,
            )
            if evaluation is not None:
                last_evaluation = evaluation.path
                print(
                    "Evaluation summary saved to "
                    f"{evaluation.path} (win_rate={evaluation.win_rate:.3f}, elo={evaluation.final_elo:.1f})"
                )
                promoted, champion = _maybe_promote_checkpoint(
                    checkpoint=checkpoint_path,
                    summary=evaluation,
                    config=config,
                    artifacts=artifacts,
                    champion=champion,
                )
                if promoted:
                    print(
                        "Promoted new champion: "
                        f"{checkpoint_path.name} "
                        f"(elo={evaluation.final_elo:.1f}, win_rate={evaluation.win_rate:.3f})"
                    )

    return global_steps, episodes, last_checkpoint, last_evaluation


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - git may be missing
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _default_run_id(*, phase: str, seed: int, git_sha: str) -> str:
    date_part = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    sha_part = (git_sha or "unknown")[:7] or "unknown"
    return f"{phase}-sp-{date_part}-{sha_part}-{seed}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:  # Optional numpy seeding
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:  # pragma: no cover - numpy optional
        pass

    try:  # Optional torch seeding
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch optional
        pass


def _prepare_artifacts(root: Path, run_id: str) -> RunArtifacts:
    run_root = root / run_id
    checkpoints = run_root / "checkpoints"
    metrics = run_root / "metrics"
    rollouts = run_root / "rollouts"
    eval_dir = run_root / "eval"
    manifest = run_root / "run_manifest.json"
    champion_manifest = run_root / "champion.json"

    for path in (run_root, checkpoints, metrics, rollouts, eval_dir):
        path.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        root=run_root,
        checkpoints=checkpoints,
        metrics=metrics,
        rollouts=rollouts,
        eval=eval_dir,
        manifest=manifest,
        champion_manifest=champion_manifest,
    )


def _write_manifest(
    manifest_path: Path,
    *,
    config: TrainingConfig,
    git_sha: str,
    created_at: str,
    config_path: Path | None,
    run_id: str,
) -> None:
    payload: dict[str, Any] = {
        "runId": run_id,
        "phase": config.phase,
        "seed": config.seed,
        "gitSha": git_sha,
        "createdAt": created_at,
        "totalSteps": config.total_steps,
        "evalFrequency": config.eval_frequency,
        "opponents": config.opponents,
        "artifactRoot": str(config.artifact_root),
        "hyperparameters": {
            "rolloutSteps": config.rollout_steps,
            "reservoirSize": config.reservoir_size,
            "evalGames": config.eval_games,
            "evalSeed": config.eval_seed,
            "gamma": config.gamma,
            "gaeLambda": config.gae_lambda,
            "marginWeight": config.margin_weight,
            "jitterScale": config.jitter_scale,
            "learningRate": config.learning_rate,
            "ppoEpochs": config.ppo_epochs,
            "ppoBatchSize": config.ppo_batch_size,
            "clipCoef": config.clip_coef,
            "valueCoef": config.value_coef,
            "entropyCoef": config.entropy_coef,
            "maxGradNorm": config.max_grad_norm,
            "device": config.device,
        },
        "promotionCriteria": {
            "minGames": config.promotion_min_games,
            "minWinRate": config.promotion_min_win_rate,
            "minEloDelta": config.promotion_min_elo_delta,
        },
    }
    if config.resume_from is not None:
        payload["resumeFrom"] = str(config.resume_from)
    if config.notes is not None:
        payload["notes"] = config.notes
    if config_path is not None:
        payload["sourceConfig"] = str(config_path)
    payload["manifestVersion"] = 1

    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _print_summary(run_id: str, config: TrainingConfig, artifacts: RunArtifacts, git_sha: str) -> None:
    print("=== Self-play run configuration ===")
    print(f"run_id      : {run_id}")
    print(f"git_sha     : {git_sha}")
    print(f"seed        : {config.seed}")
    print(f"phase       : {config.phase}")
    print(f"opponents   : {', '.join(config.opponents)}")
    print(f"total_steps : {config.total_steps}")
    print(f"eval_freq   : {config.eval_frequency}")
    print(f"rollout     : {config.rollout_steps}")
    print(f"reservoir   : {config.reservoir_size}")
    print(f"eval_games  : {config.eval_games}")
    print(f"device      : {config.device}")
    print(f"artifacts   : {artifacts.root}")
    if config.resume_from is not None:
        print(f"resume_from : {config.resume_from}")
    if config.notes:
        print(f"notes       : {config.notes}")
    print(f"manifest    : {artifacts.manifest}")


def _run_evaluation_suite(
    *,
    checkpoint: Path,
    config: TrainingConfig,
    artifacts: RunArtifacts,
    iteration: int,
    step: int,
    eval_device: torch.device,
) -> EvaluationSummary | None:
    if not config.opponents:
        return None
    if config.eval_games <= 0:
        return None

    eval_dir = artifacts.eval
    eval_dir.mkdir(parents=True, exist_ok=True)

    candidate_rating = _EVAL_ELO_BASE
    opponent_ratings = {name: float(_EVAL_ELO_BASE) for name in config.opponents}

    results: list[dict[str, Any]] = []
    base_seed = config.eval_seed + iteration * 10_000
    eval_device_str = str(eval_device)

    for index, name in enumerate(config.opponents):
        series_seed = base_seed + index * max(config.eval_games, 1)
        opponent_seed = base_seed + 97 * (index + 1)
        candidate_agent = _policy_agent_from_checkpoint(checkpoint, eval_device_str)
        opponent_agent = _build_evaluation_opponent(name, opponent_seed)

        series = tournament.SeriesConfig(
            games=config.eval_games,
            seed=series_seed,
            agent_a=candidate_agent,
            agent_b=opponent_agent,
            alternate_start=True,
            collect_actions=False,
        )
        result = tournament.play_series(series)
        summary = result.summary
        wins_a, wins_b = summary.wins
        ties = summary.ties
        rating_before = candidate_rating
        candidate_rating, opponent_rating = _elo_series_update(
            candidate_rating,
            opponent_ratings[name],
            wins_a=wins_a,
            wins_b=wins_b,
            ties=ties,
            k_factor=_EVAL_ELO_K,
        )
        opponent_ratings[name] = opponent_rating
        total_games = summary.games
        win_rate = wins_a / total_games if total_games else 0.0

        results.append(
            {
                "opponent": name,
                "games": total_games,
                "wins": wins_a,
                "losses": wins_b,
                "ties": ties,
                "winRate": win_rate,
                "averageScore": summary.average_scores[0],
                "averageOpponentScore": summary.average_scores[1],
                "averageTurns": summary.average_turns,
                "elo": {
                    "before": rating_before,
                    "after": candidate_rating,
                    "delta": candidate_rating - rating_before,
                    "opponentAfter": opponent_rating,
                },
            }
        )

    payload = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "step": step,
        "iteration": iteration,
        "checkpoint": str(checkpoint),
        "gamesPerOpponent": config.eval_games,
        "seedBase": config.eval_seed,
        "opponents": results,
        "finalElo": candidate_rating,
    }

    eval_path = eval_dir / f"step_{step}_eval.json"
    eval_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    total_games = sum(entry.get("games", 0) for entry in results)
    total_wins = sum(entry.get("wins", 0) + 0.5 * entry.get("ties", 0) for entry in results)
    average_win_rate = total_wins / total_games if total_games else 0.0
    elo_delta = candidate_rating - _EVAL_ELO_BASE

    return EvaluationSummary(
        path=eval_path,
        iteration=iteration,
        step=step,
        final_elo=candidate_rating,
        win_rate=average_win_rate,
        games=total_games,
        elo_delta=elo_delta,
        payload=payload,
    )


def _policy_agent_from_checkpoint(checkpoint: Path, device: str) -> Agent:
    model_fn = policy_loader.load_policy(checkpoint=checkpoint, device=device)
    return SelfPlayRLAgent(model_factory=lambda: model_fn)


def _build_evaluation_opponent(name: str, seed: int) -> Agent:
    normalized = name.lower()
    if normalized == "first":
        return FirstLegalAgent()
    if normalized == "random":
        return RandomAgent(seed=seed)
    if normalized == "greedy":
        return GreedyAgent()
    if normalized == "diego":
        return DiegoAgent()
    raise SystemExit(f"Unsupported evaluation opponent: {name}")


def _elo_series_update(
    rating_a: float,
    rating_b: float,
    *,
    wins_a: int,
    wins_b: int,
    ties: int,
    k_factor: float,
) -> tuple[float, float]:
    def expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    for _ in range(wins_a):
        exp_a = expected(rating_a, rating_b)
        exp_b = 1.0 - exp_a
        rating_a += k_factor * (1.0 - exp_a)
        rating_b += k_factor * (0.0 - exp_b)

    for _ in range(ties):
        exp_a = expected(rating_a, rating_b)
        exp_b = 1.0 - exp_a
        rating_a += k_factor * (0.5 - exp_a)
        rating_b += k_factor * (0.5 - exp_b)

    for _ in range(wins_b):
        exp_a = expected(rating_a, rating_b)
        exp_b = 1.0 - exp_a
        rating_a += k_factor * (0.0 - exp_a)
        rating_b += k_factor * (1.0 - exp_b)

    return rating_a, rating_b


def _checkpoint_agent_factory(model: models.PolicyValueNet, device: torch.device) -> rollout.AgentFactory:
    """Create a factory that wraps ``model`` in a :class:`SelfPlayRLAgent`."""

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    def model_factory() -> Callable[[Any], Sequence[float]]:
        def logits_fn(obs: Any) -> Sequence[float]:
            encoded = encoders.encode_observation(obs).to(device)
            with torch.no_grad():
                logits, _ = model(encoded)
            return logits.squeeze(0).cpu().tolist()

        return logits_fn

    def factory() -> Agent:
        return SelfPlayRLAgent(model_factory=model_factory)

    return factory


def _reservoir_device(training_device: torch.device) -> torch.device:
    if training_device.type == "cpu":
        return training_device
    return torch.device("cpu")


def _build_opponent_factories(names: Sequence[str], seed: int) -> list[rollout.AgentFactory]:
    factories: list[rollout.AgentFactory] = []
    for name in names:
        normalized = name.lower()
        if normalized == "first":
            factories.append(FirstLegalAgent)
        elif normalized == "random":
            counter = itertools.count()

            def factory(counter=counter) -> Agent:
                agent_seed = seed + next(counter)
                return RandomAgent(seed=agent_seed)

            factories.append(factory)
        elif normalized == "greedy":
            factories.append(GreedyAgent)
        elif normalized == "diego":
            factories.append(DiegoAgent)
        else:
            raise SystemExit(f"Unsupported opponent: {name}")
    return factories


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA device but CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")
    return device


def _resolve_resume_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Resume path not found: {path}")
    candidates = sorted(path.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {path}")
    return candidates[-1]


if __name__ == "__main__":  # pragma: no cover - CLI support
    main()
