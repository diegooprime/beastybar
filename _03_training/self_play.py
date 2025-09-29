"""Self-play RL training entry point scaffolding.

This module seeds the environment, prepares artifact directories, and writes a
run manifest according to the Phase 3 design brief. The actual rollout and
optimization logic will be added in subsequent iterations.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_DEFAULT_ARTIFACT_ROOT = Path("_03_training/artifacts")
_DEFAULT_OPPONENT_POOL = ["first", "random", "greedy", "diego"]


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


@dataclass
class RunArtifacts:
    """Filesystem layout prepared for the run."""

    root: Path
    checkpoints: Path
    metrics: Path
    rollouts: Path
    eval: Path
    manifest: Path


@dataclass
class SelfPlayRunContext:
    """Summary of the seeded run."""

    config: TrainingConfig
    run_id: str
    git_sha: str
    created_at: str
    artifacts: RunArtifacts


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

    return SelfPlayRunContext(
        config=merged_config,
        run_id=run_id,
        git_sha=git_sha,
        created_at=created_at,
        artifacts=artifacts,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a self-play RL run")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a JSON config file",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Project phase tag for the run id (default: p3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global seed for the run (default: 2025)",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        help="Opponent name to include in the evaluation pool (repeatable)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total environment steps to budget for the run (default: 1,000,000)",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=None,
        help="How often to schedule evaluation tournaments in steps (default: 50,000)",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Root directory for artifacts (default: _03_training/artifacts)",
    )
    parser.add_argument(
        "--run-id",
        help="Override the auto-generated run identifier",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional checkpoint directory to resume from",
    )
    parser.add_argument(
        "--notes",
        help="Free-form notes stored alongside the manifest",
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

    base_phase = _config_lookup(base, "phase", "Phase")
    phase = str(args.phase if args.phase is not None else (base_phase or "p3"))

    base_seed = _config_lookup(base, "seed", "Seed")
    seed_value = args.seed if args.seed is not None else base_seed
    seed = int(seed_value) if seed_value is not None else 2025

    base_total_steps = _config_lookup(base, "total_steps", "totalSteps")
    total_steps_value = args.total_steps if args.total_steps is not None else base_total_steps
    total_steps = int(total_steps_value) if total_steps_value is not None else 1_000_000

    base_eval_freq = _config_lookup(base, "eval_frequency", "evalFrequency")
    eval_freq_value = args.eval_frequency if args.eval_frequency is not None else base_eval_freq
    eval_frequency = int(eval_freq_value) if eval_freq_value is not None else 50_000

    base_artifact = _config_lookup(base, "artifact_root", "artifactRoot")
    if args.artifact_root is not None:
        artifact_root = Path(args.artifact_root)
    elif base_artifact is not None:
        artifact_root = Path(base_artifact)
    else:
        artifact_root = _DEFAULT_ARTIFACT_ROOT

    base_run_id = _config_lookup(base, "run_id", "runId")
    run_id = args.run_id or base_run_id

    base_resume = _config_lookup(base, "resume_from", "resumeFrom")
    if args.resume_from is not None:
        resume_from = Path(args.resume_from)
    elif base_resume is not None:
        resume_from = Path(base_resume)
    else:
        resume_from = None

    notes_value = args.notes if args.notes is not None else _config_lookup(base, "notes")
    notes = str(notes_value) if notes_value is not None else None

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
    )


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
        import torch

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

    for path in (run_root, checkpoints, metrics, rollouts, eval_dir):
        path.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        root=run_root,
        checkpoints=checkpoints,
        metrics=metrics,
        rollouts=rollouts,
        eval=eval_dir,
        manifest=manifest,
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
    print("=== Self-play run prepared ===")
    print(f"run_id      : {run_id}")
    print(f"git_sha     : {git_sha}")
    print(f"seed        : {config.seed}")
    print(f"phase       : {config.phase}")
    print(f"opponents   : {', '.join(config.opponents)}")
    print(f"total_steps : {config.total_steps}")
    print(f"eval_freq   : {config.eval_frequency}")
    print(f"artifacts   : {artifacts.root}")
    if config.resume_from is not None:
        print(f"resume_from : {config.resume_from}")
    if config.notes:
        print(f"notes       : {config.notes}")
    print(f"manifest    : {artifacts.manifest}")


if __name__ == "__main__":  # pragma: no cover - CLI support
    main()
