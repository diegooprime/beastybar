"""Experiment tracking interface for training.

Provides a simple interface for logging metrics, hyperparameters, and artifacts.
Supports console output (default) and Weights & Biases for cloud tracking.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import wandb as wandb_module  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking backends."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics at a given training step.

        Args:
            metrics: Dictionary of metric names to values.
            step: Training step number.
        """
        ...

    @abstractmethod
    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameter names to values.
        """
        ...

    @abstractmethod
    def log_artifact(self, path: str, name: str) -> None:
        """Log an artifact (file) to the experiment.

        Args:
            path: Local filesystem path to the artifact.
            name: Name to use for the artifact in the tracking system.
        """
        ...

    @abstractmethod
    def finish(self) -> None:
        """Finish and close the experiment tracking session."""
        ...


class ConsoleTracker(ExperimentTracker):
    """Simple console-based tracker for debugging without external dependencies."""

    def __init__(
        self,
        project: str = "beastybar",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize console tracker.

        Args:
            project: Project name for display.
            run_name: Optional run name for display.
            config: Optional configuration dictionary.
        """
        self.project = project
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._last_step = -1
        self._metric_history: list[tuple[int, dict[str, float]]] = []

        logger.info(f"[ConsoleTracker] Starting experiment: {project}/{self.run_name}")
        if config:
            self.log_hyperparameters(config)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to console."""
        self._metric_history.append((step, metrics.copy()))
        formatted = " | ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        print(f"[Step {step:6d}] {formatted}", file=sys.stderr)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to console."""
        logger.info("[ConsoleTracker] Hyperparameters:")
        for key, value in sorted(params.items()):
            logger.info(f"  {key}: {value}")

    def log_artifact(self, path: str, name: str) -> None:
        """Log artifact path to console."""
        logger.info(f"[ConsoleTracker] Artifact '{name}': {path}")

    def finish(self) -> None:
        """Finish console tracking session."""
        logger.info(f"[ConsoleTracker] Finished experiment: {self.project}/{self.run_name}")
        logger.info(f"[ConsoleTracker] Total steps logged: {len(self._metric_history)}")


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker.

    Requires wandb package: pip install wandb
    """

    def __init__(
        self,
        project: str = "beastybar",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize W&B tracker.

        Args:
            project: W&B project name.
            run_name: Optional run name.
            config: Optional configuration dictionary.

        Raises:
            ImportError: If wandb is not installed.
        """
        try:
            import wandb  # type: ignore[import-not-found]

            self._wandb: wandb_module = wandb  # type: ignore[valid-type]
        except ImportError as e:
            raise ImportError("wandb is not installed. Install it with: pip install wandb") from e

        self._run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            reinit=True,
        )
        self.project = project
        self.run_name = self._run.name

        logger.info(f"[WandbTracker] Started run: {self._run.url}")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to W&B."""
        self._wandb.log(metrics, step=step)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to W&B config."""
        if self._run is not None:
            self._run.config.update(params, allow_val_change=True)

    def log_artifact(self, path: str, name: str) -> None:
        """Log artifact to W&B."""
        artifact = self._wandb.Artifact(name=name, type="model")
        artifact_path = Path(path)
        if artifact_path.is_dir():
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self._wandb.log_artifact(artifact)
        logger.info(f"[WandbTracker] Logged artifact: {name}")

    def finish(self) -> None:
        """Finish W&B run."""
        if self._run is not None:
            self._run.finish()
            logger.info("[WandbTracker] Run finished")


def create_tracker(
    backend: str = "console",
    project: str = "beastybar",
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> ExperimentTracker:
    """Factory function to create an experiment tracker.

    Args:
        backend: Backend type - "console" (default) or "wandb".
        project: Project name for organizing experiments.
        run_name: Optional run name. Auto-generated if not provided.
        config: Optional configuration dictionary to log.

    Returns:
        An ExperimentTracker instance.

    Raises:
        ValueError: If backend is not recognized.
        ImportError: If wandb backend is requested but not installed.

    Examples:
        >>> tracker = create_tracker()  # Console tracker (default)
        >>> tracker = create_tracker("wandb", project="my_project")
    """
    backend_name = backend.strip().lower()

    if backend_name == "console":
        return ConsoleTracker(project=project, run_name=run_name, config=config)
    elif backend_name == "wandb":
        return WandbTracker(project=project, run_name=run_name, config=config)
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Choose from: console, wandb")


def log_training_step(
    tracker: ExperimentTracker,
    step: int,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    learning_rate: float,
) -> None:
    """Log common training metrics.

    Args:
        tracker: The experiment tracker to use.
        step: Training step number.
        policy_loss: Policy loss value.
        value_loss: Value loss value.
        entropy: Policy entropy value.
        learning_rate: Current learning rate.
    """
    tracker.log_metrics(
        {
            "train/policy_loss": policy_loss,
            "train/value_loss": value_loss,
            "train/entropy": entropy,
            "train/learning_rate": learning_rate,
            "train/total_loss": policy_loss + 0.5 * value_loss - 0.01 * entropy,
        },
        step=step,
    )


def log_evaluation(
    tracker: ExperimentTracker,
    step: int,
    opponent: str,
    win_rate: float,
    avg_score: float,
    games_played: int,
) -> None:
    """Log evaluation metrics against an opponent.

    Args:
        tracker: The experiment tracker to use.
        step: Training step number.
        opponent: Name of the opponent evaluated against.
        win_rate: Win rate (0.0 to 1.0).
        avg_score: Average score difference.
        games_played: Number of games played.
    """
    tracker.log_metrics(
        {
            f"eval/{opponent}/win_rate": win_rate,
            f"eval/{opponent}/avg_score": avg_score,
            f"eval/{opponent}/games_played": float(games_played),
        },
        step=step,
    )


def log_self_play_stats(
    tracker: ExperimentTracker,
    step: int,
    games_generated: int,
    avg_game_length: float,
    action_entropy: float,
) -> None:
    """Log self-play game generation statistics.

    Args:
        tracker: The experiment tracker to use.
        step: Training step number.
        games_generated: Number of games generated.
        avg_game_length: Average number of turns per game.
        action_entropy: Average entropy of action distributions.
    """
    tracker.log_metrics(
        {
            "self_play/games_generated": float(games_generated),
            "self_play/avg_game_length": avg_game_length,
            "self_play/action_entropy": action_entropy,
        },
        step=step,
    )


__all__ = [
    "ConsoleTracker",
    "ExperimentTracker",
    "WandbTracker",
    "create_tracker",
    "log_evaluation",
    "log_self_play_stats",
    "log_training_step",
]
