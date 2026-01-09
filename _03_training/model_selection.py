"""Model selection utilities for neural network training.

This module provides:
- ModelTracker: Track and save best-performing models during training
- EarlyStopping: Stop training when no improvement is observed
- compare_checkpoints: Head-to-head comparison between checkpoints
- cleanup_old_checkpoints: Remove old checkpoints while keeping best and recent

Example:
    config = ModelSelectionConfig(
        metric="weighted",
        weights={"random": 0.7, "heuristic": 0.3},
        patience=20,
        regression_threshold=0.10,
    )
    tracker = ModelTracker(checkpoint_dir="checkpoints", config=config)
    early_stopping = EarlyStopping(config)

    for iteration in range(1000):
        # ... training ...
        eval_results = evaluate_agent(agent, eval_config)

        is_best = tracker.update(checkpoint_path, iteration, eval_results)
        should_stop, reason = early_stopping.update(eval_results, iteration)

        if should_stop:
            print(f"Early stopping: {reason}")
            break
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .evaluation import EvaluationResult

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ModelSelectionConfig:
    """Unified configuration for model tracking and early stopping.

    Attributes:
        metric: Selection metric type:
            - "mcts": Use MCTS opponent win rate (highest among MCTS variants)
            - "heuristic": Use heuristic opponent win rate
            - "weighted": Use weighted combination of opponent win rates
            - "custom": Use custom metric extractor function
        weights: Opponent name to weight mapping for "weighted" metric.
            Default: {"random": 0.7, "heuristic": 0.3}
        keep_top_k: Number of top models to maintain in tracker.
        patience: Number of evaluations without improvement before stopping.
        min_improvement: Minimum score improvement to count as progress.
        target_score: Optional target score; stop if achieved.
        regression_threshold: Stop if score drops more than this from best.
            Set to None to disable regression detection.
        stagnation_window: Window size for stagnation detection.
            Set to 0 to disable stagnation detection.
        min_evaluations: Minimum evaluations before early stopping can trigger.
        enabled: Whether early stopping is enabled.
    """

    # Metric selection
    metric: str = "weighted"
    weights: dict[str, float] | None = None
    target_opponent: str | None = None  # For single-opponent metrics

    # Model tracking
    keep_top_k: int = 5

    # Early stopping - basic
    patience: int = 20
    min_improvement: float = 0.02
    target_score: float | None = None
    min_evaluations: int = 5
    enabled: bool = True

    # Early stopping - advanced (optional)
    regression_threshold: float | None = 0.10
    stagnation_window: int = 10

    def __post_init__(self) -> None:
        """Set default weights if not provided."""
        if self.weights is None:
            self.weights = {"random": 0.7, "heuristic": 0.3}
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSelectionConfig:
        """Create config from dictionary."""
        return cls(**data)


# ============================================================================
# Model Record
# ============================================================================


@dataclass
class ModelRecord:
    """Record of a saved model checkpoint with evaluation metrics.

    Attributes:
        checkpoint_path: Absolute path to the checkpoint file.
        iteration: Training iteration when checkpoint was saved.
        score: Primary score used for ranking (based on config.metric).
        timestamp: Unix timestamp when record was created.
        metrics: Dictionary of all evaluation metrics.
    """

    checkpoint_path: str
    iteration: int
    score: float
    timestamp: float
    metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRecord:
        """Create record from dictionary."""
        return cls(**data)


# ============================================================================
# Metric Extraction
# ============================================================================


def extract_score(
    eval_results: list[EvaluationResult],
    config: ModelSelectionConfig,
) -> tuple[float, dict[str, float]]:
    """Extract primary score and all metrics from evaluation results.

    Args:
        eval_results: List of evaluation results.
        config: Model selection configuration.

    Returns:
        Tuple of (primary_score, all_metrics_dict).
    """
    metrics: dict[str, float] = {}

    # Extract all metrics
    for result in eval_results:
        metrics[f"{result.opponent_name}/win_rate"] = result.win_rate
        metrics[f"{result.opponent_name}/avg_margin"] = result.avg_point_margin

    # Compute primary score based on metric type
    if config.metric == "mcts":
        # Highest MCTS win rate
        score = 0.0
        for result in eval_results:
            if "mcts" in result.opponent_name.lower():
                score = max(score, result.win_rate)

    elif config.metric == "heuristic":
        # Heuristic win rate
        score = 0.0
        for result in eval_results:
            if "heuristic" in result.opponent_name.lower():
                score = result.win_rate
                break

    elif config.metric == "weighted":
        # Weighted combination
        score = 0.0
        for result in eval_results:
            name = result.opponent_name.lower()
            weight = config.weights.get(name, 0.0)
            score += weight * result.win_rate

    elif config.metric == "single" and config.target_opponent:
        # Single specified opponent
        target = config.target_opponent.lower()
        score = 0.0
        for result in eval_results:
            if result.opponent_name.lower() == target:
                score = result.win_rate
                break
            # Partial match for MCTS variants
            if "mcts" in target and "mcts" in result.opponent_name.lower():
                score = result.win_rate
                break

    else:
        # Default: use first result's win rate
        score = eval_results[0].win_rate if eval_results else 0.0

    return score, metrics


# ============================================================================
# Model Tracker
# ============================================================================


class ModelTracker:
    """Track and maintain the best-performing model checkpoints.

    Monitors evaluation results and saves models that achieve the highest
    score based on the configured metric. Maintains a leaderboard of top-k models.

    The tracker persists its state to disk, allowing training to resume
    with the correct best model information.

    Example:
        config = ModelSelectionConfig(metric="heuristic", keep_top_k=5)
        tracker = ModelTracker(
            checkpoint_dir="checkpoints/experiment1",
            config=config,
        )

        # After evaluation
        is_new_best = tracker.update(checkpoint_path, iteration, eval_results)
        if is_new_best:
            print(f"New best model! Score: {tracker.best_score:.1%}")
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config: ModelSelectionConfig | None = None,
        best_model_path: str = "best_model.pt",
    ) -> None:
        """Initialize the model tracker.

        Args:
            checkpoint_dir: Directory for storing checkpoints and tracker state.
            config: Model selection configuration. Uses defaults if None.
            best_model_path: Relative or absolute path for best model.
                If relative, resolved relative to checkpoint_dir.
        """
        self._config = config or ModelSelectionConfig()

        if self._config.keep_top_k < 1:
            raise ValueError(f"keep_top_k must be at least 1, got {self._config.keep_top_k}")

        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resolve best model path
        best_path = Path(best_model_path)
        if not best_path.is_absolute():
            best_path = self._checkpoint_dir / best_path
        self._best_model_path = best_path

        self._records: list[ModelRecord] = []
        self._score_history: list[float] = []
        self._state_file = self._checkpoint_dir / "model_tracker_state.json"

        # Load existing state if available
        self._load_state()

    @property
    def config(self) -> ModelSelectionConfig:
        """The model selection configuration."""
        return self._config

    @property
    def best_model_path(self) -> str | None:
        """Path to the current best model, or None if no models tracked."""
        if not self._records:
            return None
        return str(self._best_model_path)

    @property
    def best_score(self) -> float:
        """Score of the current best model, or 0.0 if no models tracked."""
        if not self._records:
            return 0.0
        return self._records[0].score

    @property
    def best_record(self) -> ModelRecord | None:
        """The current best model record, or None if no models tracked."""
        if not self._records:
            return None
        return self._records[0]

    @property
    def score_history(self) -> list[float]:
        """History of scores for trend analysis."""
        return self._score_history.copy()

    def get_top_k(self) -> list[ModelRecord]:
        """Get the top-k model records by score.

        Returns:
            List of ModelRecord objects sorted by score (descending).
            Length is min(keep_top_k, total tracked models).
        """
        return self._records[: self._config.keep_top_k]

    def update(
        self,
        checkpoint_path: str,
        iteration: int,
        eval_results: list[EvaluationResult],
    ) -> bool:
        """Update tracker with new evaluation results.

        Args:
            checkpoint_path: Path to the checkpoint file.
            iteration: Training iteration number.
            eval_results: List of EvaluationResult from evaluate_agent.

        Returns:
            True if this checkpoint is the new best model, False otherwise.

        Raises:
            FileNotFoundError: If checkpoint_path does not exist.
        """
        checkpoint_path = str(Path(checkpoint_path).resolve())

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Extract score and metrics
        score, metrics = extract_score(eval_results, self._config)

        # Store in history
        self._score_history.append(score)

        # Create record
        record = ModelRecord(
            checkpoint_path=checkpoint_path,
            iteration=iteration,
            score=score,
            timestamp=time.time(),
            metrics=metrics,
        )

        # Determine if this is a new best
        previous_best = self._records[0].score if self._records else 0.0
        is_new_best = score > previous_best

        # Add to records and sort
        self._records.append(record)
        self._records.sort(key=lambda r: r.score, reverse=True)

        # Keep only top-k
        self._records = self._records[: self._config.keep_top_k]

        # Update best model file if this is new best
        if is_new_best:
            self._update_best_model(checkpoint_path)
            logger.info(
                f"New best model at iteration {iteration}: "
                f"score={score:.1%} (was {previous_best:.1%})"
            )

        # Persist state
        self._save_state()

        return is_new_best

    def _update_best_model(self, checkpoint_path: str) -> None:
        """Update the best model file by copying the checkpoint."""
        src = Path(checkpoint_path)
        dst = self._best_model_path

        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            dst.unlink()

        shutil.copy2(src, dst)
        logger.debug(f"Copied best model to {dst}")

    def _save_state(self) -> None:
        """Persist tracker state to disk."""
        state = {
            "records": [r.to_dict() for r in self._records],
            "config": self._config.to_dict(),
            "best_model_path": str(self._best_model_path),
            "score_history": self._score_history[-1000:],  # Keep last 1000
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load tracker state from disk if available."""
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file) as f:
                state = json.load(f)

            self._records = [ModelRecord.from_dict(r) for r in state.get("records", [])]
            self._score_history = state.get("score_history", [])
            logger.info(f"Loaded {len(self._records)} model records from state file")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load tracker state from {self._state_file}: {e}")
            self._records = []

    def clear(self) -> None:
        """Clear all tracked records and remove state file."""
        self._records = []
        self._score_history = []
        if self._state_file.exists():
            self._state_file.unlink()
        logger.info("Cleared model tracker state")


# ============================================================================
# Early Stopping
# ============================================================================


class EarlyStopping:
    """Early stopping handler with optional advanced features.

    Monitors evaluation results and signals when training should stop based on:
    1. No improvement for `patience` evaluations (basic)
    2. Target score achieved (optional)
    3. Significant regression from best (optional)
    4. Prolonged stagnation (optional)

    Example:
        config = ModelSelectionConfig(
            patience=20,
            regression_threshold=0.10,
            stagnation_window=10,
        )
        early_stopping = EarlyStopping(config)

        for iteration in range(1000):
            eval_results = evaluate_agent(agent, config)
            should_stop, reason = early_stopping.update(eval_results, iteration)

            if should_stop:
                print(f"Stopping: {reason}")
                break
    """

    def __init__(self, config: ModelSelectionConfig | None = None) -> None:
        """Initialize early stopping handler.

        Args:
            config: Model selection configuration with early stopping parameters.
        """
        self._config = config or ModelSelectionConfig()

        # State
        self._best_score = 0.0
        self._best_iteration = 0
        self._evaluations_since_improvement = 0
        self._total_evaluations = 0
        self._score_history: list[float] = []
        self._iteration_history: list[int] = []

    @property
    def config(self) -> ModelSelectionConfig:
        """The early stopping configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Whether early stopping is enabled."""
        return self._config.enabled

    @property
    def best_score(self) -> float:
        """Best score observed so far."""
        return self._best_score

    @property
    def best_iteration(self) -> int:
        """Iteration where best score was achieved."""
        return self._best_iteration

    @property
    def evaluations_since_improvement(self) -> int:
        """Number of evaluations since last improvement."""
        return self._evaluations_since_improvement

    @property
    def total_evaluations(self) -> int:
        """Total number of evaluations processed."""
        return self._total_evaluations

    @property
    def patience_remaining(self) -> int:
        """Number of evaluations remaining before patience exhausted."""
        return max(0, self._config.patience - self._evaluations_since_improvement)

    def update(
        self,
        eval_results: list[EvaluationResult],
        iteration: int | None = None,
    ) -> tuple[bool, str]:
        """Update early stopping state and check stopping conditions.

        Args:
            eval_results: List of evaluation results.
            iteration: Optional iteration number for logging.

        Returns:
            Tuple of (should_stop, reason) where reason explains why stopping.
        """
        # Extract score
        score, _ = extract_score(eval_results, self._config)

        # Update history
        self._score_history.append(score)
        self._iteration_history.append(iteration or self._total_evaluations)
        self._total_evaluations += 1

        # Not enabled - never stop
        if not self._config.enabled:
            return False, ""

        # Not enough evaluations yet
        if self._total_evaluations < self._config.min_evaluations:
            # Still track best
            if score > self._best_score:
                self._best_score = score
                self._best_iteration = iteration or self._total_evaluations
                self._evaluations_since_improvement = 0
            return False, ""

        # Check if target achieved
        if self._config.target_score is not None and score >= self._config.target_score:
            reason = (
                f"Target score {self._config.target_score:.1%} achieved "
                f"({score:.1%}). Stopping."
            )
            logger.info(f"Early stopping: {reason}")
            return True, reason

        # Check for regression from best
        if self._config.regression_threshold is not None and self._best_score > 0:
            regression = self._best_score - score
            if regression > self._config.regression_threshold:
                reason = (
                    f"Score regressed {regression:.1%} from best "
                    f"({self._best_score:.1%} -> {score:.1%}), "
                    f"exceeds threshold {self._config.regression_threshold:.1%}"
                )
                logger.warning(f"Early stopping: {reason}")
                return True, reason

        # Update best tracking
        improvement = score - self._best_score
        if improvement >= self._config.min_improvement:
            self._best_score = score
            self._best_iteration = iteration or self._total_evaluations
            self._evaluations_since_improvement = 0
            logger.debug(f"Score improved to {score:.1%}")
        else:
            self._evaluations_since_improvement += 1

        # Check patience exhausted
        if self._evaluations_since_improvement >= self._config.patience:
            reason = (
                f"No improvement for {self._config.patience} evaluations. "
                f"Best: {self._best_score:.1%}, Current: {score:.1%}"
            )
            logger.info(f"Early stopping: {reason}")
            return True, reason

        # Check stagnation (low variance in recent window)
        if (
            self._config.stagnation_window > 0
            and len(self._score_history) >= self._config.stagnation_window
        ):
            recent = self._score_history[-self._config.stagnation_window :]
            mean_score = sum(recent) / len(recent)
            variance = sum((x - mean_score) ** 2 for x in recent) / len(recent)

            # Stagnation: low variance AND below best AND patience nearly exhausted
            is_stagnating = (
                variance < 0.0001  # Very low variance
                and mean_score < self._best_score - self._config.min_improvement
                and self._evaluations_since_improvement >= self._config.patience // 2
            )
            if is_stagnating:
                reason = (
                    f"Score stagnating around {mean_score:.1%} "
                    f"(variance={variance:.6f}) for {self._config.stagnation_window} "
                    f"evaluations, below best {self._best_score:.1%}"
                )
                logger.info(f"Early stopping: {reason}")
                return True, reason

        return False, ""

    def get_statistics(self) -> dict[str, Any]:
        """Get current early stopping statistics.

        Returns:
            Dictionary of statistics.
        """
        stats: dict[str, Any] = {
            "best_score": self._best_score,
            "best_iteration": self._best_iteration,
            "evaluations_since_improvement": self._evaluations_since_improvement,
            "total_evaluations": self._total_evaluations,
            "patience": self._config.patience,
            "patience_remaining": self.patience_remaining,
        }

        if self._score_history:
            stats["current_score"] = self._score_history[-1]

            # Compute recent trend
            if len(self._score_history) >= 5:
                recent = self._score_history[-5:]
                trend = (recent[-1] - recent[0]) / 4
                stats["recent_trend"] = trend
                if trend > 0.01:
                    stats["trend_direction"] = "improving"
                elif trend < -0.01:
                    stats["trend_direction"] = "declining"
                else:
                    stats["trend_direction"] = "stable"

        return stats

    def reset(self) -> None:
        """Reset early stopping state to initial values."""
        self._best_score = 0.0
        self._best_iteration = 0
        self._evaluations_since_improvement = 0
        self._total_evaluations = 0
        self._score_history = []
        self._iteration_history = []
        logger.debug("Early stopping state reset")


# ============================================================================
# Checkpoint Comparison
# ============================================================================


def compare_checkpoints(
    checkpoint1: str,
    checkpoint2: str,
    num_games: int = 100,
    device: str | None = None,
) -> dict[str, Any]:
    """Head-to-head comparison between two neural network checkpoints.

    Loads both checkpoints, creates agents, and plays games between them
    to determine which checkpoint performs better.

    Args:
        checkpoint1: Path to first checkpoint file.
        checkpoint2: Path to second checkpoint file.
        num_games: Number of games to play (alternating sides).
        device: Device for inference. If None, auto-detects.

    Returns:
        Dictionary with comparison results:
            - winner: "checkpoint1", "checkpoint2", or "tie"
            - checkpoint1_wins: Number of wins for checkpoint1
            - checkpoint2_wins: Number of wins for checkpoint2
            - draws: Number of draws
            - win_rate_1: Win rate for checkpoint1
            - win_rate_2: Win rate for checkpoint2
            - is_significant: Whether result is statistically significant (p < 0.05)
            - p_value: Approximate p-value for the difference

    Raises:
        FileNotFoundError: If either checkpoint does not exist.
        ImportError: If required modules are not available.

    Example:
        result = compare_checkpoints(
            "checkpoints/iter_1000.pt",
            "checkpoints/iter_2000.pt",
            num_games=100,
        )
        print(f"Winner: {result['winner']} (p={result['p_value']:.4f})")
    """
    from _02_agents.neural.agent import load_neural_agent

    from .evaluation import compare_agents

    # Validate paths
    path1 = Path(checkpoint1)
    path2 = Path(checkpoint2)

    if not path1.exists():
        raise FileNotFoundError(f"Checkpoint 1 not found: {checkpoint1}")
    if not path2.exists():
        raise FileNotFoundError(f"Checkpoint 2 not found: {checkpoint2}")

    # Load agents
    logger.info(f"Loading checkpoint 1: {checkpoint1}")
    agent1 = load_neural_agent(checkpoint1, mode="greedy", device=device)

    logger.info(f"Loading checkpoint 2: {checkpoint2}")
    agent2 = load_neural_agent(checkpoint2, mode="greedy", device=device)

    # Run comparison
    logger.info(f"Comparing checkpoints with {num_games} games...")
    result = compare_agents(
        agent1=agent1,
        agent2=agent2,
        num_games=num_games,
        play_both_sides=True,
    )

    # Determine winner
    wins_1 = result["wins_1"]
    wins_2 = result["wins_2"]

    if not result["significantly_different"]:
        winner = "tie"
    elif wins_1 > wins_2:
        winner = "checkpoint1"
    else:
        winner = "checkpoint2"

    return {
        "winner": winner,
        "checkpoint1_wins": wins_1,
        "checkpoint2_wins": wins_2,
        "draws": result["draws"],
        "win_rate_1": result["win_rate_1"],
        "win_rate_2": result["win_rate_2"],
        "is_significant": result["significantly_different"],
        "p_value": result["p_value"],
        "checkpoint1": checkpoint1,
        "checkpoint2": checkpoint2,
        "num_games": num_games,
    }


# ============================================================================
# Checkpoint Cleanup
# ============================================================================


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_best: int = 5,
    keep_recent: int = 10,
    protected_paths: list[str] | None = None,
) -> list[str]:
    """Remove old checkpoints while keeping best and most recent.

    Analyzes checkpoint directory to identify and remove old checkpoints,
    preserving the top-performing models and most recent saves.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        keep_best: Number of best-performing checkpoints to keep.
        keep_recent: Number of most recent checkpoints to keep.
        protected_paths: List of checkpoint paths that should never be deleted.

    Returns:
        List of deleted checkpoint paths.

    Example:
        deleted = cleanup_old_checkpoints(
            checkpoint_dir="checkpoints/experiment1",
            keep_best=5,
            keep_recent=10,
            protected_paths=["best_model.pt"],
        )
        print(f"Deleted {len(deleted)} old checkpoints")
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return []

    # Normalize protected paths
    protected_set: set[Path] = set()
    if protected_paths:
        for p in protected_paths:
            path = Path(p)
            if not path.is_absolute():
                path = checkpoint_path / path
            protected_set.add(path.resolve())

    # Always protect state files
    protected_set.add((checkpoint_path / "model_tracker_state.json").resolve())

    # Find all checkpoint files
    checkpoint_files: list[tuple[Path, float]] = []
    for pt_file in checkpoint_path.glob("*.pt"):
        resolved = pt_file.resolve()
        if resolved in protected_set:
            continue
        # Get modification time
        mtime = pt_file.stat().st_mtime
        checkpoint_files.append((pt_file, mtime))

    if not checkpoint_files:
        logger.info("No checkpoints found to clean up")
        return []

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)

    # Determine which to keep
    recent_to_keep = {f for f, _ in checkpoint_files[:keep_recent]}

    # Load tracker state to identify best models
    best_to_keep: set[Path] = set()
    tracker_state_file = checkpoint_path / "model_tracker_state.json"
    if tracker_state_file.exists():
        try:
            with open(tracker_state_file) as f:
                state = json.load(f)
            records = state.get("records", [])[:keep_best]
            for record in records:
                best_path = Path(record.get("checkpoint_path", ""))
                if best_path.exists():
                    best_to_keep.add(best_path.resolve())
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load tracker state for cleanup: {e}")

    # Identify files to delete
    keep_set = recent_to_keep | best_to_keep
    to_delete = [f for f, _ in checkpoint_files if f not in keep_set]

    # Delete files
    deleted: list[str] = []
    for checkpoint in to_delete:
        try:
            checkpoint.unlink()
            deleted.append(str(checkpoint))
            logger.debug(f"Deleted checkpoint: {checkpoint}")

            # Also delete associated JSON config if present
            config_file = checkpoint.with_suffix(".json")
            if config_file.exists():
                config_file.unlink()
                logger.debug(f"Deleted config: {config_file}")
        except OSError as e:
            logger.warning(f"Failed to delete {checkpoint}: {e}")

    if deleted:
        logger.info(
            f"Cleaned up {len(deleted)} checkpoints. "
            f"Kept {len(recent_to_keep)} recent + {len(best_to_keep)} best"
        )

    return deleted


# ============================================================================
# Factory Functions
# ============================================================================


def create_model_selection(
    checkpoint_dir: str,
    experiment_name: str = "",
    config: ModelSelectionConfig | None = None,
) -> tuple[ModelTracker, EarlyStopping, Callable[[int, str, list[EvaluationResult]], tuple[bool, bool, str]]]:
    """Factory function to create model selection components and callback.

    Convenience function that creates all model selection components
    and returns a callback suitable for use in custom training loops.

    Args:
        checkpoint_dir: Base checkpoint directory.
        experiment_name: Name of the experiment (appended to checkpoint_dir).
        config: Model selection configuration. Uses defaults if None.

    Returns:
        Tuple of (tracker, early_stopping, callback) where:
            - tracker: ModelTracker instance
            - early_stopping: EarlyStopping instance
            - callback: Function(iteration, checkpoint_path, eval_results) ->
                (is_best, should_stop, stop_reason)

    Example:
        tracker, early_stopping, callback = create_model_selection(
            checkpoint_dir="checkpoints",
            experiment_name="exp1",
            config=ModelSelectionConfig(
                metric="weighted",
                weights={"random": 0.7, "heuristic": 0.3},
                patience=20,
                regression_threshold=0.10,
            ),
        )

        for iteration in range(1000):
            # ... training and evaluation ...
            is_best, should_stop, reason = callback(iteration, checkpoint_path, eval_results)

            if is_best:
                print("New best model!")

            if should_stop:
                print(f"Stopping: {reason}")
                break
    """
    if config is None:
        config = ModelSelectionConfig()

    full_checkpoint_dir = str(Path(checkpoint_dir) / experiment_name) if experiment_name else checkpoint_dir

    tracker = ModelTracker(
        checkpoint_dir=full_checkpoint_dir,
        config=config,
    )

    early_stopping = EarlyStopping(config)

    def callback(
        iteration: int,
        checkpoint_path: str,
        eval_results: list[EvaluationResult],
    ) -> tuple[bool, bool, str]:
        """Model selection callback.

        Args:
            iteration: Current training iteration.
            checkpoint_path: Path to current checkpoint.
            eval_results: Evaluation results.

        Returns:
            Tuple of (is_best, should_stop, stop_reason).
        """
        is_best = False
        if Path(checkpoint_path).exists():
            is_best = tracker.update(checkpoint_path, iteration, eval_results)

        should_stop, reason = early_stopping.update(eval_results, iteration)

        return is_best, should_stop, reason

    return tracker, early_stopping, callback


__all__ = [
    "EarlyStopping",
    "ModelRecord",
    "ModelSelectionConfig",
    "ModelTracker",
    "cleanup_old_checkpoints",
    "compare_checkpoints",
    "create_model_selection",
    "extract_score",
]
