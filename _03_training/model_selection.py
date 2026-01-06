"""Model selection utilities for neural network training.

This module provides:
- BestModelTracker: Track and save best-performing models during training
- EarlyStopping: Stop training when no improvement is observed
- compare_checkpoints: Head-to-head comparison between checkpoints
- cleanup_old_checkpoints: Remove old checkpoints while keeping best and recent
- integrate_model_selection: Integrate model selection with trainer

Example:
    tracker = BestModelTracker(checkpoint_dir="checkpoints")
    early_stopping = EarlyStopping(EarlyStoppingConfig(patience=20))

    for iteration in range(1000):
        # ... training ...
        eval_results = evaluate_agent(agent, eval_config)

        is_best = tracker.update(checkpoint_path, iteration, eval_results)
        early_stopping.update(eval_results)

        if early_stopping.should_stop(eval_results):
            print("Early stopping triggered")
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
    from .trainer import Trainer

logger = logging.getLogger(__name__)


# ============================================================================
# Model Record
# ============================================================================


@dataclass
class ModelRecord:
    """Record of a saved model checkpoint with evaluation metrics.

    Attributes:
        checkpoint_path: Absolute path to the checkpoint file.
        iteration: Training iteration when checkpoint was saved.
        win_rate_vs_mcts: Win rate against MCTS opponent.
        win_rate_vs_heuristic: Win rate against heuristic opponent.
        timestamp: Unix timestamp when record was created.
        metrics: Optional dictionary of additional evaluation metrics.
    """

    checkpoint_path: str
    iteration: int
    win_rate_vs_mcts: float
    win_rate_vs_heuristic: float
    timestamp: float
    metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRecord:
        """Create record from dictionary."""
        return cls(**data)

    @property
    def primary_score(self) -> float:
        """Primary metric for ranking models (win rate vs MCTS)."""
        return self.win_rate_vs_mcts


# ============================================================================
# Best Model Tracker
# ============================================================================


class BestModelTracker:
    """Track and maintain the best-performing model checkpoints.

    Monitors evaluation results and saves models that achieve the highest
    win rate against MCTS opponents. Maintains a leaderboard of top-k models.

    The tracker persists its state to disk, allowing training to resume
    with the correct best model information.

    Attributes:
        checkpoint_dir: Directory containing checkpoint files.
        best_model_path: Path where the current best model is symlinked/copied.
        keep_top_k: Number of top models to track.

    Example:
        tracker = BestModelTracker(
            checkpoint_dir="checkpoints/experiment1",
            best_model_path="best_model.pt",
            keep_top_k=5,
        )

        # After evaluation
        is_new_best = tracker.update(checkpoint_path, iteration, eval_results)
        if is_new_best:
            print(f"New best model! Win rate: {tracker.best_win_rate:.1%}")
    """

    def __init__(
        self,
        checkpoint_dir: str,
        best_model_path: str = "best_model.pt",
        keep_top_k: int = 5,
    ) -> None:
        """Initialize the best model tracker.

        Args:
            checkpoint_dir: Directory for storing checkpoints and tracker state.
            best_model_path: Relative or absolute path for best model.
                If relative, resolved relative to checkpoint_dir.
            keep_top_k: Number of top models to maintain in the leaderboard.

        Raises:
            ValueError: If keep_top_k is less than 1.
        """
        if keep_top_k < 1:
            raise ValueError(f"keep_top_k must be at least 1, got {keep_top_k}")

        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resolve best model path
        best_path = Path(best_model_path)
        if not best_path.is_absolute():
            best_path = self._checkpoint_dir / best_path
        self._best_model_path = best_path

        self._keep_top_k = keep_top_k
        self._records: list[ModelRecord] = []
        self._state_file = self._checkpoint_dir / "model_tracker_state.json"

        # Load existing state if available
        self._load_state()

    @property
    def best_model_path(self) -> str | None:
        """Path to the current best model, or None if no models tracked."""
        if not self._records:
            return None
        return str(self._best_model_path)

    @property
    def best_win_rate(self) -> float:
        """Win rate of the current best model, or 0.0 if no models tracked."""
        if not self._records:
            return 0.0
        return self._records[0].primary_score

    @property
    def best_record(self) -> ModelRecord | None:
        """The current best model record, or None if no models tracked."""
        if not self._records:
            return None
        return self._records[0]

    def get_top_k(self) -> list[ModelRecord]:
        """Get the top-k model records by win rate.

        Returns:
            List of ModelRecord objects sorted by primary score (descending).
            Length is min(keep_top_k, total tracked models).
        """
        return self._records[: self._keep_top_k]

    def update(
        self,
        checkpoint_path: str,
        iteration: int,
        eval_results: list[EvaluationResult],
    ) -> bool:
        """Update tracker with new evaluation results.

        Extracts win rates from evaluation results, creates a model record,
        and updates the leaderboard. If the new model is the best so far,
        updates the best model symlink/copy.

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

        # Extract win rates from results
        win_rate_mcts = 0.0
        win_rate_heuristic = 0.0
        metrics: dict[str, float] = {}

        for result in eval_results:
            opponent = result.opponent_name.lower()
            if "mcts" in opponent:
                # Use the highest MCTS win rate if multiple MCTS opponents
                win_rate_mcts = max(win_rate_mcts, result.win_rate)
            elif "heuristic" in opponent:
                win_rate_heuristic = result.win_rate

            # Store all metrics
            metrics[f"{result.opponent_name}/win_rate"] = result.win_rate
            metrics[f"{result.opponent_name}/avg_margin"] = result.avg_point_margin

        # Create record
        record = ModelRecord(
            checkpoint_path=checkpoint_path,
            iteration=iteration,
            win_rate_vs_mcts=win_rate_mcts,
            win_rate_vs_heuristic=win_rate_heuristic,
            timestamp=time.time(),
            metrics=metrics,
        )

        # Determine if this is a new best
        previous_best = self._records[0].primary_score if self._records else 0.0
        is_new_best = record.primary_score > previous_best

        # Add to records and sort
        self._records.append(record)
        self._records.sort(key=lambda r: r.primary_score, reverse=True)

        # Keep only top-k
        self._records = self._records[: self._keep_top_k]

        # Update best model file if this is new best
        if is_new_best:
            self._update_best_model(checkpoint_path)
            logger.info(
                f"New best model at iteration {iteration}: "
                f"MCTS win rate {win_rate_mcts:.1%} (was {previous_best:.1%})"
            )

        # Persist state
        self._save_state()

        return is_new_best

    def _update_best_model(self, checkpoint_path: str) -> None:
        """Update the best model file by copying the checkpoint.

        Args:
            checkpoint_path: Path to the new best checkpoint.
        """
        src = Path(checkpoint_path)
        dst = self._best_model_path

        # Ensure parent directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing best model if present
        if dst.exists():
            dst.unlink()

        # Copy checkpoint to best model path
        shutil.copy2(src, dst)
        logger.debug(f"Copied best model to {dst}")

    def _save_state(self) -> None:
        """Persist tracker state to disk."""
        state = {
            "records": [r.to_dict() for r in self._records],
            "keep_top_k": self._keep_top_k,
            "best_model_path": str(self._best_model_path),
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
            logger.info(f"Loaded {len(self._records)} model records from state file")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load tracker state: {e}")
            self._records = []

    def clear(self) -> None:
        """Clear all tracked records and remove state file."""
        self._records = []
        if self._state_file.exists():
            self._state_file.unlink()
        logger.info("Cleared model tracker state")


# ============================================================================
# Early Stopping
# ============================================================================


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping criteria.

    Attributes:
        patience: Number of evaluations without improvement before stopping.
        min_improvement: Minimum win rate improvement to count as improvement.
        target_win_rate: Optional target win rate; stop if achieved.
        target_opponent: Opponent name to use for tracking improvement.
    """

    patience: int = 20
    min_improvement: float = 0.01
    target_win_rate: float | None = None
    target_opponent: str = "mcts-500"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EarlyStoppingConfig:
        """Create config from dictionary."""
        return cls(**data)


class EarlyStopping:
    """Early stopping handler for training.

    Monitors evaluation results and signals when training should stop
    based on lack of improvement or achieving target performance.

    The handler tracks the best observed win rate and counts evaluations
    since the last improvement. Training stops when:
    1. No improvement for `patience` evaluations, OR
    2. Target win rate is achieved (if configured)

    Example:
        config = EarlyStoppingConfig(patience=20, target_win_rate=0.7)
        early_stopping = EarlyStopping(config)

        for iteration in range(1000):
            # ... training ...
            eval_results = evaluate_agent(...)

            if early_stopping.should_stop(eval_results):
                print("Early stopping triggered")
                break

            early_stopping.update(eval_results)
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        """Initialize early stopping handler.

        Args:
            config: Configuration for early stopping criteria.
        """
        self._config = config
        self._best_win_rate = 0.0
        self._evaluations_without_improvement = 0
        self._total_evaluations = 0

    @property
    def config(self) -> EarlyStoppingConfig:
        """The early stopping configuration."""
        return self._config

    @property
    def evaluations_without_improvement(self) -> int:
        """Number of evaluations since last improvement."""
        return self._evaluations_without_improvement

    @property
    def best_win_rate(self) -> float:
        """Best win rate observed so far."""
        return self._best_win_rate

    @property
    def total_evaluations(self) -> int:
        """Total number of evaluations processed."""
        return self._total_evaluations

    def should_stop(self, eval_results: list[EvaluationResult]) -> bool:
        """Check if training should stop based on evaluation results.

        Note: This checks the current state plus the new results, but does
        not update the internal state. Call update() after should_stop()
        to record the results.

        Args:
            eval_results: List of EvaluationResult from evaluate_agent.

        Returns:
            True if training should stop, False otherwise.
        """
        # Extract win rate for target opponent
        current_win_rate = self._get_target_win_rate(eval_results)

        # Check if target achieved
        if (
            self._config.target_win_rate is not None
            and current_win_rate >= self._config.target_win_rate
        ):
            logger.info(
                f"Target win rate {self._config.target_win_rate:.1%} achieved "
                f"({current_win_rate:.1%}). Stopping."
            )
            return True

        # Check patience
        improvement = current_win_rate - self._best_win_rate
        would_exceed_patience = (
            improvement < self._config.min_improvement
            and self._evaluations_without_improvement + 1 >= self._config.patience
        )
        if would_exceed_patience:
            logger.info(
                f"No improvement for {self._config.patience} evaluations. "
                f"Best: {self._best_win_rate:.1%}, Current: {current_win_rate:.1%}. Stopping."
            )
            return True

        return False

    def update(self, eval_results: list[EvaluationResult]) -> None:
        """Update early stopping state with new evaluation results.

        Args:
            eval_results: List of EvaluationResult from evaluate_agent.
        """
        current_win_rate = self._get_target_win_rate(eval_results)
        improvement = current_win_rate - self._best_win_rate

        if improvement >= self._config.min_improvement:
            # Significant improvement
            self._best_win_rate = current_win_rate
            self._evaluations_without_improvement = 0
            logger.debug(
                f"Win rate improved to {current_win_rate:.1%} "
                f"(+{improvement:.1%})"
            )
        else:
            # No significant improvement
            self._evaluations_without_improvement += 1
            logger.debug(
                f"No improvement ({self._evaluations_without_improvement}/{self._config.patience}). "
                f"Best: {self._best_win_rate:.1%}, Current: {current_win_rate:.1%}"
            )

        self._total_evaluations += 1

    def _get_target_win_rate(self, eval_results: list[EvaluationResult]) -> float:
        """Extract win rate for the target opponent.

        Args:
            eval_results: List of evaluation results.

        Returns:
            Win rate against target opponent, or 0.0 if not found.
        """
        target = self._config.target_opponent.lower()

        for result in eval_results:
            if result.opponent_name.lower() == target:
                return result.win_rate

        # Try partial match for MCTS variants
        for result in eval_results:
            if "mcts" in target and "mcts" in result.opponent_name.lower():
                return result.win_rate

        logger.warning(f"Target opponent '{self._config.target_opponent}' not found in results")
        return 0.0

    def reset(self) -> None:
        """Reset early stopping state to initial values."""
        self._best_win_rate = 0.0
        self._evaluations_without_improvement = 0
        self._total_evaluations = 0
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
# Trainer Integration
# ============================================================================


def integrate_model_selection(
    trainer: Trainer,
    tracker: BestModelTracker,
    early_stopping: EarlyStopping | None = None,
) -> Callable[[list[EvaluationResult]], bool]:
    """Create callback for integrating model selection with trainer.

    Returns a callback function that the trainer calls after each evaluation.
    The callback:
    1. Updates the best model tracker with new results
    2. Checks early stopping criteria (if configured)
    3. Returns whether training should stop

    Args:
        trainer: The Trainer instance being monitored.
        tracker: BestModelTracker for saving best models.
        early_stopping: Optional EarlyStopping handler.

    Returns:
        Callback function that accepts evaluation results and returns
        True if training should stop, False otherwise.

    Example:
        tracker = BestModelTracker(checkpoint_dir="checkpoints")
        early_stopping = EarlyStopping(EarlyStoppingConfig(patience=20))

        callback = integrate_model_selection(trainer, tracker, early_stopping)

        # In training loop:
        eval_results = evaluate_agent(...)
        should_stop = callback(eval_results)
        if should_stop:
            break
    """

    def model_selection_callback(eval_results: list[EvaluationResult]) -> bool:
        """Callback for processing evaluation results.

        Args:
            eval_results: Evaluation results from evaluate_agent.

        Returns:
            True if training should stop, False otherwise.
        """
        iteration = trainer.current_iteration

        # Get latest checkpoint path
        checkpoint_dir = Path(trainer.config.checkpoint_dir) / trainer.config.experiment_name
        checkpoint_path = checkpoint_dir / f"iter_{iteration:06d}.pt"

        # Update best model tracker if checkpoint exists
        if checkpoint_path.exists():
            is_best = tracker.update(str(checkpoint_path), iteration, eval_results)
            if is_best:
                logger.info(f"New best model saved at iteration {iteration}")

        # Check early stopping
        if early_stopping is not None:
            if early_stopping.should_stop(eval_results):
                return True
            early_stopping.update(eval_results)

        return False

    return model_selection_callback


def create_model_selection_callback(
    checkpoint_dir: str,
    experiment_name: str,
    early_stopping_config: EarlyStoppingConfig | None = None,
    keep_top_k: int = 5,
) -> tuple[BestModelTracker, EarlyStopping | None, Callable[[int, str, list[EvaluationResult]], bool]]:
    """Factory function to create model selection components and callback.

    Convenience function that creates all model selection components
    and returns a callback suitable for use in custom training loops.

    Args:
        checkpoint_dir: Base checkpoint directory.
        experiment_name: Name of the experiment.
        early_stopping_config: Optional early stopping configuration.
        keep_top_k: Number of best models to keep.

    Returns:
        Tuple of (tracker, early_stopping, callback) where:
            - tracker: BestModelTracker instance
            - early_stopping: EarlyStopping instance or None
            - callback: Function(iteration, checkpoint_path, eval_results) -> should_stop

    Example:
        tracker, early_stopping, callback = create_model_selection_callback(
            checkpoint_dir="checkpoints",
            experiment_name="exp1",
            early_stopping_config=EarlyStoppingConfig(patience=20),
        )

        # In training loop:
        should_stop = callback(iteration, checkpoint_path, eval_results)
    """
    full_checkpoint_dir = str(Path(checkpoint_dir) / experiment_name)

    tracker = BestModelTracker(
        checkpoint_dir=full_checkpoint_dir,
        keep_top_k=keep_top_k,
    )

    early_stopping = None
    if early_stopping_config is not None:
        early_stopping = EarlyStopping(early_stopping_config)

    def callback(
        iteration: int,
        checkpoint_path: str,
        eval_results: list[EvaluationResult],
    ) -> bool:
        """Model selection callback.

        Args:
            iteration: Current training iteration.
            checkpoint_path: Path to current checkpoint.
            eval_results: Evaluation results.

        Returns:
            True if training should stop.
        """
        # Update tracker
        if Path(checkpoint_path).exists():
            tracker.update(checkpoint_path, iteration, eval_results)

        # Check early stopping
        if early_stopping is not None:
            if early_stopping.should_stop(eval_results):
                return True
            early_stopping.update(eval_results)

        return False

    return tracker, early_stopping, callback


__all__ = [
    "BestModelTracker",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ModelRecord",
    "cleanup_old_checkpoints",
    "compare_checkpoints",
    "create_model_selection_callback",
    "integrate_model_selection",
]
