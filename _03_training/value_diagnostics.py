"""Value network diagnostic tools for detecting training failures early.

This module provides tools to diagnose value network calibration issues
that can cause MCTS training to fail. The primary failure mode is when
the value network outputs constant values regardless of game state,
preventing MCTS from learning meaningful evaluations.

Key diagnostic metrics:
- Pearson r-squared: Correlation between predicted values and actual outcomes
- Mean absolute error (MAE): Average prediction error magnitude
- Value distribution statistics: Mean, std, min, max of predictions
- Calibration by outcome: Average predicted value per game outcome (win/draw/loss)

Early warning signs of cold-start failure:
- r-squared close to 0: Value network not learning
- Very low value std (< 0.1): Predictions collapsing to constant
- MAE close to 0.5-1.0 with r-squared near 0: Random predictions
- Calibration by outcome showing similar values: Not distinguishing states

Example:
    from _03_training.value_diagnostics import (
        ValueCalibrationTracker,
        diagnose_value_network,
        log_value_diagnostics,
    )

    # During training
    tracker = ValueCalibrationTracker()

    # After each iteration, collect samples
    tracker.add_samples(predicted_values, actual_outcomes)

    # Compute and log diagnostics
    report = tracker.compute_report()
    log_value_diagnostics(tracker, iteration, logger)

    # Standalone diagnosis of a network
    report = diagnose_value_network(network, num_games=100)
    print(f"r-squared: {report['r_squared']:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from _02_agents.neural.network import BeastyBarNetwork
    from _03_training.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


# ============================================================================
# Value Calibration Tracker
# ============================================================================


@dataclass
class ValueCalibrationTracker:
    """Track predicted values vs actual game outcomes for calibration analysis.

    Collects (predicted_value, actual_outcome) pairs during training and
    computes calibration metrics to detect value network learning failures.

    The tracker maintains a rolling window of samples to provide both
    recent and cumulative statistics.

    Attributes:
        predicted_values: List of value network predictions [-1, 1].
        actual_outcomes: List of actual game outcomes (-1=loss, 0=draw, 1=win).
        max_samples: Maximum samples to retain (rolling window).

    Example:
        tracker = ValueCalibrationTracker(max_samples=10000)

        # During self-play, collect predictions and outcomes
        for game in games:
            for step in game.steps:
                tracker.add_sample(step.value, step.final_reward)

        # Compute calibration report
        report = tracker.compute_report()
        print(f"r-squared: {report['r_squared']:.4f}")
    """

    predicted_values: list[float] = field(default_factory=list)
    actual_outcomes: list[float] = field(default_factory=list)
    max_samples: int = 10000

    def add_sample(self, predicted_value: float, actual_outcome: float) -> None:
        """Add a single (prediction, outcome) sample.

        Args:
            predicted_value: Value network prediction in [-1, 1].
            actual_outcome: Actual game outcome (-1=loss, 0=draw, 1=win).
        """
        self.predicted_values.append(predicted_value)
        self.actual_outcomes.append(actual_outcome)

        # Maintain rolling window
        if len(self.predicted_values) > self.max_samples:
            self.predicted_values = self.predicted_values[-self.max_samples :]
            self.actual_outcomes = self.actual_outcomes[-self.max_samples :]

    def add_samples(
        self,
        predicted_values: list[float] | NDArray[np.float32],
        actual_outcomes: list[float] | NDArray[np.float32],
    ) -> None:
        """Add multiple samples at once.

        Args:
            predicted_values: Array of value predictions.
            actual_outcomes: Array of actual outcomes.

        Raises:
            ValueError: If arrays have different lengths.
        """
        if len(predicted_values) != len(actual_outcomes):
            raise ValueError(
                f"Length mismatch: {len(predicted_values)} predictions vs "
                f"{len(actual_outcomes)} outcomes"
            )

        for pred, outcome in zip(predicted_values, actual_outcomes, strict=True):
            self.add_sample(float(pred), float(outcome))

    def clear(self) -> None:
        """Clear all collected samples."""
        self.predicted_values.clear()
        self.actual_outcomes.clear()

    def sample_count(self) -> int:
        """Return number of samples collected."""
        return len(self.predicted_values)

    def compute_report(self) -> dict[str, float]:
        """Compute calibration metrics from collected samples.

        Returns:
            Dictionary containing:
            - sample_count: Number of samples
            - r_squared: Pearson r-squared (0 = no correlation, 1 = perfect)
            - mae: Mean absolute error
            - mse: Mean squared error
            - pred_mean: Mean of predictions
            - pred_std: Standard deviation of predictions
            - pred_min: Minimum prediction
            - pred_max: Maximum prediction
            - outcome_mean: Mean of actual outcomes
            - calibration_win: Mean prediction for wins
            - calibration_loss: Mean prediction for losses
            - calibration_draw: Mean prediction for draws
        """
        n = len(self.predicted_values)

        if n == 0:
            return self._empty_report()

        preds = np.array(self.predicted_values, dtype=np.float32)
        outcomes = np.array(self.actual_outcomes, dtype=np.float32)

        # Basic statistics
        pred_mean = float(np.mean(preds))
        pred_std = float(np.std(preds))
        pred_min = float(np.min(preds))
        pred_max = float(np.max(preds))
        outcome_mean = float(np.mean(outcomes))

        # Error metrics
        errors = preds - outcomes
        mae = float(np.mean(np.abs(errors)))
        mse = float(np.mean(errors**2))

        # Pearson r-squared
        r_squared = self._compute_r_squared(preds, outcomes)

        # Calibration by outcome
        calibration_win = self._mean_prediction_for_outcome(preds, outcomes, 1.0)
        calibration_loss = self._mean_prediction_for_outcome(preds, outcomes, -1.0)
        calibration_draw = self._mean_prediction_for_outcome(preds, outcomes, 0.0)

        return {
            "sample_count": float(n),
            "r_squared": r_squared,
            "mae": mae,
            "mse": mse,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "pred_min": pred_min,
            "pred_max": pred_max,
            "outcome_mean": outcome_mean,
            "calibration_win": calibration_win,
            "calibration_loss": calibration_loss,
            "calibration_draw": calibration_draw,
        }

    def _empty_report(self) -> dict[str, float]:
        """Return report with NaN values when no samples available."""
        return {
            "sample_count": 0.0,
            "r_squared": float("nan"),
            "mae": float("nan"),
            "mse": float("nan"),
            "pred_mean": float("nan"),
            "pred_std": float("nan"),
            "pred_min": float("nan"),
            "pred_max": float("nan"),
            "outcome_mean": float("nan"),
            "calibration_win": float("nan"),
            "calibration_loss": float("nan"),
            "calibration_draw": float("nan"),
        }

    def _compute_r_squared(
        self, preds: NDArray[np.float32], outcomes: NDArray[np.float32]
    ) -> float:
        """Compute Pearson r-squared between predictions and outcomes.

        Args:
            preds: Predicted values.
            outcomes: Actual outcomes.

        Returns:
            R-squared value in [0, 1], or 0 if computation fails.
        """
        if len(preds) < 2:
            return 0.0

        # Compute Pearson correlation coefficient
        pred_centered = preds - np.mean(preds)
        outcome_centered = outcomes - np.mean(outcomes)

        numerator = np.sum(pred_centered * outcome_centered)
        denominator = np.sqrt(np.sum(pred_centered**2) * np.sum(outcome_centered**2))

        if denominator < 1e-10:
            # No variance in predictions or outcomes
            return 0.0

        r = numerator / denominator
        r_squared = float(r**2)

        # Clamp to valid range (floating point issues)
        return max(0.0, min(1.0, r_squared))

    def _mean_prediction_for_outcome(
        self,
        preds: NDArray[np.float32],
        outcomes: NDArray[np.float32],
        target_outcome: float,
    ) -> float:
        """Compute mean prediction for a specific outcome value.

        Args:
            preds: Predicted values.
            outcomes: Actual outcomes.
            target_outcome: Outcome to filter by (-1, 0, or 1).

        Returns:
            Mean prediction for states with the target outcome, or NaN if none.
        """
        # Handle shaped rewards by checking sign for win/loss
        if target_outcome > 0:
            mask = outcomes > 0.5
        elif target_outcome < 0:
            mask = outcomes < -0.5
        else:
            mask = (outcomes >= -0.5) & (outcomes <= 0.5)

        if not mask.any():
            return float("nan")

        return float(np.mean(preds[mask]))

    def is_healthy(self, min_r_squared: float = 0.05, min_pred_std: float = 0.1) -> bool:
        """Check if value network appears to be learning.

        Args:
            min_r_squared: Minimum acceptable r-squared (default 0.05).
            min_pred_std: Minimum acceptable prediction std (default 0.1).

        Returns:
            True if value network shows signs of learning, False otherwise.
        """
        if self.sample_count() < 100:
            # Not enough samples to judge
            return True

        report = self.compute_report()

        # Check for collapsed predictions (constant output)
        if report["pred_std"] < min_pred_std:
            return False

        # Check for learning signal
        return report["r_squared"] >= min_r_squared

    def get_diagnostic_summary(self) -> str:
        """Generate human-readable diagnostic summary.

        Returns:
            Multi-line string with diagnostic information.
        """
        report = self.compute_report()

        if report["sample_count"] == 0:
            return "No samples collected yet."

        lines = [
            f"Value Network Diagnostics ({int(report['sample_count'])} samples)",
            "-" * 50,
            f"R-squared:     {report['r_squared']:.4f} (correlation with outcomes)",
            f"MAE:           {report['mae']:.4f} (mean absolute error)",
            f"Pred mean:     {report['pred_mean']:.4f}",
            f"Pred std:      {report['pred_std']:.4f}",
            f"Pred range:    [{report['pred_min']:.4f}, {report['pred_max']:.4f}]",
            "",
            "Calibration by outcome:",
            f"  Win states:  {report['calibration_win']:.4f} (should be near +1)",
            f"  Loss states: {report['calibration_loss']:.4f} (should be near -1)",
            f"  Draw states: {report['calibration_draw']:.4f} (should be near 0)",
        ]

        # Add warnings
        warnings = []
        if report["pred_std"] < 0.1:
            warnings.append("WARNING: Low prediction std - values may be collapsing")
        if report["r_squared"] < 0.05 and report["sample_count"] >= 1000:
            warnings.append("WARNING: Very low r-squared - value network not learning")
        if abs(report["calibration_win"] - report["calibration_loss"]) < 0.2:
            warnings.append("WARNING: Win/loss calibration too similar - poor discrimination")

        if warnings:
            lines.append("")
            lines.extend(warnings)

        return "\n".join(lines)


# ============================================================================
# Standalone Diagnosis Function
# ============================================================================


def diagnose_value_network(
    network: BeastyBarNetwork,
    num_games: int = 100,
    temperature: float = 1.0,
    device: Any = None,
) -> dict[str, Any]:
    """Play games and diagnose value network calibration.

    Runs self-play games using the network and collects value predictions
    along with actual game outcomes to compute calibration metrics.

    Args:
        network: Neural network to diagnose.
        num_games: Number of games to play for diagnosis.
        temperature: Temperature for action sampling.
        device: Device for inference (auto-detected if None).

    Returns:
        Dictionary containing:
        - All metrics from ValueCalibrationTracker.compute_report()
        - games_played: Number of games completed
        - avg_game_length: Average turns per game
        - p0_win_rate: Player 0 win rate
        - is_healthy: Boolean indicating if network appears healthy

    Example:
        report = diagnose_value_network(network, num_games=100)
        if not report['is_healthy']:
            print("Value network may have training issues!")
            print(f"R-squared: {report['r_squared']:.4f}")
    """
    import torch

    from _03_training.self_play import generate_games

    # Auto-detect device
    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Generate games
    network.eval()
    with torch.no_grad():
        trajectories = generate_games(
            network=network,
            num_games=num_games,
            temperature=temperature,
            device=device,
            use_vectorized=True,
        )

    # Collect value predictions and outcomes
    tracker = ValueCalibrationTracker()

    total_steps = 0
    for trajectory in trajectories:
        # Determine final reward for this game
        if trajectory.winner == 0:
            p0_reward = 1.0
            p1_reward = -1.0
        elif trajectory.winner == 1:
            p0_reward = -1.0
            p1_reward = 1.0
        else:
            p0_reward = 0.0
            p1_reward = 0.0

        # Collect P0 predictions
        for step in trajectory.steps_p0:
            tracker.add_sample(step.value, p0_reward)
            total_steps += 1

        # Collect P1 predictions
        for step in trajectory.steps_p1:
            tracker.add_sample(step.value, p1_reward)
            total_steps += 1

    # Compute calibration report
    report = tracker.compute_report()

    # Add game statistics
    games_played = len(trajectories)
    avg_game_length = (
        sum(t.game_length for t in trajectories) / games_played
        if games_played > 0
        else 0.0
    )
    p0_wins = sum(1 for t in trajectories if t.winner == 0)
    p0_win_rate = p0_wins / games_played if games_played > 0 else 0.0

    report["games_played"] = float(games_played)
    report["avg_game_length"] = avg_game_length
    report["p0_win_rate"] = p0_win_rate
    report["total_steps"] = float(total_steps)

    # Determine health
    report["is_healthy"] = tracker.is_healthy()

    return report


# ============================================================================
# Logging Integration
# ============================================================================


def log_value_diagnostics(
    tracker: ValueCalibrationTracker,
    iteration: int,
    experiment_tracker: ExperimentTracker,
) -> dict[str, float]:
    """Log value diagnostics to experiment tracker.

    Args:
        tracker: ValueCalibrationTracker with collected samples.
        iteration: Current training iteration.
        experiment_tracker: Tracker for logging metrics.

    Returns:
        Dictionary of logged metrics.
    """
    report = tracker.compute_report()

    # Prepare metrics with prefix (filter out NaN values)
    metrics = {}
    for k, v in report.items():
        if isinstance(v, float) and np.isnan(v):
            continue
        metrics[f"value_calibration/{k}"] = v

    # Log to tracker
    experiment_tracker.log_metrics(metrics, step=iteration)

    # Log warnings to console
    if report["sample_count"] >= 1000:
        if report["pred_std"] < 0.1:
            logger.warning(
                f"[Iteration {iteration}] Value network predictions collapsing "
                f"(std={report['pred_std']:.4f})"
            )
        if report["r_squared"] < 0.05:
            logger.warning(
                f"[Iteration {iteration}] Value network not learning "
                f"(r_squared={report['r_squared']:.4f})"
            )

    return metrics


def extract_values_from_trajectories(
    trajectories: list,
) -> tuple[list[float], list[float]]:
    """Extract predicted values and actual outcomes from trajectories.

    Helper function to convert trajectory data into format suitable
    for ValueCalibrationTracker.

    Args:
        trajectories: List of GameTrajectory objects.

    Returns:
        Tuple of (predicted_values, actual_outcomes) lists.
    """
    predicted_values: list[float] = []
    actual_outcomes: list[float] = []

    for trajectory in trajectories:
        # Determine rewards
        if trajectory.winner == 0:
            p0_reward = 1.0
            p1_reward = -1.0
        elif trajectory.winner == 1:
            p0_reward = -1.0
            p1_reward = 1.0
        else:
            p0_reward = 0.0
            p1_reward = 0.0

        # Collect P0 data
        for step in trajectory.steps_p0:
            predicted_values.append(step.value)
            actual_outcomes.append(p0_reward)

        # Collect P1 data
        for step in trajectory.steps_p1:
            predicted_values.append(step.value)
            actual_outcomes.append(p1_reward)

    return predicted_values, actual_outcomes


def check_cold_start_failure(
    tracker: ValueCalibrationTracker,
    iteration: int,
    min_iterations: int = 10,
    max_healthy_mae: float = 0.9,
) -> tuple[bool, str]:
    """Check for cold-start failure patterns.

    Cold-start failure occurs when the value network fails to learn
    meaningful state evaluations, typically showing:
    - Near-zero r-squared (no correlation with outcomes)
    - Very low prediction variance (collapsed to constant)
    - Poor calibration (similar predictions for wins and losses)

    Args:
        tracker: ValueCalibrationTracker with samples.
        iteration: Current training iteration.
        min_iterations: Don't flag as failure before this iteration.
        max_healthy_mae: MAE above this with low r-squared indicates failure.

    Returns:
        Tuple of (is_failing, reason_message).
    """
    if iteration < min_iterations:
        return False, "Too early to diagnose"

    if tracker.sample_count() < 1000:
        return False, "Insufficient samples"

    report = tracker.compute_report()

    # Check for collapsed predictions
    if report["pred_std"] < 0.05:
        return True, f"Predictions collapsed to constant (std={report['pred_std']:.4f})"

    # Check for no learning signal
    if report["r_squared"] < 0.01 and report["mae"] > max_healthy_mae:
        return True, f"No correlation with outcomes (r_squared={report['r_squared']:.4f})"

    # Check for poor discrimination
    cal_diff = abs(report["calibration_win"] - report["calibration_loss"])
    if cal_diff < 0.1 and iteration > min_iterations * 2:
        return True, f"Poor win/loss discrimination (diff={cal_diff:.4f})"

    return False, "Value network appears healthy"


__all__ = [
    "ValueCalibrationTracker",
    "check_cold_start_failure",
    "diagnose_value_network",
    "extract_values_from_trajectories",
    "log_value_diagnostics",
]
