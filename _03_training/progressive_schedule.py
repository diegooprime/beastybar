"""Progressive MCTS simulation scheduling based on training progress.

This module implements adaptive MCTS simulation counts that scale with training
progress. The intuition is:
- Early training: weak value network, fewer simulations suffice (saves compute)
- Late training: strong value network, more simulations refine the policy

Schedule types:
- Linear: Gradually increase simulations from min to max over N iterations
- WinRate-based: Scale simulations based on win rate milestones
- Step: Discrete jumps at win rate thresholds

Example usage:
    schedule = ProgressiveMCTSSchedule.from_config({
        "type": "winrate",
        "min_simulations": 100,
        "max_simulations": 400,
        "milestones": {0.50: 200, 0.70: 400},
    })

    # During training loop
    num_sims = schedule.get_num_simulations(iteration=50, win_rate=0.55)
    schedule.update(iteration=50, metrics={"win_rate": 0.55})

Integration with MCTSTrainer:
    # In mcts_trainer.py, modify _generate_mcts_games:
    current_sims = self.schedule.get_num_simulations(
        iteration=self._iteration,
        win_rate=self._current_win_rate,
    )
    modified_config = dataclasses.replace(
        self.config.mcts_config,
        num_simulations=current_sims,
    )
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of progressive MCTS schedules."""

    LINEAR = "linear"
    WINRATE = "winrate"
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"


@dataclass
class ScheduleState:
    """Mutable state for schedule tracking.

    Attributes:
        current_simulations: Current number of simulations.
        last_iteration: Last iteration when update was called.
        last_win_rate: Last recorded win rate.
        peak_win_rate: Highest win rate achieved.
        milestone_history: History of milestone achievements.
    """

    current_simulations: int = 100
    last_iteration: int = 0
    last_win_rate: float = 0.0
    peak_win_rate: float = 0.0
    milestone_history: list[tuple[int, float, int]] = field(default_factory=list)


class BaseSchedule(ABC):
    """Abstract base class for MCTS simulation schedules."""

    @abstractmethod
    def get_num_simulations(self, iteration: int, win_rate: float) -> int:
        """Get number of MCTS simulations for current training state.

        Args:
            iteration: Current training iteration.
            win_rate: Current win rate against evaluation opponents.

        Returns:
            Number of MCTS simulations to use.
        """
        pass

    @abstractmethod
    def update(self, iteration: int, metrics: dict[str, float]) -> None:
        """Update schedule state based on training metrics.

        Args:
            iteration: Current training iteration.
            metrics: Dictionary of training metrics (may include win_rate, loss, etc.).
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize schedule configuration to dictionary."""
        pass

    @abstractmethod
    def get_state(self) -> ScheduleState:
        """Get current schedule state for checkpointing."""
        pass

    @abstractmethod
    def set_state(self, state: ScheduleState) -> None:
        """Restore schedule state from checkpoint."""
        pass


class LinearSchedule(BaseSchedule):
    """Linear interpolation from min to max simulations over N iterations.

    Simulations increase linearly regardless of win rate:
        sims = min_sims + (max_sims - min_sims) * (iteration / total_iterations)

    Attributes:
        min_simulations: Starting number of simulations.
        max_simulations: Target number of simulations.
        total_iterations: Number of iterations to reach max.
        warmup_iterations: Iterations to stay at min before scaling.
    """

    def __init__(
        self,
        min_simulations: int = 100,
        max_simulations: int = 400,
        total_iterations: int = 200,
        warmup_iterations: int = 0,
    ) -> None:
        if min_simulations <= 0:
            raise ValueError(f"min_simulations must be positive, got {min_simulations}")
        if max_simulations < min_simulations:
            raise ValueError(f"max_simulations ({max_simulations}) must be >= min_simulations ({min_simulations})")
        if total_iterations <= 0:
            raise ValueError(f"total_iterations must be positive, got {total_iterations}")

        self.min_simulations = min_simulations
        self.max_simulations = max_simulations
        self.total_iterations = total_iterations
        self.warmup_iterations = warmup_iterations
        self._state = ScheduleState(current_simulations=min_simulations)

    def get_num_simulations(self, iteration: int, win_rate: float = 0.0) -> int:
        """Get linearly interpolated simulation count."""
        # Stay at min during warmup
        if iteration < self.warmup_iterations:
            return self.min_simulations

        # Adjust iteration for warmup
        effective_iteration = iteration - self.warmup_iterations
        effective_total = self.total_iterations - self.warmup_iterations

        # Calculate progress
        progress = min(1.0, effective_iteration / max(1, effective_total))

        # Linear interpolation
        sim_range = self.max_simulations - self.min_simulations
        current_sims = self.min_simulations + int(sim_range * progress)

        self._state.current_simulations = current_sims
        return current_sims

    def update(self, iteration: int, metrics: dict[str, float]) -> None:
        """Update state (linear schedule doesn't need metrics)."""
        self._state.last_iteration = iteration
        if "win_rate" in metrics:
            self._state.last_win_rate = metrics["win_rate"]
            self._state.peak_win_rate = max(self._state.peak_win_rate, metrics["win_rate"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "linear",
            "min_simulations": self.min_simulations,
            "max_simulations": self.max_simulations,
            "total_iterations": self.total_iterations,
            "warmup_iterations": self.warmup_iterations,
        }

    def get_state(self) -> ScheduleState:
        """Get current state."""
        return self._state

    def set_state(self, state: ScheduleState) -> None:
        """Restore state."""
        self._state = state


class WinRateSchedule(BaseSchedule):
    """Scale simulations based on win rate milestones with smooth interpolation.

    Simulations scale smoothly between milestones:
        - Below first milestone: use min_simulations
        - Between milestones: linear interpolation
        - Above last milestone: use max_simulations

    Example milestones: {0.50: 200, 0.70: 400}
        - WR < 50%: 100 simulations
        - WR = 60%: 300 simulations (interpolated)
        - WR >= 70%: 400 simulations

    Attributes:
        min_simulations: Base simulations before first milestone.
        max_simulations: Maximum simulations after last milestone.
        milestones: Dict mapping win rate thresholds to simulation counts.
        use_peak_winrate: If True, use peak win rate instead of current.
        smoothing_window: Number of evaluations to average for stability.
    """

    def __init__(
        self,
        min_simulations: int = 100,
        max_simulations: int = 400,
        milestones: dict[float, int] | None = None,
        use_peak_winrate: bool = False,
        smoothing_window: int = 3,
    ) -> None:
        if min_simulations <= 0:
            raise ValueError(f"min_simulations must be positive, got {min_simulations}")
        if max_simulations < min_simulations:
            raise ValueError(f"max_simulations ({max_simulations}) must be >= min_simulations ({min_simulations})")

        self.min_simulations = min_simulations
        self.max_simulations = max_simulations

        # Default milestones if not provided
        if milestones is None:
            milestones = {0.50: 200, 0.70: 400}

        # Validate and sort milestones
        self.milestones = dict(sorted(milestones.items()))
        for wr, sims in self.milestones.items():
            if not 0.0 <= wr <= 1.0:
                raise ValueError(f"Win rate milestone must be in [0, 1], got {wr}")
            if sims < min_simulations:
                raise ValueError(f"Milestone simulation count ({sims}) < min_simulations ({min_simulations})")

        self.use_peak_winrate = use_peak_winrate
        self.smoothing_window = smoothing_window
        self._state = ScheduleState(current_simulations=min_simulations)
        self._recent_winrates: list[float] = []

    def get_num_simulations(self, iteration: int, win_rate: float = 0.0) -> int:
        """Get simulation count based on win rate with interpolation."""
        # Use peak or current win rate
        effective_wr = self._state.peak_win_rate if self.use_peak_winrate else win_rate

        # Apply smoothing if we have history
        if self._recent_winrates:
            smoothed_wr = sum(self._recent_winrates) / len(self._recent_winrates)
            effective_wr = max(effective_wr, smoothed_wr)

        # Find position in milestones
        sorted_milestones = list(self.milestones.items())

        if not sorted_milestones:
            return self.min_simulations

        # Below first milestone
        first_wr, first_sims = sorted_milestones[0]
        if effective_wr < first_wr:
            # Interpolate from min to first milestone
            progress = effective_wr / first_wr if first_wr > 0 else 0
            current_sims = self.min_simulations + int((first_sims - self.min_simulations) * progress)
            self._state.current_simulations = current_sims
            return current_sims

        # Above last milestone
        last_wr, _last_sims = sorted_milestones[-1]
        if effective_wr >= last_wr:
            self._state.current_simulations = self.max_simulations
            return self.max_simulations

        # Between milestones: find the two surrounding milestones and interpolate
        for i in range(len(sorted_milestones) - 1):
            low_wr, low_sims = sorted_milestones[i]
            high_wr, high_sims = sorted_milestones[i + 1]

            if low_wr <= effective_wr < high_wr:
                # Linear interpolation between milestones
                progress = (effective_wr - low_wr) / (high_wr - low_wr)
                current_sims = low_sims + int((high_sims - low_sims) * progress)
                self._state.current_simulations = current_sims
                return current_sims

        # Fallback
        return self.min_simulations

    def update(self, iteration: int, metrics: dict[str, float]) -> None:
        """Update state with new metrics."""
        self._state.last_iteration = iteration

        if "win_rate" in metrics:
            win_rate = metrics["win_rate"]
            self._state.last_win_rate = win_rate

            # Track peak
            if win_rate > self._state.peak_win_rate:
                self._state.peak_win_rate = win_rate
                logger.info(f"New peak win rate: {win_rate:.1%}")

            # Update smoothing window
            self._recent_winrates.append(win_rate)
            if len(self._recent_winrates) > self.smoothing_window:
                self._recent_winrates.pop(0)

            # Check for milestone achievements
            for milestone_wr in self.milestones:
                if win_rate >= milestone_wr:
                    # Check if this milestone was just achieved
                    already_achieved = any(wr >= milestone_wr for _, wr, _ in self._state.milestone_history)
                    if not already_achieved:
                        self._state.milestone_history.append((iteration, win_rate, self.milestones[milestone_wr]))
                        logger.info(
                            f"Milestone achieved: WR {milestone_wr:.0%} -> "
                            f"{self.milestones[milestone_wr]} simulations at iter {iteration}"
                        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "winrate",
            "min_simulations": self.min_simulations,
            "max_simulations": self.max_simulations,
            "milestones": self.milestones,
            "use_peak_winrate": self.use_peak_winrate,
            "smoothing_window": self.smoothing_window,
        }

    def get_state(self) -> ScheduleState:
        """Get current state."""
        return self._state

    def set_state(self, state: ScheduleState) -> None:
        """Restore state."""
        self._state = state


class StepSchedule(BaseSchedule):
    """Discrete step increases at win rate thresholds.

    Unlike WinRateSchedule, this uses discrete jumps (no interpolation):
        - WR < 50%: 100 simulations
        - 50% <= WR < 70%: 200 simulations
        - WR >= 70%: 400 simulations

    Includes hysteresis to prevent oscillation at boundaries:
    - Only increase when exceeding threshold + margin
    - Only decrease when dropping below threshold - margin

    Attributes:
        min_simulations: Base simulations before first threshold.
        thresholds: Dict mapping win rate thresholds to simulation counts.
        hysteresis: Margin around thresholds to prevent oscillation.
        require_consecutive: Number of consecutive evaluations above threshold.
    """

    def __init__(
        self,
        min_simulations: int = 100,
        thresholds: dict[float, int] | None = None,
        hysteresis: float = 0.02,
        require_consecutive: int = 1,
    ) -> None:
        if min_simulations <= 0:
            raise ValueError(f"min_simulations must be positive, got {min_simulations}")

        self.min_simulations = min_simulations

        # Default thresholds if not provided
        if thresholds is None:
            thresholds = {0.50: 200, 0.70: 400}

        self.thresholds = dict(sorted(thresholds.items()))
        self.hysteresis = hysteresis
        self.require_consecutive = require_consecutive

        self._state = ScheduleState(current_simulations=min_simulations)
        self._consecutive_above: dict[float, int] = dict.fromkeys(self.thresholds, 0)
        self._current_level = 0  # Index of current threshold level

    def get_num_simulations(self, iteration: int, win_rate: float = 0.0) -> int:
        """Get simulation count at current step level."""
        return self._state.current_simulations

    def update(self, iteration: int, metrics: dict[str, float]) -> None:
        """Update step level based on win rate with hysteresis."""
        self._state.last_iteration = iteration

        if "win_rate" not in metrics:
            return

        win_rate = metrics["win_rate"]
        self._state.last_win_rate = win_rate
        self._state.peak_win_rate = max(self._state.peak_win_rate, win_rate)

        sorted_thresholds = list(self.thresholds.items())

        # Check for step up
        for i, (threshold_wr, threshold_sims) in enumerate(sorted_thresholds):
            if i < self._current_level:
                continue  # Already above this level

            # Check if we should step up (with hysteresis)
            if win_rate >= threshold_wr + self.hysteresis:
                self._consecutive_above[threshold_wr] += 1

                if self._consecutive_above[threshold_wr] >= self.require_consecutive:
                    self._current_level = i + 1
                    self._state.current_simulations = threshold_sims
                    self._state.milestone_history.append((iteration, win_rate, threshold_sims))
                    logger.info(f"Step up: WR {win_rate:.1%} >= {threshold_wr:.0%} -> {threshold_sims} simulations")
                    # Reset counters for higher thresholds
                    for wr in self._consecutive_above:
                        if wr > threshold_wr:
                            self._consecutive_above[wr] = 0
                    break
            else:
                self._consecutive_above[threshold_wr] = 0

        # Check for step down (only if win rate drops significantly)
        if self._current_level > 0:
            current_threshold = sorted_thresholds[self._current_level - 1][0]
            if win_rate < current_threshold - self.hysteresis * 2:
                # Step down one level
                self._current_level -= 1
                if self._current_level == 0:
                    self._state.current_simulations = self.min_simulations
                else:
                    self._state.current_simulations = sorted_thresholds[self._current_level - 1][1]
                logger.info(
                    f"Step down: WR {win_rate:.1%} < {current_threshold:.0%} -> "
                    f"{self._state.current_simulations} simulations"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "step",
            "min_simulations": self.min_simulations,
            "thresholds": self.thresholds,
            "hysteresis": self.hysteresis,
            "require_consecutive": self.require_consecutive,
        }

    def get_state(self) -> ScheduleState:
        """Get current state."""
        return self._state

    def set_state(self, state: ScheduleState) -> None:
        """Restore state."""
        self._state = state


class ExponentialSchedule(BaseSchedule):
    """Exponential growth from min to max simulations.

    sims = min_sims * (max_sims / min_sims) ^ (iteration / total_iterations)

    Provides slower initial growth and faster later growth compared to linear.
    """

    def __init__(
        self,
        min_simulations: int = 100,
        max_simulations: int = 400,
        total_iterations: int = 200,
        warmup_iterations: int = 0,
    ) -> None:
        if min_simulations <= 0:
            raise ValueError(f"min_simulations must be positive, got {min_simulations}")
        if max_simulations < min_simulations:
            raise ValueError(f"max_simulations ({max_simulations}) must be >= min_simulations ({min_simulations})")

        self.min_simulations = min_simulations
        self.max_simulations = max_simulations
        self.total_iterations = total_iterations
        self.warmup_iterations = warmup_iterations
        self._state = ScheduleState(current_simulations=min_simulations)

    def get_num_simulations(self, iteration: int, win_rate: float = 0.0) -> int:
        """Get exponentially interpolated simulation count."""
        if iteration < self.warmup_iterations:
            return self.min_simulations

        effective_iteration = iteration - self.warmup_iterations
        effective_total = self.total_iterations - self.warmup_iterations

        progress = min(1.0, effective_iteration / max(1, effective_total))

        # Exponential interpolation
        ratio = self.max_simulations / self.min_simulations
        current_sims = int(self.min_simulations * (ratio**progress))

        self._state.current_simulations = min(current_sims, self.max_simulations)
        return self._state.current_simulations

    def update(self, iteration: int, metrics: dict[str, float]) -> None:
        """Update state."""
        self._state.last_iteration = iteration
        if "win_rate" in metrics:
            self._state.last_win_rate = metrics["win_rate"]
            self._state.peak_win_rate = max(self._state.peak_win_rate, metrics["win_rate"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "exponential",
            "min_simulations": self.min_simulations,
            "max_simulations": self.max_simulations,
            "total_iterations": self.total_iterations,
            "warmup_iterations": self.warmup_iterations,
        }

    def get_state(self) -> ScheduleState:
        """Get current state."""
        return self._state

    def set_state(self, state: ScheduleState) -> None:
        """Restore state."""
        self._state = state


class CosineSchedule(BaseSchedule):
    """Cosine annealing from min to max simulations.

    sims = min_sims + (max_sims - min_sims) * (1 - cos(pi * progress)) / 2

    Provides smooth S-curve growth: slow start, fast middle, slow end.
    """

    def __init__(
        self,
        min_simulations: int = 100,
        max_simulations: int = 400,
        total_iterations: int = 200,
        warmup_iterations: int = 0,
    ) -> None:
        if min_simulations <= 0:
            raise ValueError(f"min_simulations must be positive, got {min_simulations}")
        if max_simulations < min_simulations:
            raise ValueError(f"max_simulations ({max_simulations}) must be >= min_simulations ({min_simulations})")

        self.min_simulations = min_simulations
        self.max_simulations = max_simulations
        self.total_iterations = total_iterations
        self.warmup_iterations = warmup_iterations
        self._state = ScheduleState(current_simulations=min_simulations)

    def get_num_simulations(self, iteration: int, win_rate: float = 0.0) -> int:
        """Get cosine-interpolated simulation count."""
        if iteration < self.warmup_iterations:
            return self.min_simulations

        effective_iteration = iteration - self.warmup_iterations
        effective_total = self.total_iterations - self.warmup_iterations

        progress = min(1.0, effective_iteration / max(1, effective_total))

        # Cosine interpolation (0 -> 1 as progress 0 -> 1)
        cosine_progress = (1 - math.cos(math.pi * progress)) / 2
        sim_range = self.max_simulations - self.min_simulations
        current_sims = self.min_simulations + int(sim_range * cosine_progress)

        self._state.current_simulations = current_sims
        return current_sims

    def update(self, iteration: int, metrics: dict[str, float]) -> None:
        """Update state."""
        self._state.last_iteration = iteration
        if "win_rate" in metrics:
            self._state.last_win_rate = metrics["win_rate"]
            self._state.peak_win_rate = max(self._state.peak_win_rate, metrics["win_rate"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "cosine",
            "min_simulations": self.min_simulations,
            "max_simulations": self.max_simulations,
            "total_iterations": self.total_iterations,
            "warmup_iterations": self.warmup_iterations,
        }

    def get_state(self) -> ScheduleState:
        """Get current state."""
        return self._state

    def set_state(self, state: ScheduleState) -> None:
        """Restore state."""
        self._state = state


class ProgressiveMCTSSchedule:
    """Factory and wrapper for progressive MCTS simulation schedules.

    This is the main interface for using progressive schedules in training.
    It wraps the specific schedule implementations and provides a unified API.

    Example:
        # From config dict
        schedule = ProgressiveMCTSSchedule.from_config({
            "type": "winrate",
            "min_simulations": 100,
            "max_simulations": 400,
            "milestones": {0.50: 200, 0.70: 400},
        })

        # From YAML file
        schedule = ProgressiveMCTSSchedule.from_yaml("configs/schedule.yaml")

        # Direct construction
        schedule = ProgressiveMCTSSchedule(
            schedule_type="linear",
            min_simulations=100,
            max_simulations=400,
            total_iterations=200,
        )
    """

    def __init__(
        self,
        schedule_type: str = "linear",
        min_simulations: int = 100,
        max_simulations: int = 400,
        total_iterations: int = 200,
        warmup_iterations: int = 0,
        milestones: dict[float, int] | None = None,
        thresholds: dict[float, int] | None = None,
        use_peak_winrate: bool = False,
        smoothing_window: int = 3,
        hysteresis: float = 0.02,
        require_consecutive: int = 1,
    ) -> None:
        """Initialize progressive schedule.

        Args:
            schedule_type: Type of schedule ("linear", "winrate", "step", "exponential", "cosine").
            min_simulations: Minimum number of simulations.
            max_simulations: Maximum number of simulations.
            total_iterations: Total training iterations (for iteration-based schedules).
            warmup_iterations: Iterations to stay at min before scaling.
            milestones: Win rate -> simulation count mapping (for winrate schedule).
            thresholds: Win rate -> simulation count mapping (for step schedule).
            use_peak_winrate: Use peak instead of current win rate (for winrate schedule).
            smoothing_window: Window size for win rate smoothing (for winrate schedule).
            hysteresis: Margin around thresholds (for step schedule).
            require_consecutive: Consecutive evaluations needed to step up (for step schedule).
        """
        self.schedule_type = schedule_type
        self._schedule: BaseSchedule

        if schedule_type == "linear":
            self._schedule = LinearSchedule(
                min_simulations=min_simulations,
                max_simulations=max_simulations,
                total_iterations=total_iterations,
                warmup_iterations=warmup_iterations,
            )
        elif schedule_type == "winrate":
            self._schedule = WinRateSchedule(
                min_simulations=min_simulations,
                max_simulations=max_simulations,
                milestones=milestones,
                use_peak_winrate=use_peak_winrate,
                smoothing_window=smoothing_window,
            )
        elif schedule_type == "step":
            self._schedule = StepSchedule(
                min_simulations=min_simulations,
                thresholds=thresholds or milestones,
                hysteresis=hysteresis,
                require_consecutive=require_consecutive,
            )
        elif schedule_type == "exponential":
            self._schedule = ExponentialSchedule(
                min_simulations=min_simulations,
                max_simulations=max_simulations,
                total_iterations=total_iterations,
                warmup_iterations=warmup_iterations,
            )
        elif schedule_type == "cosine":
            self._schedule = CosineSchedule(
                min_simulations=min_simulations,
                max_simulations=max_simulations,
                total_iterations=total_iterations,
                warmup_iterations=warmup_iterations,
            )
        else:
            raise ValueError(
                f"Unknown schedule type: {schedule_type}. Must be one of: linear, winrate, step, exponential, cosine"
            )

        logger.info(f"Created {schedule_type} MCTS schedule: {min_simulations} -> {max_simulations} simulations")

    def get_num_simulations(self, iteration: int, win_rate: float = 0.0) -> int:
        """Get number of MCTS simulations for current training state.

        Args:
            iteration: Current training iteration.
            win_rate: Current win rate against evaluation opponents (0-1).

        Returns:
            Number of MCTS simulations to use.
        """
        return self._schedule.get_num_simulations(iteration, win_rate)

    def update(self, iteration: int, metrics: dict[str, float]) -> None:
        """Update schedule state based on training metrics.

        Args:
            iteration: Current training iteration.
            metrics: Dictionary of training metrics. Expected keys:
                - "win_rate": Current win rate (0-1)
                - Optional: "loss", "policy_loss", "value_loss", etc.
        """
        self._schedule.update(iteration, metrics)

    def to_dict(self) -> dict[str, Any]:
        """Serialize schedule configuration to dictionary."""
        return self._schedule.to_dict()

    def get_state(self) -> ScheduleState:
        """Get current schedule state for checkpointing."""
        return self._schedule.get_state()

    def set_state(self, state: ScheduleState) -> None:
        """Restore schedule state from checkpoint."""
        self._schedule.set_state(state)

    @property
    def current_simulations(self) -> int:
        """Get current simulation count."""
        return self._schedule.get_state().current_simulations

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ProgressiveMCTSSchedule:
        """Create schedule from configuration dictionary.

        Args:
            config: Configuration dictionary with keys:
                - type: Schedule type ("linear", "winrate", "step", "exponential", "cosine")
                - min_simulations: Minimum simulations
                - max_simulations: Maximum simulations
                - total_iterations: Total training iterations (for iteration-based)
                - milestones: Win rate milestones (for winrate/step types)
                - Other schedule-specific options

        Returns:
            Configured ProgressiveMCTSSchedule instance.

        Example config:
            {
                "type": "winrate",
                "min_simulations": 100,
                "max_simulations": 400,
                "milestones": {0.50: 200, 0.70: 400},
            }
        """
        schedule_type = config.get("type", "linear")

        # Convert milestones keys from string to float if needed (YAML compatibility)
        milestones = config.get("milestones")
        if milestones is not None:
            milestones = {float(k): int(v) for k, v in milestones.items()}

        thresholds = config.get("thresholds")
        if thresholds is not None:
            thresholds = {float(k): int(v) for k, v in thresholds.items()}

        return cls(
            schedule_type=schedule_type,
            min_simulations=config.get("min_simulations", 100),
            max_simulations=config.get("max_simulations", 400),
            total_iterations=config.get("total_iterations", 200),
            warmup_iterations=config.get("warmup_iterations", 0),
            milestones=milestones,
            thresholds=thresholds,
            use_peak_winrate=config.get("use_peak_winrate", False),
            smoothing_window=config.get("smoothing_window", 3),
            hysteresis=config.get("hysteresis", 0.02),
            require_consecutive=config.get("require_consecutive", 1),
        )

    @classmethod
    def from_yaml(cls, path: str) -> ProgressiveMCTSSchedule:
        """Load schedule from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Configured ProgressiveMCTSSchedule instance.

        Example YAML:
            mcts_schedule:
              type: winrate
              min_simulations: 100
              max_simulations: 400
              milestones:
                0.50: 200
                0.70: 400
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle nested config under "mcts_schedule" key
        config = data.get("mcts_schedule", data)

        return cls.from_config(config)


# Convenience functions for common configurations


def linear_schedule(
    min_sims: int = 100,
    max_sims: int = 400,
    total_iters: int = 200,
) -> ProgressiveMCTSSchedule:
    """Create a linear simulation schedule.

    Args:
        min_sims: Starting simulations.
        max_sims: Target simulations.
        total_iters: Iterations to reach max.

    Returns:
        Linear ProgressiveMCTSSchedule.
    """
    return ProgressiveMCTSSchedule(
        schedule_type="linear",
        min_simulations=min_sims,
        max_simulations=max_sims,
        total_iterations=total_iters,
    )


def winrate_schedule(
    min_sims: int = 100,
    max_sims: int = 400,
    milestones: dict[float, int] | None = None,
) -> ProgressiveMCTSSchedule:
    """Create a win rate-based simulation schedule.

    Args:
        min_sims: Base simulations.
        max_sims: Maximum simulations.
        milestones: Win rate -> simulation count mapping.

    Returns:
        WinRate ProgressiveMCTSSchedule.
    """
    return ProgressiveMCTSSchedule(
        schedule_type="winrate",
        min_simulations=min_sims,
        max_simulations=max_sims,
        milestones=milestones or {0.50: 200, 0.70: 400},
    )


def step_schedule(
    min_sims: int = 100,
    thresholds: dict[float, int] | None = None,
) -> ProgressiveMCTSSchedule:
    """Create a step-based simulation schedule.

    Args:
        min_sims: Base simulations.
        thresholds: Win rate -> simulation count thresholds.

    Returns:
        Step ProgressiveMCTSSchedule.
    """
    return ProgressiveMCTSSchedule(
        schedule_type="step",
        min_simulations=min_sims,
        max_simulations=max(thresholds.values()) if thresholds else 400,
        thresholds=thresholds or {0.50: 200, 0.70: 400},
    )


__all__ = [
    "BaseSchedule",
    "CosineSchedule",
    "ExponentialSchedule",
    "LinearSchedule",
    "ProgressiveMCTSSchedule",
    "ScheduleState",
    "ScheduleType",
    "StepSchedule",
    "WinRateSchedule",
    "linear_schedule",
    "step_schedule",
    "winrate_schedule",
]
