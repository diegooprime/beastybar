"""Experience replay buffer for PPO training.

This module provides efficient storage and sampling of training experiences
for reinforcement learning. It uses pre-allocated numpy arrays with circular
indexing for O(1) add operations and constant memory usage.

Example:
    buffer = ReplayBuffer(max_size=100000)

    # Add single transition
    buffer.add(Transition(obs, mask, action, prob, value, reward, done))

    # Add complete game trajectory
    buffer.add_trajectory([t1, t2, t3, ...])

    # Sample for training
    batch = buffer.sample(batch_size=256)

    # Get all data for PPO epochs
    all_data = buffer.sample_all()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Transition:
    """Single experience tuple for PPO training.

    Attributes:
        observation: State observation tensor of shape (OBSERVATION_DIM,).
        action_mask: Legal action mask of shape (ACTION_DIM,).
        action: Selected action index.
        action_prob: Probability of selected action under old policy.
        value: Value estimate from critic at this state.
        reward: Reward received after taking action.
        done: Whether this transition ended the episode.
    """

    observation: NDArray[np.float32]  # (OBSERVATION_DIM,)
    action_mask: NDArray[np.float32]  # (ACTION_DIM,)
    action: int
    action_prob: float
    value: float
    reward: float
    done: bool

    def __post_init__(self) -> None:
        """Validate transition data shapes and types."""
        if self.observation.ndim != 1:
            raise ValueError(f"Observation must be 1D, got shape {self.observation.shape}")
        if self.action_mask.ndim != 1:
            raise ValueError(f"Action mask must be 1D, got shape {self.action_mask.shape}")
        if not isinstance(self.action, int) or self.action < 0:
            raise ValueError(f"Action must be non-negative integer, got {self.action}")
        if not (0.0 <= self.action_prob <= 1.0):
            raise ValueError(f"Action probability must be in [0, 1], got {self.action_prob}")


@dataclass(frozen=True, slots=True)
class Batch:
    """Batched training data for PPO updates.

    All arrays have batch dimension as first axis.

    Attributes:
        observations: State observations of shape (batch, OBSERVATION_DIM).
        action_masks: Legal action masks of shape (batch, ACTION_DIM).
        actions: Selected action indices of shape (batch,).
        action_probs: Action probabilities of shape (batch,).
        values: Value estimates of shape (batch,).
        rewards: Rewards of shape (batch,).
        dones: Episode termination flags of shape (batch,).
    """

    observations: NDArray[np.float32]  # (batch, OBSERVATION_DIM)
    action_masks: NDArray[np.float32]  # (batch, ACTION_DIM)
    actions: NDArray[np.int64]  # (batch,)
    action_probs: NDArray[np.float32]  # (batch,)
    values: NDArray[np.float32]  # (batch,)
    rewards: NDArray[np.float32]  # (batch,)
    dones: NDArray[np.bool_]  # (batch,)

    @property
    def batch_size(self) -> int:
        """Return number of samples in batch."""
        return self.observations.shape[0]

    def __len__(self) -> int:
        """Return number of samples in batch."""
        return self.batch_size


@dataclass(frozen=True, slots=True)
class TrajectoryBoundary:
    """Marks the start and end indices of a complete game trajectory.

    Used for GAE (Generalized Advantage Estimation) computation which
    requires knowing episode boundaries for proper value bootstrapping.

    Attributes:
        start_idx: Inclusive start index in buffer.
        end_idx: Exclusive end index in buffer.
        final_reward: Terminal reward of the trajectory.
    """

    start_idx: int
    end_idx: int
    final_reward: float

    @property
    def length(self) -> int:
        """Return number of transitions in trajectory."""
        return self.end_idx - self.start_idx


class ReplayBuffer:
    """Circular replay buffer for storing PPO training experiences.

    Uses pre-allocated numpy arrays for efficient memory usage and O(1)
    insertion. When full, oldest experiences are overwritten (FIFO).

    Attributes:
        max_size: Maximum number of transitions to store.
        observation_dim: Dimension of observation vectors.
        action_dim: Dimension of action mask vectors.

    Example:
        buffer = ReplayBuffer(max_size=100000, observation_dim=988, action_dim=124)
        buffer.add(transition)
        batch = buffer.sample(256)
    """

    def __init__(
        self,
        max_size: int = 100_000,
        observation_dim: int = 988,
        action_dim: int = 124,
    ) -> None:
        """Initialize replay buffer with pre-allocated arrays.

        Args:
            max_size: Maximum capacity of buffer. Default 100,000.
            observation_dim: Dimension of observation vectors. Default 988.
            action_dim: Dimension of action mask vectors. Default 124.

        Raises:
            ValueError: If any dimension is non-positive.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.max_size = max_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Pre-allocate storage arrays
        self._observations = np.zeros((max_size, observation_dim), dtype=np.float32)
        self._action_masks = np.zeros((max_size, action_dim), dtype=np.float32)
        self._actions = np.zeros(max_size, dtype=np.int64)
        self._action_probs = np.zeros(max_size, dtype=np.float32)
        self._values = np.zeros(max_size, dtype=np.float32)
        self._rewards = np.zeros(max_size, dtype=np.float32)
        self._dones = np.zeros(max_size, dtype=np.bool_)

        # Circular buffer state
        self._position = 0  # Next write position
        self._size = 0  # Current number of valid entries
        self._total_added = 0  # Total transitions ever added

        # Trajectory tracking
        self._trajectory_boundaries: list[TrajectoryBoundary] = []

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Return True if buffer is at capacity."""
        return self._size == self.max_size

    @property
    def total_added(self) -> int:
        """Return total number of transitions ever added to buffer."""
        return self._total_added

    def add(self, transition: Transition) -> None:
        """Add a single transition to the buffer.

        Uses circular indexing - when buffer is full, oldest entry is overwritten.

        Args:
            transition: Experience tuple to store.

        Raises:
            ValueError: If transition dimensions don't match buffer configuration.
        """
        if transition.observation.shape[0] != self.observation_dim:
            raise ValueError(
                f"Observation dimension mismatch: expected {self.observation_dim}, "
                f"got {transition.observation.shape[0]}"
            )
        if transition.action_mask.shape[0] != self.action_dim:
            raise ValueError(
                f"Action mask dimension mismatch: expected {self.action_dim}, "
                f"got {transition.action_mask.shape[0]}"
            )

        idx = self._position

        self._observations[idx] = transition.observation
        self._action_masks[idx] = transition.action_mask
        self._actions[idx] = transition.action
        self._action_probs[idx] = transition.action_prob
        self._values[idx] = transition.value
        self._rewards[idx] = transition.reward
        self._dones[idx] = transition.done

        # Update circular buffer state
        self._position = (self._position + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)
        self._total_added += 1

    def add_batch(self, transitions: list[Transition]) -> None:
        """Add multiple transitions to the buffer.

        More efficient than calling add() repeatedly when adding many
        transitions at once.

        Args:
            transitions: List of experience tuples to store.
        """
        if not transitions:
            return

        for transition in transitions:
            self.add(transition)

    def add_trajectory(self, trajectory: list[Transition]) -> None:
        """Add a complete game trajectory and track its boundaries.

        Trajectories are tracked for GAE computation which requires
        episode boundaries. When the buffer wraps around, old trajectory
        boundaries are automatically pruned.

        Args:
            trajectory: Complete sequence of transitions from one game.
                Should end with a transition where done=True.
        """
        if not trajectory:
            return

        start_idx = self._position
        self.add_batch(trajectory)
        end_idx = self._position

        # Handle wrap-around case
        if end_idx <= start_idx and self._size == self.max_size:
            # Trajectory wrapped around buffer - boundaries become invalid
            # Clear old boundaries that may have been overwritten
            self._prune_stale_boundaries()
        else:
            final_reward = trajectory[-1].reward
            boundary = TrajectoryBoundary(
                start_idx=start_idx,
                end_idx=end_idx,
                final_reward=final_reward,
            )
            self._trajectory_boundaries.append(boundary)
            self._prune_stale_boundaries()

    def _prune_stale_boundaries(self) -> None:
        """Remove trajectory boundaries that have been overwritten."""
        if not self._trajectory_boundaries:
            return

        # When buffer wraps, trajectory boundaries become unreliable
        # due to circular indexing complexity. Clear all on wrap.
        if self._size == self.max_size:
            self._trajectory_boundaries.clear()
            return

        # Keep only boundaries where all indices are still valid
        valid_boundaries = []
        for boundary in self._trajectory_boundaries:
            # Check if boundary indices are still in valid range
            if self._size < self.max_size:
                # Buffer hasn't wrapped yet - all indices valid
                valid_boundaries.append(boundary)

        self._trajectory_boundaries = valid_boundaries

    def sample(self, batch_size: int) -> Batch:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batch of randomly sampled transitions.

        Raises:
            ValueError: If batch_size exceeds buffer size.
            ValueError: If buffer is empty.
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        if batch_size > self._size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({self._size})"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        indices = np.random.choice(self._size, size=batch_size, replace=False)

        return Batch(
            observations=self._observations[indices].copy(),
            action_masks=self._action_masks[indices].copy(),
            actions=self._actions[indices].copy(),
            action_probs=self._action_probs[indices].copy(),
            values=self._values[indices].copy(),
            rewards=self._rewards[indices].copy(),
            dones=self._dones[indices].copy(),
        )

    def sample_all(self) -> Batch:
        """Return all transitions in buffer as a single batch.

        Useful for PPO which iterates over all collected experiences
        multiple times per update.

        Returns:
            Batch containing all stored transitions.

        Raises:
            ValueError: If buffer is empty.
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")

        return Batch(
            observations=self._observations[: self._size].copy(),
            action_masks=self._action_masks[: self._size].copy(),
            actions=self._actions[: self._size].copy(),
            action_probs=self._action_probs[: self._size].copy(),
            values=self._values[: self._size].copy(),
            rewards=self._rewards[: self._size].copy(),
            dones=self._dones[: self._size].copy(),
        )

    def sample_recent(self, count: int) -> Batch:
        """Sample the most recently added transitions.

        Useful for on-policy algorithms that only train on fresh data.

        Args:
            count: Number of most recent transitions to return.

        Returns:
            Batch of most recent transitions.

        Raises:
            ValueError: If count exceeds buffer size or is non-positive.
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        if count > self._size:
            raise ValueError(f"count ({count}) exceeds buffer size ({self._size})")
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        # Calculate indices for most recent entries
        if self._size < self.max_size:
            # Buffer hasn't wrapped - recent items are at end
            start = self._size - count
            indices = np.arange(start, self._size)
        else:
            # Buffer has wrapped - recent items end at _position-1
            indices = np.arange(count)
            indices = (self._position - count + indices) % self.max_size

        return Batch(
            observations=self._observations[indices].copy(),
            action_masks=self._action_masks[indices].copy(),
            actions=self._actions[indices].copy(),
            action_probs=self._action_probs[indices].copy(),
            values=self._values[indices].copy(),
            rewards=self._rewards[indices].copy(),
            dones=self._dones[indices].copy(),
        )

    def get_trajectory_boundaries(self) -> list[TrajectoryBoundary]:
        """Return list of tracked trajectory boundaries.

        Note: Boundaries become invalid when buffer wraps around and
        overwrites old data.

        Returns:
            List of TrajectoryBoundary objects for complete games.
        """
        return self._trajectory_boundaries.copy()

    def clear(self) -> None:
        """Reset buffer to empty state.

        Clears all stored data and resets position counters.
        Pre-allocated arrays are retained for efficiency.
        """
        self._position = 0
        self._size = 0
        # Note: total_added is NOT reset - it tracks lifetime additions
        self._trajectory_boundaries.clear()

        # Zero out arrays for clean state (optional but good for debugging)
        self._observations.fill(0)
        self._action_masks.fill(0)
        self._actions.fill(0)
        self._action_probs.fill(0)
        self._values.fill(0)
        self._rewards.fill(0)
        self._dones.fill(False)

    def stats(self) -> dict[str, float | int]:
        """Compute buffer statistics.

        Returns:
            Dictionary with:
                - size: Current number of transitions
                - max_size: Maximum capacity
                - total_added: Lifetime transitions added
                - utilization: size / max_size
                - num_trajectories: Number of tracked complete games
                - reward_mean: Mean reward (if buffer non-empty)
                - reward_std: Reward standard deviation (if buffer non-empty)
                - reward_min: Minimum reward (if buffer non-empty)
                - reward_max: Maximum reward (if buffer non-empty)
                - done_ratio: Fraction of transitions that are terminal
        """
        result: dict[str, float | int] = {
            "size": self._size,
            "max_size": self.max_size,
            "total_added": self._total_added,
            "utilization": self._size / self.max_size if self.max_size > 0 else 0.0,
            "num_trajectories": len(self._trajectory_boundaries),
        }

        if self._size > 0:
            rewards = self._rewards[: self._size]
            result["reward_mean"] = float(np.mean(rewards))
            result["reward_std"] = float(np.std(rewards))
            result["reward_min"] = float(np.min(rewards))
            result["reward_max"] = float(np.max(rewards))

            dones = self._dones[: self._size]
            result["done_ratio"] = float(np.mean(dones))
        else:
            result["reward_mean"] = 0.0
            result["reward_std"] = 0.0
            result["reward_min"] = 0.0
            result["reward_max"] = 0.0
            result["done_ratio"] = 0.0

        return result


__all__ = [
    "Batch",
    "ReplayBuffer",
    "TrajectoryBoundary",
    "Transition",
]
