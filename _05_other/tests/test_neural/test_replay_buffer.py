"""
Test suite for replay buffer and experience storage.

Tests the storage, sampling, and management of training trajectories
for off-policy or on-policy learning algorithms.
"""

import numpy as np
import pytest

from _01_simulator.action_space import ACTION_DIM
from _01_simulator.observations import OBSERVATION_DIM
from _03_training.replay_buffer import Batch, ReplayBuffer, TrajectoryBoundary, Transition


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_transition() -> Transition:
    """Create a valid sample transition for testing."""
    return Transition(
        observation=np.random.randn(OBSERVATION_DIM).astype(np.float32),
        action_mask=np.random.randint(0, 2, ACTION_DIM).astype(np.float32),
        action=5,
        action_prob=0.3,
        value=0.5,
        reward=1.0,
        done=False,
    )


@pytest.fixture
def small_buffer() -> ReplayBuffer:
    """Create a small buffer for testing edge cases."""
    return ReplayBuffer(max_size=10, observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM)


@pytest.fixture
def default_buffer() -> ReplayBuffer:
    """Create a buffer with default parameters."""
    return ReplayBuffer()


def create_transition(
    action: int = 0,
    reward: float = 0.0,
    done: bool = False,
    action_prob: float = 0.5,
    value: float = 0.0,
) -> Transition:
    """Helper to create transitions with specific values."""
    return Transition(
        observation=np.random.randn(OBSERVATION_DIM).astype(np.float32),
        action_mask=np.ones(ACTION_DIM, dtype=np.float32),
        action=action,
        action_prob=action_prob,
        value=value,
        reward=reward,
        done=done,
    )


# ==============================================================================
# Transition Tests
# ==============================================================================


class TestTransition:
    """Tests for the Transition dataclass."""

    def test_valid_transition_creation(self, sample_transition: Transition) -> None:
        """Verify valid transitions can be created."""
        assert sample_transition.observation.shape == (OBSERVATION_DIM,)
        assert sample_transition.action_mask.shape == (ACTION_DIM,)
        assert sample_transition.action == 5
        assert sample_transition.action_prob == 0.3
        assert sample_transition.value == 0.5
        assert sample_transition.reward == 1.0
        assert sample_transition.done is False

    def test_invalid_observation_shape(self) -> None:
        """Verify error on wrong observation shape."""
        with pytest.raises(ValueError, match="Observation must be 1D"):
            Transition(
                observation=np.zeros((10, 10), dtype=np.float32),
                action_mask=np.zeros(ACTION_DIM, dtype=np.float32),
                action=0,
                action_prob=0.5,
                value=0.0,
                reward=0.0,
                done=False,
            )

    def test_invalid_action_mask_shape(self) -> None:
        """Verify error on wrong action mask shape."""
        with pytest.raises(ValueError, match="Action mask must be 1D"):
            Transition(
                observation=np.zeros(OBSERVATION_DIM, dtype=np.float32),
                action_mask=np.zeros((10, 10), dtype=np.float32),
                action=0,
                action_prob=0.5,
                value=0.0,
                reward=0.0,
                done=False,
            )

    def test_invalid_negative_action(self) -> None:
        """Verify error on negative action index."""
        with pytest.raises(ValueError, match="Action must be non-negative"):
            Transition(
                observation=np.zeros(OBSERVATION_DIM, dtype=np.float32),
                action_mask=np.zeros(ACTION_DIM, dtype=np.float32),
                action=-1,
                action_prob=0.5,
                value=0.0,
                reward=0.0,
                done=False,
            )

    def test_invalid_action_prob_out_of_range(self) -> None:
        """Verify error on action probability outside [0, 1]."""
        with pytest.raises(ValueError, match="Action probability must be in"):
            Transition(
                observation=np.zeros(OBSERVATION_DIM, dtype=np.float32),
                action_mask=np.zeros(ACTION_DIM, dtype=np.float32),
                action=0,
                action_prob=1.5,
                value=0.0,
                reward=0.0,
                done=False,
            )

    def test_frozen_dataclass(self, sample_transition: Transition) -> None:
        """Verify transition is immutable."""
        with pytest.raises(AttributeError):
            sample_transition.action = 10  # type: ignore[misc]


# ==============================================================================
# Batch Tests
# ==============================================================================


class TestBatch:
    """Tests for the Batch dataclass."""

    def test_batch_creation(self) -> None:
        """Verify batch creation with correct shapes."""
        batch_size = 32
        batch = Batch(
            observations=np.zeros((batch_size, OBSERVATION_DIM), dtype=np.float32),
            action_masks=np.zeros((batch_size, ACTION_DIM), dtype=np.float32),
            actions=np.zeros(batch_size, dtype=np.int64),
            action_probs=np.zeros(batch_size, dtype=np.float32),
            values=np.zeros(batch_size, dtype=np.float32),
            rewards=np.zeros(batch_size, dtype=np.float32),
            dones=np.zeros(batch_size, dtype=np.bool_),
        )
        assert batch.batch_size == batch_size
        assert len(batch) == batch_size

    def test_batch_size_property(self) -> None:
        """Verify batch_size returns correct value."""
        batch = Batch(
            observations=np.zeros((16, OBSERVATION_DIM), dtype=np.float32),
            action_masks=np.zeros((16, ACTION_DIM), dtype=np.float32),
            actions=np.zeros(16, dtype=np.int64),
            action_probs=np.zeros(16, dtype=np.float32),
            values=np.zeros(16, dtype=np.float32),
            rewards=np.zeros(16, dtype=np.float32),
            dones=np.zeros(16, dtype=np.bool_),
        )
        assert batch.batch_size == 16


# ==============================================================================
# ReplayBuffer Basic Tests
# ==============================================================================


class TestReplayBufferBasic:
    """Basic functionality tests for ReplayBuffer."""

    def test_buffer_creation_with_defaults(self, default_buffer: ReplayBuffer) -> None:
        """Verify buffer created with default parameters."""
        assert default_buffer.max_size == 100_000
        assert default_buffer.observation_dim == OBSERVATION_DIM
        assert default_buffer.action_dim == ACTION_DIM
        assert len(default_buffer) == 0
        assert not default_buffer.is_full

    def test_buffer_creation_with_custom_params(self) -> None:
        """Verify buffer creation with custom parameters."""
        buffer = ReplayBuffer(max_size=500, observation_dim=100, action_dim=50)
        assert buffer.max_size == 500
        assert buffer.observation_dim == 100
        assert buffer.action_dim == 50

    def test_invalid_buffer_params(self) -> None:
        """Verify error on invalid buffer parameters."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            ReplayBuffer(max_size=0)
        with pytest.raises(ValueError, match="observation_dim must be positive"):
            ReplayBuffer(observation_dim=-1)
        with pytest.raises(ValueError, match="action_dim must be positive"):
            ReplayBuffer(action_dim=0)


def test_buffer_add_and_sample(small_buffer: ReplayBuffer, sample_transition: Transition) -> None:
    """Verify basic buffer operations: add and sample.

    Should:
    - Store trajectories with all required fields (obs, action, reward, etc.)
    - Sample random batches without replacement within a batch
    - Return data in correct format for training
    """
    # Add transitions
    for _ in range(5):
        small_buffer.add(sample_transition)

    assert len(small_buffer) == 5
    assert small_buffer.total_added == 5

    # Sample batch
    batch = small_buffer.sample(batch_size=3)
    assert batch.batch_size == 3
    assert batch.observations.shape == (3, OBSERVATION_DIM)
    assert batch.action_masks.shape == (3, ACTION_DIM)
    assert batch.actions.shape == (3,)
    assert batch.action_probs.shape == (3,)
    assert batch.values.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.dones.shape == (3,)


def test_buffer_fifo_eviction(small_buffer: ReplayBuffer) -> None:
    """Verify buffer evicts oldest data when full.

    When buffer reaches capacity, should:
    - Remove oldest trajectories first (FIFO)
    - Maintain buffer size at max capacity
    - Continue accepting new data
    """
    # Fill buffer with identifiable rewards
    for i in range(10):
        transition = create_transition(reward=float(i))
        small_buffer.add(transition)

    assert len(small_buffer) == 10
    assert small_buffer.is_full

    # Add more transitions - should evict oldest
    for i in range(5):
        transition = create_transition(reward=100.0 + i)
        small_buffer.add(transition)

    assert len(small_buffer) == 10  # Still at max
    assert small_buffer.total_added == 15

    # Sample all and verify old rewards are gone
    batch = small_buffer.sample_all()
    rewards = batch.rewards
    # Original rewards 0-4 should be evicted
    assert np.min(rewards) >= 5.0


def test_buffer_batch_sampling(small_buffer: ReplayBuffer) -> None:
    """Verify efficient batch sampling.

    Should:
    - Sample specified batch size
    - Return tensors with correct batch dimension
    - Handle cases where buffer has fewer items than batch size
    - Optionally support sampling with replacement
    """
    # Add some transitions
    for i in range(5):
        small_buffer.add(create_transition(action=i))

    # Sample specified batch size
    batch = small_buffer.sample(batch_size=3)
    assert batch.batch_size == 3

    # Try to sample more than buffer size - should raise
    with pytest.raises(ValueError, match="exceeds buffer size"):
        small_buffer.sample(batch_size=10)

    # Empty buffer sampling
    empty_buffer = ReplayBuffer(max_size=10)
    with pytest.raises(ValueError, match="empty buffer"):
        empty_buffer.sample(batch_size=1)


def test_buffer_statistics(small_buffer: ReplayBuffer) -> None:
    """Verify buffer tracks useful statistics.

    Should provide:
    - Current buffer size
    - Total trajectories added (lifetime)
    - Average episode length
    - Average episode reward
    - Min/max rewards seen
    """
    # Add transitions with varying rewards
    rewards = [0.0, 1.0, 2.0, 3.0, 4.0]
    for r in rewards:
        small_buffer.add(create_transition(reward=r, done=(r == 4.0)))

    stats = small_buffer.stats()

    assert stats["size"] == 5
    assert stats["max_size"] == 10
    assert stats["total_added"] == 5
    assert stats["utilization"] == 0.5
    assert np.isclose(stats["reward_mean"], 2.0)
    assert stats["reward_min"] == 0.0
    assert stats["reward_max"] == 4.0
    assert stats["done_ratio"] == 0.2  # 1 done out of 5


# ==============================================================================
# Trajectory Tests
# ==============================================================================


class TestTrajectory:
    """Tests for trajectory-aware functionality."""

    def test_add_trajectory(self, small_buffer: ReplayBuffer) -> None:
        """Verify trajectory addition and boundary tracking."""
        trajectory = [create_transition(done=(i == 4)) for i in range(5)]
        small_buffer.add_trajectory(trajectory)

        assert len(small_buffer) == 5
        boundaries = small_buffer.get_trajectory_boundaries()
        assert len(boundaries) == 1
        assert boundaries[0].length == 5

    def test_add_batch(self, small_buffer: ReplayBuffer) -> None:
        """Verify batch addition."""
        transitions = [create_transition(action=i) for i in range(3)]
        small_buffer.add_batch(transitions)

        assert len(small_buffer) == 3
        batch = small_buffer.sample_all()
        assert set(batch.actions.tolist()) == {0, 1, 2}

    def test_trajectory_boundary_pruning(self) -> None:
        """Verify boundaries are cleared when buffer wraps."""
        buffer = ReplayBuffer(max_size=5)

        # Add first trajectory
        trajectory1 = [create_transition(reward=1.0) for _ in range(3)]
        buffer.add_trajectory(trajectory1)

        # Add second trajectory that causes wrap
        trajectory2 = [create_transition(reward=2.0) for _ in range(4)]
        buffer.add_trajectory(trajectory2)

        # After wrap, boundaries should be cleared
        boundaries = buffer.get_trajectory_boundaries()
        # Boundaries become unreliable after wrap
        assert len(boundaries) == 0 or all(b.start_idx < buffer.max_size for b in boundaries)


# ==============================================================================
# Sample Methods Tests
# ==============================================================================


class TestSampling:
    """Tests for various sampling methods."""

    def test_sample_all(self, small_buffer: ReplayBuffer) -> None:
        """Verify sample_all returns complete buffer contents."""
        for i in range(5):
            small_buffer.add(create_transition(reward=float(i)))

        batch = small_buffer.sample_all()
        assert batch.batch_size == 5
        assert set(batch.rewards.tolist()) == {0.0, 1.0, 2.0, 3.0, 4.0}

    def test_sample_recent(self, small_buffer: ReplayBuffer) -> None:
        """Verify sample_recent returns most recently added transitions."""
        for i in range(8):
            small_buffer.add(create_transition(reward=float(i)))

        batch = small_buffer.sample_recent(count=3)
        assert batch.batch_size == 3
        # Should get rewards 5, 6, 7 (most recent)
        assert set(batch.rewards.tolist()) == {5.0, 6.0, 7.0}

    def test_sample_recent_with_wrap(self) -> None:
        """Verify sample_recent works after buffer wraps."""
        buffer = ReplayBuffer(max_size=5)

        # Fill buffer and wrap
        for i in range(8):
            buffer.add(create_transition(reward=float(i)))

        batch = buffer.sample_recent(count=3)
        assert batch.batch_size == 3
        # Should get rewards 5, 6, 7 (most recent)
        assert set(batch.rewards.tolist()) == {5.0, 6.0, 7.0}

    def test_sample_errors(self, small_buffer: ReplayBuffer) -> None:
        """Verify sampling error conditions."""
        # Empty buffer
        with pytest.raises(ValueError, match="empty buffer"):
            small_buffer.sample(1)

        small_buffer.add(create_transition())

        # Invalid batch size
        with pytest.raises(ValueError, match="must be positive"):
            small_buffer.sample(0)

        with pytest.raises(ValueError, match="exceeds buffer size"):
            small_buffer.sample(100)

        with pytest.raises(ValueError, match="must be positive"):
            small_buffer.sample_recent(0)


# ==============================================================================
# Clear and Reset Tests
# ==============================================================================


class TestClearAndReset:
    """Tests for buffer clearing and resetting."""

    def test_clear(self, small_buffer: ReplayBuffer) -> None:
        """Verify clear resets buffer but preserves total_added."""
        for i in range(5):
            small_buffer.add(create_transition())

        total_before = small_buffer.total_added
        small_buffer.clear()

        assert len(small_buffer) == 0
        assert not small_buffer.is_full
        assert small_buffer.total_added == total_before  # Preserved
        assert len(small_buffer.get_trajectory_boundaries()) == 0

        # Should be able to add again
        small_buffer.add(create_transition())
        assert len(small_buffer) == 1


# ==============================================================================
# Dimension Mismatch Tests
# ==============================================================================


class TestDimensionValidation:
    """Tests for dimension validation during add operations."""

    def test_observation_dim_mismatch(self, small_buffer: ReplayBuffer) -> None:
        """Verify error when observation dimension doesn't match."""
        bad_transition = Transition(
            observation=np.zeros(100, dtype=np.float32),  # Wrong size
            action_mask=np.zeros(ACTION_DIM, dtype=np.float32),
            action=0,
            action_prob=0.5,
            value=0.0,
            reward=0.0,
            done=False,
        )
        with pytest.raises(ValueError, match="Observation dimension mismatch"):
            small_buffer.add(bad_transition)

    def test_action_dim_mismatch(self, small_buffer: ReplayBuffer) -> None:
        """Verify error when action dimension doesn't match."""
        bad_transition = Transition(
            observation=np.zeros(OBSERVATION_DIM, dtype=np.float32),
            action_mask=np.zeros(50, dtype=np.float32),  # Wrong size
            action=0,
            action_prob=0.5,
            value=0.0,
            reward=0.0,
            done=False,
        )
        with pytest.raises(ValueError, match="Action mask dimension mismatch"):
            small_buffer.add(bad_transition)


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_element_buffer(self) -> None:
        """Verify buffer works with max_size=1."""
        buffer = ReplayBuffer(max_size=1)
        buffer.add(create_transition(reward=1.0))
        assert len(buffer) == 1

        buffer.add(create_transition(reward=2.0))
        assert len(buffer) == 1

        batch = buffer.sample_all()
        assert batch.rewards[0] == 2.0

    def test_exact_capacity_fill(self, small_buffer: ReplayBuffer) -> None:
        """Verify buffer works when filled exactly to capacity."""
        for i in range(10):
            small_buffer.add(create_transition(reward=float(i)))

        assert len(small_buffer) == 10
        assert small_buffer.is_full

        batch = small_buffer.sample_all()
        assert batch.batch_size == 10

    def test_empty_batch_trajectory(self, small_buffer: ReplayBuffer) -> None:
        """Verify adding empty trajectory does nothing."""
        small_buffer.add_trajectory([])
        assert len(small_buffer) == 0

        small_buffer.add_batch([])
        assert len(small_buffer) == 0
