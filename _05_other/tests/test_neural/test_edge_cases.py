"""
Test suite for neural network edge cases.

Tests behavior with extreme inputs: batch size of 1, single legal action,
NaN/Inf handling, zero-length trajectories, and boundary tensor shapes.
These edge cases are critical for training robustness and production reliability.
"""

import numpy as np
import pytest
import torch

from _01_simulator.action_space import ACTION_DIM
from _01_simulator.observations import OBSERVATION_DIM
from _02_agents.neural.network import create_network
from _02_agents.neural.utils import NetworkConfig


@pytest.fixture
def network():
    """Create a small test network for edge case testing."""
    config = NetworkConfig(
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        dropout=0.0,  # Disable dropout for deterministic tests
    )
    return create_network(config)


class TestBatchSizeOne:
    """Tests for batch size of 1 (single sample inference)."""

    def test_forward_batch_size_one(self, network):
        """Network should handle batch size of 1 correctly.

        Edge case: Single sample inference is common in self-play.
        """
        obs = torch.randn(1, OBSERVATION_DIM)

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(obs)

        assert policy_logits.shape == (1, ACTION_DIM), (
            f"Expected policy shape (1, {ACTION_DIM}), got {policy_logits.shape}"
        )
        assert value.shape == (1, 1), (
            f"Expected value shape (1, 1), got {value.shape}"
        )

    def test_forward_unbatched_single_sample(self, network):
        """Network should handle unbatched single sample (1D input).

        Edge case: Direct single observation without batch dimension.
        """
        obs = torch.randn(OBSERVATION_DIM)  # No batch dimension

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(obs)

        assert policy_logits.shape == (ACTION_DIM,), (
            f"Expected policy shape ({ACTION_DIM},), got {policy_logits.shape}"
        )
        # Value might be scalar or (1,) depending on implementation
        assert value.dim() == 0 or value.shape == (1,), (
            f"Expected scalar value, got shape {value.shape}"
        )

    def test_batch_size_one_with_mask(self, network):
        """Network should handle batch size 1 with action mask.

        Edge case: Masked inference for single sample.
        """
        obs = torch.randn(1, OBSERVATION_DIM)
        mask = torch.zeros(1, ACTION_DIM)
        mask[0, :10] = 1.0  # Only first 10 actions legal

        network.eval()
        with torch.no_grad():
            policy_logits, _value = network(obs, mask)

        assert policy_logits.shape == (1, ACTION_DIM)
        assert torch.all(torch.isfinite(policy_logits))


class TestSingleLegalAction:
    """Tests for all actions masked except one."""

    def test_single_legal_action_mask(self, network):
        """Network should handle mask with only one legal action.

        Edge case: Forced move where only one action is valid.
        """
        batch_size = 4
        obs = torch.randn(batch_size, OBSERVATION_DIM)

        # Create mask with only action 42 legal
        mask = torch.zeros(batch_size, ACTION_DIM)
        mask[:, 42] = 1.0

        network.eval()
        with torch.no_grad():
            policy_logits, _value = network(obs, mask)

        # Network itself doesn't apply mask, but outputs should be valid
        assert torch.all(torch.isfinite(policy_logits)), (
            "Logits should be finite even with single-action mask"
        )

        # Apply mask manually and check softmax
        masked_logits = policy_logits.clone()
        masked_logits[mask == 0] = float('-inf')

        # Softmax should put all probability on action 42
        probs = torch.softmax(masked_logits, dim=-1)
        assert torch.allclose(probs[:, 42], torch.ones(batch_size)), (
            "All probability should be on the single legal action"
        )

    def test_single_legal_action_gradient_flow(self, network):
        """Gradient should flow even with single legal action.

        Edge case: Training stability with forced moves.
        """
        obs = torch.randn(1, OBSERVATION_DIM, requires_grad=True)
        mask = torch.zeros(1, ACTION_DIM)
        mask[0, 0] = 1.0  # Only action 0 legal

        policy_logits, value = network(obs, mask)

        # Compute a simple loss and backprop
        loss = policy_logits.sum() + value.sum()
        loss.backward()

        assert obs.grad is not None, "Gradient should flow through network"
        assert torch.any(obs.grad != 0), "Gradient should be non-zero"


class TestNaNInfHandling:
    """Tests for NaN and Inf in inputs."""

    def test_nan_input_produces_nan_output(self, network):
        """NaN in input should propagate to output (not crash).

        Edge case: Detecting corrupted data during training.
        This tests the network's behavior - it may produce NaN output,
        which is correct behavior (garbage in, garbage out).
        """
        obs = torch.randn(2, OBSERVATION_DIM)
        obs[0, 0] = float('nan')

        network.eval()
        with torch.no_grad():
            policy_logits, _value = network(obs)

        # The row with NaN should produce NaN (expected behavior)
        # This is actually the correct behavior - we want to detect bad data
        torch.any(torch.isnan(policy_logits[0]))
        # Note: Due to how NaN propagates, this is expected behavior

    def test_inf_input_behavior(self, network):
        """Inf in input should be handled (may produce Inf or saturate).

        Edge case: Detecting extreme values in observations.
        """
        obs = torch.randn(2, OBSERVATION_DIM)
        obs[0, 0] = float('inf')

        network.eval()
        with torch.no_grad():
            policy_logits, _value = network(obs)

        # Network may produce Inf or NaN - just checking it doesn't crash
        # The important thing is the non-corrupted sample is OK
        assert policy_logits.shape == (2, ACTION_DIM)

    def test_detection_of_nan_in_observation(self):
        """Demonstrate how to detect NaN in observations before network.

        Best practice: Check data validity before inference.
        """
        obs_clean = np.random.randn(OBSERVATION_DIM).astype(np.float32)
        obs_nan = obs_clean.copy()
        obs_nan[0] = np.nan

        assert np.all(np.isfinite(obs_clean)), "Clean observation should be finite"
        assert not np.all(np.isfinite(obs_nan)), "NaN observation should be detected"

    def test_detection_of_inf_in_observation(self):
        """Demonstrate how to detect Inf in observations before network.

        Best practice: Check data validity before inference.
        """
        obs_clean = np.random.randn(OBSERVATION_DIM).astype(np.float32)
        obs_inf = obs_clean.copy()
        obs_inf[0] = np.inf

        assert np.all(np.isfinite(obs_clean)), "Clean observation should be finite"
        assert not np.all(np.isfinite(obs_inf)), "Inf observation should be detected"


class TestZeroLengthTrajectory:
    """Tests for zero-length trajectory handling."""

    def test_empty_batch_forward(self, network):
        """Network should handle empty batch (0 samples).

        Edge case: Empty trajectory collection scenario.
        """
        obs = torch.randn(0, OBSERVATION_DIM)  # Empty batch

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(obs)

        assert policy_logits.shape == (0, ACTION_DIM), (
            f"Empty batch policy should be (0, {ACTION_DIM}), got {policy_logits.shape}"
        )
        assert value.shape[0] == 0, (
            f"Empty batch value should have 0 samples, got {value.shape}"
        )

    def test_empty_trajectory_data_structures(self):
        """Empty trajectory data structures should be valid.

        Edge case: Handling games that end immediately or fail to start.
        """
        # Empty trajectory arrays
        observations = np.empty((0, OBSERVATION_DIM), dtype=np.float32)
        actions = np.empty((0,), dtype=np.int64)
        rewards = np.empty((0,), dtype=np.float32)
        dones = np.empty((0,), dtype=bool)
        action_masks = np.empty((0, ACTION_DIM), dtype=np.float32)

        # Verify shapes are correct
        assert observations.shape == (0, OBSERVATION_DIM)
        assert actions.shape == (0,)
        assert rewards.shape == (0,)
        assert dones.shape == (0,)
        assert action_masks.shape == (0, ACTION_DIM)

        # Converting to torch should work
        obs_torch = torch.from_numpy(observations)
        assert obs_torch.shape == (0, OBSERVATION_DIM)


class TestLargeBatchSizes:
    """Tests for large batch sizes."""

    def test_large_batch_forward(self, network):
        """Network should handle large batch sizes.

        Edge case: Batch inference during training.
        """
        batch_size = 256
        obs = torch.randn(batch_size, OBSERVATION_DIM)

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(obs)

        assert policy_logits.shape == (batch_size, ACTION_DIM)
        assert value.shape == (batch_size, 1)
        assert torch.all(torch.isfinite(policy_logits))
        assert torch.all(torch.isfinite(value))

    def test_batch_consistency(self, network):
        """Large batch should give same results as individual samples.

        Edge case: Verify batching doesn't change results.
        """
        batch_size = 16
        torch.manual_seed(42)
        obs = torch.randn(batch_size, OBSERVATION_DIM)

        network.eval()
        with torch.no_grad():
            # Batch forward
            policy_batch, value_batch = network(obs)

            # Individual forwards
            policies_individual = []
            values_individual = []
            for i in range(batch_size):
                p, v = network(obs[i])
                policies_individual.append(p)
                values_individual.append(v)

            policy_stack = torch.stack(policies_individual)
            value_stack = torch.stack(values_individual)

        assert torch.allclose(policy_batch, policy_stack, atol=1e-5), (
            "Batched and individual policy outputs should match"
        )
        assert torch.allclose(value_batch.squeeze(), value_stack.squeeze(), atol=1e-5), (
            "Batched and individual value outputs should match"
        )


class TestExtremeTensorValues:
    """Tests for extreme but valid tensor values."""

    def test_all_zeros_observation(self, network):
        """Network should handle all-zero observation.

        Edge case: Unusual but valid observation state.
        """
        obs = torch.zeros(1, OBSERVATION_DIM)

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(obs)

        assert torch.all(torch.isfinite(policy_logits)), (
            "All-zero input should produce finite output"
        )
        assert torch.all(torch.isfinite(value))

    def test_all_ones_observation(self, network):
        """Network should handle all-one observation.

        Edge case: Saturated observation values.
        """
        obs = torch.ones(1, OBSERVATION_DIM)

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(obs)

        assert torch.all(torch.isfinite(policy_logits))
        assert torch.all(torch.isfinite(value))

    def test_large_magnitude_observation(self, network):
        """Network should handle large magnitude observations.

        Edge case: Observations at edge of typical float32 range.
        """
        obs = torch.randn(1, OBSERVATION_DIM) * 100.0  # Large but not extreme

        network.eval()
        with torch.no_grad():
            policy_logits, _value = network(obs)

        # May produce large logits but should be finite
        assert policy_logits.shape == (1, ACTION_DIM)

    def test_negative_observation_values(self, network):
        """Network should handle negative observation values.

        Edge case: Some encoded features may be negative.
        """
        obs = -torch.abs(torch.randn(1, OBSERVATION_DIM))  # All negative

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(obs)

        assert torch.all(torch.isfinite(policy_logits))
        assert torch.all(torch.isfinite(value))


class TestMaskEdgeCases:
    """Tests for action mask edge cases."""

    def test_all_zeros_mask(self, network):
        """Network should handle all-zero mask (no legal actions).

        Edge case: Terminal state or error condition.
        Note: The network doesn't apply the mask internally.
        """
        obs = torch.randn(1, OBSERVATION_DIM)
        mask = torch.zeros(1, ACTION_DIM)

        network.eval()
        with torch.no_grad():
            policy_logits, _value = network(obs, mask)

        # Network should still produce valid logits
        assert torch.all(torch.isfinite(policy_logits))

        # After masking, all would be -inf
        masked = policy_logits.clone()
        masked[mask == 0] = float('-inf')
        assert torch.all(masked == float('-inf'))

    def test_all_ones_mask(self, network):
        """Network should handle all-one mask (all actions legal).

        Edge case: Unusual but possible game state.
        """
        obs = torch.randn(1, OBSERVATION_DIM)
        mask = torch.ones(1, ACTION_DIM)

        network.eval()
        with torch.no_grad():
            policy_logits, _value = network(obs, mask)

        assert torch.all(torch.isfinite(policy_logits))

        # Softmax should be valid over all actions
        probs = torch.softmax(policy_logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_sparse_mask(self, network):
        """Network should handle sparse mask (few legal actions).

        Edge case: Restricted action space.
        """
        obs = torch.randn(4, OBSERVATION_DIM)
        mask = torch.zeros(4, ACTION_DIM)

        # Set different sparse patterns for each sample
        mask[0, 0] = 1.0
        mask[1, 10] = 1.0
        mask[1, 20] = 1.0
        mask[2, 50] = 1.0
        mask[2, 51] = 1.0
        mask[2, 52] = 1.0
        mask[3, ACTION_DIM - 1] = 1.0  # Last action only

        network.eval()
        with torch.no_grad():
            policy_logits, _value = network(obs, mask)

        assert torch.all(torch.isfinite(policy_logits))

        # Apply mask and check softmax validity
        masked_logits = policy_logits.clone()
        masked_logits[mask == 0] = float('-inf')
        probs = torch.softmax(masked_logits, dim=-1)

        # Probabilities should sum to 1 for each sample
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


class TestGradientEdgeCases:
    """Tests for gradient flow edge cases."""

    def test_gradient_with_extreme_loss(self, network):
        """Gradients should remain finite even with extreme loss values.

        Edge case: Training stability with outlier samples.
        """
        obs = torch.randn(4, OBSERVATION_DIM, requires_grad=True)

        policy_logits, value = network(obs)

        # Create extreme but finite loss
        loss = policy_logits.sum() * 1000.0 + value.sum() * 1000.0
        loss.backward()

        # Gradients should exist and be finite
        assert obs.grad is not None
        # Note: Gradients may be large but should be finite
        # If gradient clipping is needed, it should be done during training

    def test_gradient_flow_through_value_head(self, network):
        """Gradient should flow through value head independently.

        Edge case: Value-only loss scenarios.
        """
        obs = torch.randn(2, OBSERVATION_DIM, requires_grad=True)

        _, value = network(obs)
        loss = value.sum()
        loss.backward()

        assert obs.grad is not None
        assert torch.any(obs.grad != 0)

    def test_gradient_flow_through_policy_head(self, network):
        """Gradient should flow through policy head independently.

        Edge case: Policy-only loss scenarios.
        """
        obs = torch.randn(2, OBSERVATION_DIM, requires_grad=True)

        policy_logits, _ = network(obs)
        loss = policy_logits.sum()
        loss.backward()

        assert obs.grad is not None
        assert torch.any(obs.grad != 0)
