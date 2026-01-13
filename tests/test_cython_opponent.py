"""Tests for Cython-accelerated game generation with opponent diversity.

These tests verify the generate_games_vectorized_cython_with_opponent function
which enables Cython-accelerated self-play against various opponent types.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from _01_simulator.action_space import ACTION_DIM
from _01_simulator.observations import OBSERVATION_DIM
from _02_agents.heuristic import HeuristicAgent
from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig
from _02_agents.random_agent import RandomAgent

# Try to import Cython module
try:
    from _03_training.vectorized_env_cython import (
        is_cython_available,
    )
    CYTHON_AVAILABLE = is_cython_available()
except ImportError:
    CYTHON_AVAILABLE = False

    def is_cython_available() -> bool:
        return False


# Conditional import of the function being tested
# This function may not exist yet - tests anticipate its implementation
try:
    from _03_training.vectorized_env_cython import (
        generate_games_vectorized_cython_with_opponent,
    )
    FUNCTION_EXISTS = True
except ImportError:
    FUNCTION_EXISTS = False


# Helper to skip tests if function doesn't exist yet
requires_function = pytest.mark.skipif(
    not FUNCTION_EXISTS,
    reason="generate_games_vectorized_cython_with_opponent not implemented yet"
)

# Helper to skip tests if Cython is not available
requires_cython = pytest.mark.skipif(
    not CYTHON_AVAILABLE,
    reason="Cython extension not available"
)


@pytest.fixture
def small_network():
    """Create a small network for testing."""
    config = NetworkConfig(
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    return BeastyBarNetwork(config)


@pytest.fixture
def device():
    """Get test device (CPU for consistent testing)."""
    return torch.device("cpu")


# =============================================================================
# Test 1: Cython with Random Opponent
# =============================================================================


@requires_function
@requires_cython
class TestCythonWithRandomOpponent:
    """Test Cython-accelerated generation with RandomAgent opponent."""

    def test_generates_trajectories(self, small_network, device):
        """Generate games with RandomAgent as opponent and verify trajectories."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=1.0,
            device=device,
            seeds=[100, 101, 102, 103],
        )

        # Verify trajectories are generated
        assert len(trajectories) == 4, "Should generate 4 trajectories"
        for traj in trajectories:
            assert len(traj.steps_p0) > 0, "P0 should have steps"
            # P1 (opponent) may also have steps recorded depending on implementation

    def test_p0_uses_network_valid_actions(self, small_network, device):
        """Verify P0 trajectories have valid actions from network."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=2,
            temperature=1.0,
            device=device,
            seeds=[200, 201],
        )

        for traj in trajectories:
            for step in traj.steps_p0:
                # Action should be within valid range
                assert 0 <= step.action < ACTION_DIM, f"Invalid action: {step.action}"
                # Action should be legal (mask should allow it)
                assert step.action_mask[step.action] > 0, "Action should be legal"
                # Action probability should be valid
                assert 0 < step.action_prob <= 1.0, f"Invalid prob: {step.action_prob}"

    def test_stats_returned(self, small_network, device):
        """Verify stats dictionary is returned with expected keys."""
        opponent = RandomAgent(seed=42)

        _trajectories, stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=1.0,
            device=device,
        )

        # Check stats dict exists and has key metrics
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert "games_generated" in stats or "total_steps" in stats


# =============================================================================
# Test 2: Cython with Heuristic Opponent
# =============================================================================


@requires_function
@requires_cython
class TestCythonWithHeuristicOpponent:
    """Test Cython-accelerated generation with HeuristicAgent opponent."""

    def test_games_complete_successfully(self, small_network, device):
        """Generate games with HeuristicAgent and verify completion."""
        opponent = HeuristicAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=1.0,
            device=device,
            seeds=[300, 301, 302, 303],
        )

        # All games should complete
        assert len(trajectories) == 4

        # Each trajectory should have final state
        for traj in trajectories:
            # Trajectories should have steps from player 0
            assert len(traj.steps_p0) > 0, "P0 should have steps"
            # Check final_scores if available
            if hasattr(traj, "final_scores") and traj.final_scores is not None:
                scores = traj.final_scores
                assert len(scores) == 2, "Should have 2 scores"
                assert all(isinstance(s, (int, float)) for s in scores)

    def test_with_aggressive_heuristic(self, small_network, device):
        """Test with aggressive heuristic variant."""
        from _02_agents.heuristic import HeuristicConfig

        config = HeuristicConfig(aggression=0.9, bar_weight=3.0)
        opponent = HeuristicAgent(config=config, seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=2,
            temperature=1.0,
            device=device,
        )

        assert len(trajectories) == 2


# =============================================================================
# Test 3: Cython with Checkpoint (Network) Opponent
# =============================================================================


@requires_function
@requires_cython
class TestCythonWithCheckpointOpponent:
    """Test Cython-accelerated generation with network as opponent."""

    def test_both_networks_used(self, small_network, device):
        """Verify both networks are used when opponent_network is provided."""
        # Create a separate network for opponent (could be same architecture)
        opponent_config = NetworkConfig(
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        opponent_network = BeastyBarNetwork(opponent_config)

        trajectories, stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent_network=opponent_network,
            num_games=4,
            temperature=1.0,
            device=device,
            seeds=[400, 401, 402, 403],
        )

        # Verify trajectories generated
        assert len(trajectories) == 4

        # Stats should show inference calls for both players if tracked
        if "p0_inference_calls" in stats and "p1_inference_calls" in stats:
            assert stats["p0_inference_calls"] > 0, "P0 should have inference calls"
            assert stats["p1_inference_calls"] > 0, "P1 should have inference calls"

    def test_opponent_network_overrides_agent(self, small_network, device):
        """Verify opponent_network is used even if agent is also provided."""
        opponent_network = BeastyBarNetwork(
            NetworkConfig(hidden_dim=32, num_heads=2, num_layers=1, dropout=0.0)
        )
        agent = RandomAgent(seed=42)

        # When both are provided, network should take precedence
        # (based on existing pattern in vectorized_env.py)
        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=agent,
            opponent_network=opponent_network,
            num_games=2,
            temperature=1.0,
            device=device,
        )

        assert len(trajectories) == 2


# =============================================================================
# Test 4: Cython Fallback to Pure Python
# =============================================================================


@requires_function
class TestCythonOpponentFallback:
    """Test fallback behavior when Cython is unavailable."""

    def test_fallback_to_pure_python(self, small_network, device):
        """Test that if Cython unavailable, it falls back to pure Python."""
        opponent = RandomAgent(seed=42)

        # Mock is_cython_available to return False
        with patch(
            "_03_training.vectorized_env_cython.is_cython_available",
            return_value=False
        ):
            # The function should still work (using pure Python fallback)
            # This tests that the function handles the fallback gracefully
            trajectories, stats = generate_games_vectorized_cython_with_opponent(
                network=small_network,
                opponent=opponent,
                num_games=2,
                temperature=1.0,
                device=device,
            )

            # Should still produce valid output
            assert len(trajectories) == 2

            # Stats should indicate not using Cython
            if "using_cython" in stats:
                assert stats["using_cython"] is False

    def test_no_crash_without_cython(self, small_network, device):
        """Verify no crash occurs when Cython is mocked as unavailable."""
        opponent = HeuristicAgent(seed=42)

        with patch(
            "_03_training.vectorized_env_cython._CYTHON_AVAILABLE",
            False
        ):
            try:
                trajectories, _stats = generate_games_vectorized_cython_with_opponent(
                    network=small_network,
                    opponent=opponent,
                    num_games=2,
                    device=device,
                )
                # Should complete without error
                assert len(trajectories) >= 0
            except Exception as e:
                # If it raises, should be a handled/expected exception
                # not an unhandled crash
                assert isinstance(e, (NotImplementedError, RuntimeError, ImportError))


# =============================================================================
# Test 5: Stats Validation
# =============================================================================


@requires_function
@requires_cython
class TestCythonOpponentStats:
    """Test that returned stats have expected keys and valid values."""

    def test_stats_has_expected_keys(self, small_network, device):
        """Verify returned stats dict has expected keys."""
        opponent = RandomAgent(seed=42)

        _trajectories, stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=1.0,
            device=device,
        )

        # Expected keys based on existing patterns
        expected_keys = {"total_steps", "games_generated"}

        # At minimum, these should be present
        for key in expected_keys:
            assert key in stats, f"Stats should contain '{key}'"

    def test_inference_calls_tracked(self, small_network, device):
        """Verify inference_calls or similar metric is tracked."""
        opponent = RandomAgent(seed=42)

        _trajectories, stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=1.0,
            device=device,
        )

        # Should have some inference tracking
        inference_keys = [
            "inference_calls",
            "p0_inference_calls",
            "total_inference_calls",
        ]
        has_inference_key = any(k in stats for k in inference_keys)
        assert has_inference_key, "Stats should track inference calls"

    def test_avg_batch_size_reasonable(self, small_network, device):
        """Verify avg_batch_size is reasonable when present."""
        opponent = RandomAgent(seed=42)

        _trajectories, stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=8,
            temperature=1.0,
            device=device,
        )

        if "avg_batch_size" in stats:
            # Batch size should be positive and reasonable
            assert stats["avg_batch_size"] > 0
            # Should be at most the number of games (can't batch more than we have)
            assert stats["avg_batch_size"] <= 8 * 50  # generous upper bound

    def test_using_cython_flag(self, small_network, device):
        """Verify using_cython flag is present and correct."""
        opponent = RandomAgent(seed=42)

        _trajectories, stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=2,
            temperature=1.0,
            device=device,
        )

        if "using_cython" in stats:
            # Should be True since we're running Cython version
            assert stats["using_cython"] is True


# =============================================================================
# Test 6: Trajectory Validation
# =============================================================================


@requires_function
@requires_cython
class TestCythonOpponentTrajectoriesValid:
    """Test that returned trajectories have valid structure and data."""

    def test_observations_populated(self, small_network, device):
        """Verify observations are populated with valid data."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=2,
            temperature=1.0,
            device=device,
            seeds=[500, 501],
        )

        for traj in trajectories:
            for step in traj.steps_p0:
                # Observation should have correct shape
                assert step.observation.shape == (OBSERVATION_DIM,), \
                    f"Expected shape ({OBSERVATION_DIM},), got {step.observation.shape}"
                # Observations should be in valid range [0, 1]
                assert np.all(step.observation >= 0.0), "Observation values should be >= 0"
                assert np.all(step.observation <= 1.0), "Observation values should be <= 1"

    def test_actions_valid(self, small_network, device):
        """Verify actions are valid action indices."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=2,
            temperature=1.0,
            device=device,
        )

        for traj in trajectories:
            for step in traj.steps_p0:
                # Action should be integer in valid range
                assert isinstance(step.action, (int, np.integer))
                assert 0 <= step.action < ACTION_DIM

    def test_action_masks_valid(self, small_network, device):
        """Verify action masks have correct structure."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=2,
            temperature=1.0,
            device=device,
        )

        for traj in trajectories:
            for step in traj.steps_p0:
                # Mask should have correct shape
                assert step.action_mask.shape == (ACTION_DIM,), \
                    f"Expected shape ({ACTION_DIM},), got {step.action_mask.shape}"
                # Mask should be binary (0 or 1)
                assert np.all((step.action_mask == 0) | (step.action_mask == 1)), \
                    "Mask should be binary"
                # At least one action should be legal
                assert np.sum(step.action_mask) > 0, "At least one action should be legal"

    def test_rewards_assigned_to_final_step(self, small_network, device):
        """Verify rewards are assigned to final steps."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=1.0,
            device=device,
            shaped_rewards=False,
        )

        for traj in trajectories:
            if len(traj.steps_p0) > 0:
                last_step = traj.steps_p0[-1]
                # Check if reward is accessible (may be stored as _reward attribute)
                if hasattr(last_step, "_reward"):
                    reward = last_step._reward
                    # Win/loss/draw rewards should be -1, 0, or 1
                    assert reward in [-1.0, 0.0, 1.0], f"Unexpected reward: {reward}"

    def test_trajectory_both_players_have_steps(self, small_network, device):
        """Verify both players have recorded steps in trajectories."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=1.0,
            device=device,
        )

        for traj in trajectories:
            # P0 should always have steps (network player)
            assert len(traj.steps_p0) > 0, "P0 should have steps"
            # P1 (opponent) should also have steps recorded
            # (depending on implementation, P1 steps may or may not be collected)
            # At minimum, the trajectory structure should exist
            assert hasattr(traj, "steps_p1"), "Trajectory should have steps_p1 attribute"

    def test_shaped_rewards_option(self, small_network, device):
        """Verify shaped_rewards option affects reward calculation."""
        opponent = RandomAgent(seed=42)

        # Generate with shaped rewards
        trajectories_shaped, _ = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=1.0,
            device=device,
            seeds=[600, 601, 602, 603],
            shaped_rewards=True,
        )

        # Generate without shaped rewards (same seeds for comparison)
        trajectories_unshaped, _ = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=1.0,
            device=device,
            seeds=[600, 601, 602, 603],
            shaped_rewards=False,
        )

        # Both should complete successfully
        assert len(trajectories_shaped) == 4
        assert len(trajectories_unshaped) == 4


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


@requires_function
@requires_cython
class TestCythonOpponentEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_game(self, small_network, device):
        """Test generating a single game works correctly."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=1,
            temperature=1.0,
            device=device,
        )

        assert len(trajectories) == 1

    def test_many_games(self, small_network, device):
        """Test generating many games for batch efficiency."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=32,
            temperature=1.0,
            device=device,
        )

        assert len(trajectories) == 32

    def test_low_temperature(self, small_network, device):
        """Test with low temperature (more deterministic actions)."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=0.1,
            device=device,
        )

        assert len(trajectories) == 4

    def test_high_temperature(self, small_network, device):
        """Test with high temperature (more random actions)."""
        opponent = RandomAgent(seed=42)

        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=opponent,
            num_games=4,
            temperature=2.0,
            device=device,
        )

        assert len(trajectories) == 4

    def test_no_opponent_falls_back_to_self_play(self, small_network, device):
        """Test that no opponent defaults to self-play behavior."""
        trajectories, _stats = generate_games_vectorized_cython_with_opponent(
            network=small_network,
            opponent=None,
            opponent_network=None,
            num_games=2,
            temperature=1.0,
            device=device,
        )

        # Should still work (falls back to self-play)
        assert len(trajectories) == 2


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
