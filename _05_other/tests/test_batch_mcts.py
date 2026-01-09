"""Test batched MCTS implementation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from _01_simulator import state
from _02_agents.mcts.batch_mcts import BatchMCTS
from _02_agents.neural.network import BeastyBarNetwork


def test_batch_mcts_initialization():
    """Test BatchMCTS can be initialized."""
    network = BeastyBarNetwork()
    batch_mcts = BatchMCTS(
        network,
        num_simulations=10,
        virtual_loss=3.0,
        batch_size=4,
    )

    assert batch_mcts.num_simulations == 10
    assert batch_mcts.virtual_loss == 3.0
    assert batch_mcts.batch_size == 4
    assert batch_mcts.c_puct == 1.5


def test_batch_mcts_search_single_state():
    """Test batch search with single state."""
    network = BeastyBarNetwork()
    batch_mcts = BatchMCTS(
        network,
        num_simulations=10,
        batch_size=2,
    )

    # Single state
    states = [state.initial_state(seed=42)]
    distributions = batch_mcts.search_batch(states, perspective=0)

    assert len(distributions) == 1
    assert isinstance(distributions[0], dict)
    assert len(distributions[0]) > 0

    # Check probabilities sum to 1
    total_prob = sum(distributions[0].values())
    assert abs(total_prob - 1.0) < 1e-5


def test_batch_mcts_search_multiple_states():
    """Test batch search with multiple states."""
    network = BeastyBarNetwork()
    batch_mcts = BatchMCTS(
        network,
        num_simulations=20,
        virtual_loss=3.0,
        batch_size=4,
    )

    # Multiple states
    states = [state.initial_state(seed=i) for i in range(8)]
    distributions = batch_mcts.search_batch(states, perspective=0)

    assert len(distributions) == 8

    # Check each distribution
    for dist in distributions:
        assert isinstance(dist, dict)
        assert len(dist) > 0

        # Probabilities should sum to 1
        total_prob = sum(dist.values())
        assert abs(total_prob - 1.0) < 1e-5


def test_batch_mcts_empty_states():
    """Test batch search with empty state list."""
    network = BeastyBarNetwork()
    batch_mcts = BatchMCTS(network, num_simulations=10)

    states = []
    distributions = batch_mcts.search_batch(states, perspective=0)

    assert len(distributions) == 0


def test_batch_mcts_deterministic_with_no_noise():
    """Test that batch MCTS is deterministic without noise."""
    network = BeastyBarNetwork()

    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    batch_mcts = BatchMCTS(
        network,
        num_simulations=20,
        batch_size=4,
    )

    states = [state.initial_state(seed=100)]

    # First run without noise
    distributions1 = batch_mcts.search_batch(states, perspective=0, add_root_noise=False)

    # Reset seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Second run without noise
    distributions2 = batch_mcts.search_batch(states, perspective=0, add_root_noise=False)

    # Should be identical
    assert distributions1[0].keys() == distributions2[0].keys()
    for action in distributions1[0]:
        assert abs(distributions1[0][action] - distributions2[0][action]) < 1e-5


def test_batch_mcts_virtual_loss_prevents_path_collision():
    """Test that virtual loss helps spread exploration."""
    network = BeastyBarNetwork()

    # With virtual loss
    batch_mcts_with_vl = BatchMCTS(
        network,
        num_simulations=50,
        virtual_loss=5.0,  # High virtual loss
        batch_size=8,
    )

    # Without virtual loss
    batch_mcts_no_vl = BatchMCTS(
        network,
        num_simulations=50,
        virtual_loss=0.0,  # No virtual loss
        batch_size=8,
    )

    # Same state
    test_state = state.initial_state(seed=123)

    # Run searches
    with_vl_dist = batch_mcts_with_vl.search_batch([test_state], perspective=0)[0]
    no_vl_dist = batch_mcts_no_vl.search_batch([test_state], perspective=0)[0]

    # Both should produce valid distributions
    assert abs(sum(with_vl_dist.values()) - 1.0) < 1e-5
    assert abs(sum(no_vl_dist.values()) - 1.0) < 1e-5

    # Distributions might differ due to exploration pattern
    # (This is a weak test, but verifies both modes work)
    assert len(with_vl_dist) > 0
    assert len(no_vl_dist) > 0


def test_batch_mcts_different_batch_sizes():
    """Test batch MCTS with different batch sizes."""
    network = BeastyBarNetwork()

    states = [state.initial_state(seed=i) for i in range(6)]

    # Test with batch_size=1 (sequential-like)
    batch_mcts_1 = BatchMCTS(network, num_simulations=10, batch_size=1)
    dist_1 = batch_mcts_1.search_batch(states, perspective=0)

    # Test with batch_size=4
    batch_mcts_4 = BatchMCTS(network, num_simulations=10, batch_size=4)
    dist_4 = batch_mcts_4.search_batch(states, perspective=0)

    # Both should produce 6 distributions
    assert len(dist_1) == 6
    assert len(dist_4) == 6

    # All distributions should be valid
    for dist in dist_1 + dist_4:
        assert abs(sum(dist.values()) - 1.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
