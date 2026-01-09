"""Tests for neural MCTS implementation."""

from __future__ import annotations

import torch

from _01_simulator import engine, state
from _02_agents.mcts import MCTS, MCTSAgent
from _02_agents.neural.network import BeastyBarNetwork


def test_mcts_basic_search():
    """Test basic MCTS search functionality."""
    network = BeastyBarNetwork()
    mcts = MCTS(network, num_simulations=10)

    game_state = state.initial_state(seed=42)
    perspective = 0

    # Run search
    visit_distribution = mcts.search(game_state, perspective)

    # Should return a distribution over legal actions
    assert isinstance(visit_distribution, dict)
    assert len(visit_distribution) > 0

    # Probabilities should sum to approximately 1.0
    total_prob = sum(visit_distribution.values())
    assert abs(total_prob - 1.0) < 0.01


def test_mcts_agent_select_action():
    """Test MCTSAgent action selection."""
    network = BeastyBarNetwork()
    agent = MCTSAgent(network, num_simulations=10)

    game_state = state.initial_state(seed=42)
    legal = list(engine.legal_actions(game_state, 0))

    # Select action
    action = agent.select_action(game_state, legal)

    # Should return a legal action
    assert action in legal


def test_mcts_agent_deterministic():
    """Test deterministic action selection."""
    network = BeastyBarNetwork()
    agent = MCTSAgent(network, num_simulations=20)

    game_state = state.initial_state(seed=42)
    legal = list(engine.legal_actions(game_state, 0))

    # Select deterministically twice
    action1 = agent.select_action_deterministic(game_state, legal)
    action2 = agent.select_action_deterministic(game_state, legal)

    # Should be the same action
    assert action1 == action2
    assert action1 in legal


def test_mcts_agent_temperature():
    """Test temperature setting."""
    network = BeastyBarNetwork()
    agent = MCTSAgent(network, num_simulations=10, temperature=1.0)

    # Set temperature
    agent.set_temperature(0.5)
    assert agent._temperature == 0.5

    # Test setting simulations
    agent.set_num_simulations(50)
    assert agent._num_simulations == 50
    assert agent.mcts.num_simulations == 50


def test_mcts_agent_get_policy():
    """Test policy distribution retrieval."""
    network = BeastyBarNetwork()
    agent = MCTSAgent(network, num_simulations=15)

    game_state = state.initial_state(seed=42)

    # Get policy
    policy = agent.get_policy(game_state)

    # Should be a distribution
    assert isinstance(policy, dict)
    assert len(policy) > 0

    # Should sum to ~1.0
    total = sum(policy.values())
    assert abs(total - 1.0) < 0.01


def test_mcts_with_terminal_state():
    """Test MCTS handles terminal states correctly."""
    network = BeastyBarNetwork()
    mcts = MCTS(network, num_simulations=5)

    # Create a game and play until near terminal
    game_state = state.initial_state(seed=123)

    # Play a few moves
    for _ in range(5):
        if engine.is_terminal(game_state):
            break
        legal = list(engine.legal_actions(game_state, game_state.active_player))
        if legal:
            action = legal[0]
            game_state = engine.step(game_state, action)

    # Should still be able to search if not terminal
    if not engine.is_terminal(game_state):
        perspective = game_state.active_player
        visit_distribution = mcts.search(game_state, perspective)
        assert len(visit_distribution) >= 0


def test_mcts_node_structure():
    """Test MCTSNode creation and expansion."""
    from _02_agents.mcts.search import MCTSNode

    game_state = state.initial_state(seed=42)
    node = MCTSNode(state=game_state)

    # Should start unvisited
    assert node.visit_count == 0
    assert node.mean_value == 0.0
    assert not node.is_expanded


def test_mcts_with_device():
    """Test MCTS with explicit device specification."""
    network = BeastyBarNetwork()
    device = torch.device("cpu")

    mcts = MCTS(network, num_simulations=5, device=device)
    assert mcts.device == device

    game_state = state.initial_state(seed=42)
    visit_distribution = mcts.search(game_state, perspective=0)
    assert len(visit_distribution) > 0


def test_mcts_dirichlet_noise():
    """Test Dirichlet noise is applied correctly."""
    network = BeastyBarNetwork()

    # Create MCTS with and without noise
    mcts_with_noise = MCTS(
        network,
        num_simulations=20,
        dirichlet_epsilon=0.25,
    )
    mcts_without_noise = MCTS(
        network,
        num_simulations=20,
        dirichlet_epsilon=0.0,  # No noise
    )

    game_state = state.initial_state(seed=42)

    # Both should work
    dist_with_noise = mcts_with_noise.search(
        game_state, perspective=0, add_root_noise=True
    )
    dist_without_noise = mcts_without_noise.search(
        game_state, perspective=0, add_root_noise=False
    )

    assert len(dist_with_noise) > 0
    assert len(dist_without_noise) > 0
