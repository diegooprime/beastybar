"""Example demonstrating neural MCTS agent for Beasty Bar.

This example shows how to:
1. Create a neural network for policy-value estimation
2. Initialize an MCTS agent with the network
3. Use the agent to play games
4. Adjust hyperparameters for different behaviors
"""

from __future__ import annotations

from _01_simulator import engine, state
from _02_agents.mcts import MCTS, MCTSAgent
from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig


def basic_usage():
    """Basic usage of neural MCTS agent."""
    print("=" * 60)
    print("BASIC USAGE")
    print("=" * 60)

    # Create a neural network
    network = BeastyBarNetwork()
    print(f"Network created with {network.count_parameters():,} parameters")

    # Create MCTS agent
    agent = MCTSAgent(
        network=network,
        num_simulations=100,  # Number of MCTS simulations per move
        temperature=1.0,  # Stochastic action selection
    )
    print(f"Agent: {agent.name}")

    # Play a few moves
    game_state = state.initial_state(seed=42)

    for turn in range(5):
        if engine.is_terminal(game_state):
            break

        player = game_state.active_player
        legal = list(engine.legal_actions(game_state, player))

        print(f"\nTurn {turn + 1}, Player {player}, Legal actions: {len(legal)}")

        # Select action
        action = agent.select_action(game_state, legal)
        print(f"Selected: {action}")

        # Apply action
        game_state = engine.step(game_state, action)

    print("\n✓ Basic usage complete")


def deterministic_play():
    """Demonstrate deterministic (greedy) play."""
    print("\n" + "=" * 60)
    print("DETERMINISTIC PLAY")
    print("=" * 60)

    network = BeastyBarNetwork()
    agent = MCTSAgent(
        network=network,
        num_simulations=200,
        temperature=0.0,  # Greedy selection
    )

    game_state = state.initial_state(seed=123)

    # Play same position twice - should get same action
    legal = list(engine.legal_actions(game_state, 0))

    action1 = agent.select_action_deterministic(game_state, legal)
    action2 = agent.select_action_deterministic(game_state, legal)

    print(f"Action 1: {action1}")
    print(f"Action 2: {action2}")
    print(f"Deterministic: {action1 == action2}")

    print("\n✓ Deterministic play verified")


def policy_extraction():
    """Extract policy distribution for training."""
    print("\n" + "=" * 60)
    print("POLICY EXTRACTION")
    print("=" * 60)

    network = BeastyBarNetwork()
    agent = MCTSAgent(network=network, num_simulations=100)

    game_state = state.initial_state(seed=42)

    # Get policy distribution over actions
    policy = agent.get_policy(game_state)

    print(f"Policy distribution over {len(policy)} actions:")
    for action_idx, prob in sorted(policy.items(), key=lambda x: -x[1])[:5]:
        print(f"  Action {action_idx}: {prob:.3f}")

    total_prob = sum(policy.values())
    print(f"Total probability: {total_prob:.6f}")

    print("\n✓ Policy extraction complete")


def hyperparameter_tuning():
    """Demonstrate different hyperparameter settings."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    network = BeastyBarNetwork()

    # Fast exploration (few simulations, high temperature)
    agent_fast = MCTSAgent(
        network=network,
        num_simulations=50,
        temperature=1.5,
        c_puct=2.0,  # High exploration
    )

    # Strong play (many simulations, low temperature)
    agent_strong = MCTSAgent(
        network=network,
        num_simulations=400,
        temperature=0.1,
        c_puct=1.0,  # Lower exploration
    )

    game_state = state.initial_state(seed=42)
    legal = list(engine.legal_actions(game_state, 0))

    print(f"Fast agent: {agent_fast.name}")
    action_fast = agent_fast.select_action(game_state, legal)
    print(f"  Selected: {action_fast}")

    print(f"\nStrong agent: {agent_strong.name}")
    action_strong = agent_strong.select_action(game_state, legal)
    print(f"  Selected: {action_strong}")

    print("\n✓ Hyperparameter tuning demonstrated")


def custom_network_config():
    """Create agent with custom network configuration."""
    print("\n" + "=" * 60)
    print("CUSTOM NETWORK CONFIGURATION")
    print("=" * 60)

    # Create smaller network for faster inference
    config = NetworkConfig(
        hidden_dim=64,  # Smaller than default (128)
        num_heads=2,  # Fewer heads (default: 4)
        num_layers=1,  # Single layer
        dropout=0.1,
    )

    network = BeastyBarNetwork(config)
    print(f"Custom network: {network.count_parameters():,} parameters")

    agent = MCTSAgent(network=network, num_simulations=100)

    game_state = state.initial_state(seed=42)
    legal = list(engine.legal_actions(game_state, 0))

    action = agent.select_action(game_state, legal)
    print(f"Selected: {action}")

    print("\n✓ Custom network configuration complete")


def temperature_scheduling():
    """Demonstrate temperature scheduling during training."""
    print("\n" + "=" * 60)
    print("TEMPERATURE SCHEDULING")
    print("=" * 60)

    network = BeastyBarNetwork()
    agent = MCTSAgent(network=network, num_simulations=100, temperature=1.0)

    game_state = state.initial_state(seed=42)
    legal = list(engine.legal_actions(game_state, 0))

    # Early training: high temperature for exploration
    agent.set_temperature(1.5)
    print(f"Early training (temp={agent._temperature})")
    action = agent.select_action(game_state, legal)
    print(f"  Action: {action}")

    # Mid training: moderate temperature
    agent.set_temperature(1.0)
    print(f"\nMid training (temp={agent._temperature})")
    action = agent.select_action(game_state, legal)
    print(f"  Action: {action}")

    # Late training: low temperature for exploitation
    agent.set_temperature(0.3)
    print(f"\nLate training (temp={agent._temperature})")
    action = agent.select_action(game_state, legal)
    print(f"  Action: {action}")

    print("\n✓ Temperature scheduling demonstrated")


def direct_mcts_usage():
    """Use MCTS class directly for custom applications."""
    print("\n" + "=" * 60)
    print("DIRECT MCTS USAGE")
    print("=" * 60)

    network = BeastyBarNetwork()
    mcts = MCTS(
        network=network,
        num_simulations=100,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    game_state = state.initial_state(seed=42)
    perspective = 0

    # Run search and get visit distribution
    visit_distribution = mcts.search(
        game_state,
        perspective,
        temperature=1.0,
        add_root_noise=True,
    )

    print(f"Visit distribution over {len(visit_distribution)} actions:")
    for action_idx, prob in sorted(visit_distribution.items(), key=lambda x: -x[1])[:3]:
        print(f"  Action {action_idx}: {prob:.3f}")

    print("\n✓ Direct MCTS usage complete")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("NEURAL MCTS EXAMPLES FOR BEASTY BAR")
    print("=" * 60)

    basic_usage()
    deterministic_play()
    policy_extraction()
    hyperparameter_tuning()
    custom_network_config()
    temperature_scheduling()
    direct_mcts_usage()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
