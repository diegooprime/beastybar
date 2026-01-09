"""Example usage of action space tensor conversion functions.

This module demonstrates how to use the tensor conversion utilities
for neural network training with Beasty Bar.
"""

import numpy as np

from _01_simulator import action_space, state


def example_basic_usage() -> None:
    """Basic usage example: converting action masks to tensors."""
    print("=== Basic Usage Example ===\n")

    # Create a game state
    game_state = state.initial_state(seed=42)
    player = 0

    # Get action space with mask
    space = action_space.legal_action_space(game_state, player)
    print(f"Total actions in catalog: {action_space.ACTION_DIM}")
    print(f"Legal actions for player {player}: {len(space.legal_indices)}")

    # Convert to tensor
    mask_tensor = action_space.action_mask_to_tensor(space)
    print(f"Mask tensor shape: {mask_tensor.shape}")
    print(f"Mask tensor dtype: {mask_tensor.dtype}")
    print(f"Number of legal actions (sum of mask): {int(np.sum(mask_tensor))}\n")


def example_batch_encoding() -> None:
    """Batch encoding example: process multiple states at once."""
    print("=== Batch Encoding Example ===\n")

    # Create multiple game states
    states = [state.initial_state(seed=i) for i in range(5)]
    perspectives = [0, 1, 0, 1, 0]  # Alternating players

    # Batch encode all masks
    batch_masks = action_space.batch_action_masks(states, perspectives)
    print(f"Batch shape: {batch_masks.shape}")
    print(f"Batch dtype: {batch_masks.dtype}")

    # Show legal action counts per state
    for i in range(len(states)):
        num_legal = int(np.sum(batch_masks[i]))
        print(f"State {i} (player {perspectives[i]}): {num_legal} legal actions")
    print()


def example_action_sampling() -> None:
    """Action sampling example: using neural network logits with masks."""
    print("=== Action Sampling Example ===\n")

    # Create a game state
    game_state = state.initial_state(seed=123)
    mask = action_space.legal_action_mask_tensor(game_state, 0)

    # Simulate neural network output (random logits)
    logits = np.random.randn(action_space.ACTION_DIM).astype(np.float32)

    # Sample action with temperature=1.0 (stochastic)
    sampled_idx = action_space.sample_masked_action(logits, mask, temperature=1.0)
    sampled_action = action_space.index_to_action(sampled_idx)
    print(f"Sampled action index: {sampled_idx}")
    print(f"Sampled action: hand_index={sampled_action.hand_index}, params={sampled_action.params}")

    # Select action greedily (deterministic)
    greedy_idx = action_space.greedy_masked_action(logits, mask)
    greedy_action = action_space.index_to_action(greedy_idx)
    print(f"\nGreedy action index: {greedy_idx}")
    print(f"Greedy action: hand_index={greedy_action.hand_index}, params={greedy_action.params}")

    # Temperature effect demonstration
    print("\nTemperature effect on sampling:")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        samples = [action_space.sample_masked_action(logits, mask, temperature=temp) for _ in range(100)]
        unique_actions = len(set(samples))
        print(f"  Temperature={temp:.1f}: {unique_actions} unique actions sampled (out of 100 samples)")
    print()


def example_index_mapping() -> None:
    """Index mapping example: understanding action catalog structure."""
    print("=== Index Mapping Example ===\n")

    print(f"Action catalog size: {action_space.ACTION_DIM}")
    print("\nFirst 10 actions in catalog:")
    for i in range(min(10, action_space.ACTION_DIM)):
        action = action_space.index_to_action(i)
        print(f"  Index {i:3d}: hand_index={action.hand_index}, params={action.params}")

    print("\nLast 5 actions in catalog:")
    for i in range(max(0, action_space.ACTION_DIM - 5), action_space.ACTION_DIM):
        action = action_space.index_to_action(i)
        print(f"  Index {i:3d}: hand_index={action.hand_index}, params={action.params}")
    print()


def example_neural_network_workflow() -> None:
    """Complete workflow: state -> observation + mask -> network -> action."""
    print("=== Neural Network Workflow Example ===\n")

    # Step 1: Get game state
    game_state = state.initial_state(seed=999)
    player = 0
    print(f"Step 1: Game state created (player {player}'s turn)")

    # Step 2: Generate action mask
    mask = action_space.legal_action_mask_tensor(game_state, player)
    num_legal = int(np.sum(mask))
    print(f"Step 2: Action mask generated ({num_legal} legal actions)")

    # Step 3: Simulate neural network forward pass
    # (In real training, this would come from the network)
    logits = np.random.randn(action_space.ACTION_DIM).astype(np.float32)
    print("Step 3: Neural network produced logits")

    # Step 4: Apply mask and sample action
    action_idx = action_space.sample_masked_action(logits, mask, temperature=1.0)
    action = action_space.index_to_action(action_idx)
    print(f"Step 4: Sampled action index {action_idx}")
    print(f"        Action: hand_index={action.hand_index}, params={action.params}")

    # Step 5: Verify action is legal
    space = action_space.legal_action_space(game_state, player)
    is_legal = action in space.actions and space.mask[action_idx] == 1
    print(f"Step 5: Action legality verified: {is_legal}")
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_batch_encoding()
    example_action_sampling()
    example_index_mapping()
    example_neural_network_workflow()

    print("=== All Examples Complete ===")
