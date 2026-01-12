"""Demonstration of batched MCTS for efficient parallel search.

This script shows how to use BatchMCTS to search multiple game states
simultaneously with batched neural network evaluation, achieving 5-10x
speedup over sequential MCTS.

Key Features Demonstrated:
- Batch evaluation of multiple states
- Virtual loss for parallel exploration
- Configurable batch size and simulations
- Performance comparison with sequential MCTS
"""

from __future__ import annotations

import time

import torch

from _01_simulator import state
from _02_agents.mcts import MCTS, BatchMCTS
from _02_agents.neural.network import BeastyBarNetwork


def main() -> None:
    """Run batch MCTS demonstration."""
    print("=" * 70)
    print("Batched MCTS Demonstration")
    print("=" * 70)

    # Create network
    print("\n1. Creating neural network...")
    network = BeastyBarNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    print(f"   Network: {network.count_parameters():,} parameters on {device}")

    # Create test states
    num_states = 8
    print(f"\n2. Creating {num_states} game states...")
    states = [state.initial_state(seed=i) for i in range(num_states)]
    print(f"   Generated {len(states)} initial game states")

    # Configuration
    num_simulations = 100
    batch_size = 8

    print("\n3. Configuration:")
    print(f"   Simulations per state: {num_simulations}")
    print(f"   Batch size: {batch_size}")
    print("   Virtual loss: 3.0")

    # Batch MCTS
    print("\n4. Running Batched MCTS...")
    batch_mcts = BatchMCTS(
        network=network,
        num_simulations=num_simulations,
        c_puct=1.5,
        virtual_loss=3.0,
        batch_size=batch_size,
        device=device,
    )

    start = time.perf_counter()
    batch_distributions = batch_mcts.search_batch(
        states,
        perspective=0,
        add_root_noise=True,
    )
    batch_time = time.perf_counter() - start

    print(f"   Completed in {batch_time:.3f}s")
    print(f"   Average per state: {batch_time / len(states):.3f}s")
    print(f"   Total simulations: {num_simulations * len(states):,}")

    # Sequential MCTS (for comparison)
    print("\n5. Running Sequential MCTS (for comparison)...")
    sequential_mcts = MCTS(
        network=network,
        num_simulations=num_simulations,
        c_puct=1.5,
        device=device,
    )

    start = time.perf_counter()
    sequential_distributions = []
    for s in states:
        dist = sequential_mcts.search(
            s,
            perspective=0,
            add_root_noise=True,
        )
        sequential_distributions.append(dist)
    sequential_time = time.perf_counter() - start

    print(f"   Completed in {sequential_time:.3f}s")
    print(f"   Average per state: {sequential_time / len(states):.3f}s")

    # Performance comparison
    speedup = sequential_time / batch_time
    print("\n6. Performance Comparison:")
    print(f"   Batched MCTS:    {batch_time:.3f}s")
    print(f"   Sequential MCTS: {sequential_time:.3f}s")
    print(f"   Speedup:         {speedup:.2f}x")

    # Analyze results
    print("\n7. Result Analysis:")
    print(f"   Number of distributions: {len(batch_distributions)}")

    for i, dist in enumerate(batch_distributions[:3]):  # Show first 3
        top_actions = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n   State {i}:")
        print(f"     Total actions: {len(dist)}")
        print("     Top 3 actions:")
        for action_idx, prob in top_actions:
            print(f"       Action {action_idx:3d}: {prob:.4f}")

    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
