"""Example script demonstrating AlphaZero-style MCTS training.

This script shows how to:
1. Configure and initialize MCTS trainer
2. Run training iterations
3. Save and load checkpoints
4. Compare MCTS vs PPO approaches

Usage:
    python _03_training/example_mcts_training.py
"""

from __future__ import annotations

import logging

from _02_agents.neural.utils import NetworkConfig
from _03_training.mcts_trainer import MCTSTrainer, MCTSTrainerConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def example_quick_training() -> None:
    """Run a quick MCTS training example (5 iterations)."""
    logger.info("=" * 80)
    logger.info("Quick MCTS Training Example")
    logger.info("=" * 80)

    # Create configuration for quick test
    config = MCTSTrainerConfig(
        # Network architecture
        network_config=NetworkConfig(
            hidden_dim=256,
            num_layers=4,
        ),
        # Self-play settings
        games_per_iteration=10,  # Small for quick test
        mcts_simulations=100,  # Reduced for speed
        temperature=1.0,
        c_puct=1.0,
        # Training settings
        total_iterations=5,  # Just a few iterations
        batch_size=64,
        epochs_per_iteration=3,
        learning_rate=1e-3,
        # Logging
        checkpoint_frequency=5,
        eval_frequency=5,
        experiment_name="mcts_quick_test",
        device="auto",
    )

    # Create trainer
    trainer = MCTSTrainer(config)

    logger.info(f"Network has {trainer.network.count_parameters():,} parameters")
    logger.info(f"Training for {config.total_iterations} iterations")

    # Run training
    trainer.train()

    logger.info("Training complete!")


def example_production_config() -> MCTSTrainerConfig:
    """Create a production-ready MCTS training configuration.

    This configuration follows AlphaZero hyperparameters adapted for Beasty Bar.

    Returns:
        MCTSTrainerConfig ready for production training.
    """
    return MCTSTrainerConfig(
        # Network architecture
        network_config=NetworkConfig(
            observation_dim=988,  # Beasty Bar observation size
            action_dim=124,  # Beasty Bar action space size
            hidden_dim=512,  # Large capacity for complex patterns
            num_layers=6,  # Deep network
        ),
        # Self-play settings (AlphaZero-style)
        games_per_iteration=512,  # Generate 512 games per iteration
        mcts_simulations=800,  # 800 MCTS simulations per move
        temperature=1.0,  # Sample proportionally to visit counts
        c_puct=1.0,  # Standard exploration constant
        # Training settings
        total_iterations=1000,  # 1000 training iterations
        batch_size=256,  # Reasonable batch size
        epochs_per_iteration=10,  # Multiple passes over data
        learning_rate=1e-3,  # Standard learning rate
        weight_decay=1e-4,  # L2 regularization
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        entropy_bonus_weight=0.01,  # Small exploration bonus
        # Schedule
        lr_warmup_iterations=10,
        lr_decay="cosine",
        # Checkpointing
        checkpoint_frequency=50,  # Save every 50 iterations
        eval_frequency=10,  # Evaluate every 10 iterations
        # Experiment tracking
        experiment_name="beastybar_mcts_alphazero",
        checkpoint_dir="checkpoints",
        device="auto",
    )


def compare_mcts_vs_ppo() -> None:
    """Compare MCTS and PPO training approaches."""
    logger.info("=" * 80)
    logger.info("MCTS vs PPO Comparison")
    logger.info("=" * 80)

    logger.info("\nMCTS (AlphaZero) Training:")
    logger.info("  ✓ Simpler loss: policy CE + value MSE - entropy")
    logger.info("  ✓ MCTS improves policy during search (stronger training signal)")
    logger.info("  ✓ Uses terminal game outcome as value target")
    logger.info("  ✓ No advantage estimation or GAE needed")
    logger.info("  ✓ More stable for two-player zero-sum games")
    logger.info("  ✗ Slower: MCTS search is compute-intensive")
    logger.info("  ✗ Requires many simulations per move (800+)")

    logger.info("\nPPO Training:")
    logger.info("  ✓ Faster: no MCTS search overhead")
    logger.info("  ✓ Efficient GPU utilization with vectorized envs")
    logger.info("  ✓ Works well for single-agent RL")
    logger.info("  ✗ More complex: clipped surrogate, GAE, multiple PPO epochs")
    logger.info("  ✗ Policy target is just the action taken (weaker signal)")
    logger.info("  ✗ Can be unstable in competitive settings")
    logger.info("  ✗ Requires careful hyperparameter tuning")

    logger.info("\nRecommendation for Beasty Bar:")
    logger.info("  - Use MCTS for highest quality play")
    logger.info("  - Use PPO for faster iteration during development")
    logger.info("  - MCTS expected to achieve higher Elo given enough training")


def main() -> None:
    """Run example demonstrations."""
    # Show comparison
    compare_mcts_vs_ppo()

    print("\n" + "=" * 80)
    print("To run quick training test, uncomment the line below:")
    print("=" * 80)
    # Uncomment to actually run training:
    # example_quick_training()

    print("\n" + "=" * 80)
    print("Production Configuration Example:")
    print("=" * 80)
    config = example_production_config()
    print(f"Games per iteration: {config.games_per_iteration}")
    print(f"MCTS simulations: {config.mcts_simulations}")
    print(f"Total iterations: {config.total_iterations}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Network hidden dim: {config.network_config.hidden_dim}")
    print(f"Network layers: {config.network_config.num_layers}")

    print("\n" + "=" * 80)
    print("To run production training:")
    print("=" * 80)
    print("trainer = MCTSTrainer(config)")
    print("trainer.train()")


if __name__ == "__main__":
    main()
