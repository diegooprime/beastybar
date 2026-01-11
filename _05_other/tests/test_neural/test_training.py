"""
Test suite for training loop and infrastructure.

Tests the end-to-end training process, checkpointing, metrics logging,
and integration of all neural components.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from _02_agents.neural.utils import NetworkConfig
from _03_training.ppo import PPOConfig
from _03_training.trainer import (
    Trainer,
    TrainingConfig,
    create_trainer_from_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


@pytest.fixture
def small_training_config():
    """Create minimal config for fast testing."""
    return TrainingConfig(
        network_config=NetworkConfig(hidden_dim=32, num_heads=2, num_layers=1, dropout=0.0),
        ppo_config=PPOConfig(
            learning_rate=1e-3,
            ppo_epochs=1,
            minibatch_size=8,
        ),
        games_per_iteration=2,  # Very small for fast testing
        total_iterations=3,
        checkpoint_frequency=2,
        eval_frequency=10,  # Don't eval in these tests
        buffer_size=100,
        min_buffer_size=10,
        log_frequency=1,
        seed=42,
        device="cpu",
        async_game_generation=False,  # Disable async for tests to avoid multiprocessing issues
    )


def test_training_iteration(small_training_config):
    """Verify complete training iteration executes successfully."""
    trainer = Trainer(small_training_config)

    # Run a single training iteration
    metrics = trainer.train_iteration()

    # Verify metrics are returned
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "iteration" in metrics, "Should track iteration number"
    assert metrics["iteration"] == 0, "First iteration should be 0"

    # Verify expected metric keys
    expected_keys = [
        "self_play/games_generated",
        "self_play/transitions_collected",
        "total_games_played",
        "total_transitions",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric key: {key}"

    # Verify games were generated
    assert metrics["self_play/games_generated"] == small_training_config.games_per_iteration
    assert metrics["total_games_played"] == small_training_config.games_per_iteration


def test_checkpoint_save_load(small_training_config):
    """Verify model checkpointing and restoration."""
    trainer = Trainer(small_training_config)

    # Run a few iterations
    for _ in range(2):
        trainer.train_iteration()

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
        save_training_checkpoint(trainer, str(checkpoint_path))

        # Verify checkpoint exists
        assert checkpoint_path.exists(), "Checkpoint file should exist"

        # Create new trainer and load checkpoint
        trainer_restored = Trainer(small_training_config)
        original_iteration = trainer.current_iteration

        # Load checkpoint
        load_training_checkpoint(str(checkpoint_path), trainer_restored)

        # Verify state was restored
        assert trainer_restored.current_iteration == original_iteration, \
            f"Iteration not restored: {trainer_restored.current_iteration} != {original_iteration}"

        # Verify network weights match
        original_state = trainer.network.state_dict()
        restored_state = trainer_restored.network.state_dict()

        for key in original_state.keys():
            assert key in restored_state, f"Missing key in restored state: {key}"
            assert torch.allclose(original_state[key], restored_state[key], atol=1e-6), \
                f"Weights differ for {key}"


def test_training_metrics_logged(small_training_config):
    """Verify training metrics are tracked correctly."""
    # Reduce to 2 iterations for speed
    small_training_config.total_iterations = 2

    trainer = Trainer(small_training_config)

    # Run training
    trainer.train()

    # Verify metrics history exists
    assert len(trainer._metrics_history) > 0, "Should have metrics history"
    assert len(trainer._metrics_history) == small_training_config.total_iterations, \
        f"Should have {small_training_config.total_iterations} iterations of metrics"

    # Check first iteration metrics
    first_metrics = trainer._metrics_history[0]

    # Verify key metric fields
    expected_metrics = [
        "iteration",
        "total_games_played",
        "total_transitions",
        "iteration_time",
    ]

    for metric in expected_metrics:
        assert metric in first_metrics, f"Missing metric: {metric}"
        assert isinstance(first_metrics[metric], (int, float)), \
            f"Metric {metric} should be numeric"

    # Verify iteration numbers are sequential
    for i, metrics in enumerate(trainer._metrics_history):
        assert metrics["iteration"] == i, f"Iteration {i} has wrong number"
