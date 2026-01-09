"""Tests for network mode switching utilities.

This module tests the inference_mode and training_mode context managers
to ensure proper network.eval()/train() mode switching during training
and inference.
"""

import pytest
import torch
import torch.nn as nn

from _03_training.utils import inference_mode, training_mode


class SimpleNetwork(nn.Module):
    """Simple network for testing mode switching."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(5)

    def forward(self, x):
        x = self.fc(x)
        x = self.batch_norm(x)
        return self.dropout(x)


class TestInferenceMode:
    """Test cases for inference_mode context manager."""

    def test_sets_eval_mode(self):
        """Test that inference_mode sets network to eval mode."""
        network = SimpleNetwork()
        network.train()

        with inference_mode(network):
            assert not network.training, "Network should be in eval mode"

    def test_disables_gradients(self):
        """Test that inference_mode disables gradient computation."""
        network = SimpleNetwork()

        with inference_mode(network):
            assert not torch.is_grad_enabled(), "Gradients should be disabled"

    def test_restores_training_mode(self):
        """Test that inference_mode restores training mode after context."""
        network = SimpleNetwork()
        network.train()
        assert network.training, "Network should start in training mode"

        with inference_mode(network):
            pass

        assert network.training, "Network should be back in training mode"

    def test_preserves_eval_mode(self):
        """Test that inference_mode preserves eval mode if started in eval."""
        network = SimpleNetwork()
        network.eval()
        assert not network.training, "Network should start in eval mode"

        with inference_mode(network):
            pass

        assert not network.training, "Network should still be in eval mode"

    def test_exception_safety(self):
        """Test that mode is restored even if exception occurs."""
        network = SimpleNetwork()
        network.train()

        try:
            with inference_mode(network):
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert network.training, "Network should be restored to training mode after exception"

    def test_dropout_behavior(self):
        """Test that dropout behaves deterministically in inference mode."""
        network = SimpleNetwork()
        network.train()

        x = torch.randn(2, 10)

        # In training mode, dropout should be stochastic
        network.train()
        out1 = network(x)
        out2 = network(x)
        # With high probability, dropout should produce different outputs
        # (This test might rarely fail due to randomness, but it's very unlikely)

        # In inference mode, dropout should be deterministic (disabled)
        with inference_mode(network):
            out3 = network(x)
            out4 = network(x)
            torch.testing.assert_close(out3, out4, msg="Outputs should be identical in inference mode")


class TestTrainingMode:
    """Test cases for training_mode context manager."""

    def test_sets_train_mode(self):
        """Test that training_mode sets network to train mode."""
        network = SimpleNetwork()
        network.eval()

        with training_mode(network):
            assert network.training, "Network should be in train mode"

    def test_restores_eval_mode(self):
        """Test that training_mode restores eval mode after context."""
        network = SimpleNetwork()
        network.eval()
        assert not network.training, "Network should start in eval mode"

        with training_mode(network):
            pass

        assert not network.training, "Network should be back in eval mode"

    def test_preserves_train_mode(self):
        """Test that training_mode preserves train mode if started in train."""
        network = SimpleNetwork()
        network.train()
        assert network.training, "Network should start in train mode"

        with training_mode(network):
            pass

        assert network.training, "Network should still be in train mode"

    def test_exception_safety(self):
        """Test that mode is restored even if exception occurs."""
        network = SimpleNetwork()
        network.eval()

        try:
            with training_mode(network):
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert not network.training, "Network should be restored to eval mode after exception"


class TestNestedContexts:
    """Test cases for nested context managers."""

    def test_nested_inference_training(self):
        """Test nesting inference_mode inside training_mode."""
        network = SimpleNetwork()
        network.train()

        with inference_mode(network):
            assert not network.training, "Should be in eval mode"

            with training_mode(network):
                assert network.training, "Should be in train mode"

            assert not network.training, "Should be back in eval mode"

        assert network.training, "Should be back in original train mode"

    def test_nested_training_inference(self):
        """Test nesting training_mode inside inference_mode."""
        network = SimpleNetwork()
        network.eval()

        with training_mode(network):
            assert network.training, "Should be in train mode"

            with inference_mode(network):
                assert not network.training, "Should be in eval mode"

            assert network.training, "Should be back in train mode"

        assert not network.training, "Should be back in original eval mode"

    def test_multiple_nesting_levels(self):
        """Test multiple levels of nesting."""
        network = SimpleNetwork()
        network.train()

        with inference_mode(network):
            assert not network.training, "Level 1: eval"

            with training_mode(network):
                assert network.training, "Level 2: train"

                with inference_mode(network):
                    assert not network.training, "Level 3: eval"

                assert network.training, "Back to level 2: train"

            assert not network.training, "Back to level 1: eval"

        assert network.training, "Back to original: train"


class TestBatchNormBehavior:
    """Test that batch normalization behaves correctly in different modes."""

    def test_batchnorm_updates_in_train_mode(self):
        """Test that batch norm statistics update in training mode."""
        network = SimpleNetwork()
        network.train()

        # Get initial running mean
        initial_mean = network.batch_norm.running_mean.clone()

        # Forward pass in training mode
        x = torch.randn(4, 10)
        network.train()
        _ = network(x)

        # Running mean should have changed
        assert not torch.allclose(
            network.batch_norm.running_mean, initial_mean
        ), "Batch norm should update statistics in training mode"

    def test_batchnorm_frozen_in_eval_mode(self):
        """Test that batch norm statistics don't update in eval mode."""
        network = SimpleNetwork()
        network.eval()

        # Get initial running mean
        initial_mean = network.batch_norm.running_mean.clone()

        # Forward pass in eval mode
        x = torch.randn(4, 10)
        with inference_mode(network):
            _ = network(x)

        # Running mean should not have changed
        torch.testing.assert_close(
            network.batch_norm.running_mean,
            initial_mean,
            msg="Batch norm should not update statistics in eval mode",
        )


class TestRealWorldUsage:
    """Test real-world usage patterns."""

    def test_training_loop_pattern(self):
        """Test typical training loop pattern."""
        network = SimpleNetwork()
        optimizer = torch.optim.Adam(network.parameters())

        # Generate data in inference mode
        with inference_mode(network):
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)

        # Train in training mode
        network.train()
        for _ in range(3):
            optimizer.zero_grad()
            output = network(x)
            loss = ((output - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        assert network.training, "Network should be in training mode after training"

    def test_evaluation_pattern(self):
        """Test typical evaluation pattern."""
        network = SimpleNetwork()

        # Train briefly
        network.train()
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        optimizer = torch.optim.Adam(network.parameters())

        for _ in range(3):
            optimizer.zero_grad()
            output = network(x)
            loss = ((output - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Evaluate in inference mode
        with inference_mode(network):
            test_x = torch.randn(2, 10)
            output = network(test_x)
            assert output.shape == (2, 5), "Output shape should be correct"

        # Should be back in training mode
        assert network.training, "Network should be back in training mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
