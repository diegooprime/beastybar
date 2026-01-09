"""Training utilities for neural network training.

This module provides utility functions and context managers for neural network
training operations, including proper eval()/train() mode switching.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


@contextmanager
def inference_mode(network: nn.Module):
    """Context manager for network inference.

    Sets the network to eval mode and disables gradient computation.
    Restores the original training mode after the context exits.

    This ensures dropout layers and batch normalization behave correctly
    during inference (deterministic mode), while preserving the training
    state for subsequent training operations.

    Args:
        network: Neural network to set to eval mode.

    Yields:
        None

    Example:
        >>> with inference_mode(network):
        ...     # Network is in eval mode, gradients disabled
        ...     trajectories = generate_games(network, num_games=100)
        >>> # Network is back to original mode (likely train mode)
    """
    import torch

    was_training = network.training
    network.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            network.train()


@contextmanager
def training_mode(network: nn.Module):
    """Context manager for network training.

    Sets the network to train mode and enables gradient computation.
    Restores the original mode after the context exits.

    This ensures dropout layers and batch normalization behave correctly
    during training (stochastic mode), while preserving the eval state
    for subsequent inference operations.

    Args:
        network: Neural network to set to train mode.

    Yields:
        None

    Example:
        >>> with training_mode(network):
        ...     # Network is in train mode, gradients enabled
        ...     loss.backward()
        ...     optimizer.step()
        >>> # Network is back to original mode
    """
    was_training = network.training
    network.train()
    try:
        yield
    finally:
        if not was_training:
            network.eval()


__all__ = [
    "inference_mode",
    "training_mode",
]
