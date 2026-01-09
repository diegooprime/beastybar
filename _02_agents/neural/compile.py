"""Torch compile utilities for network optimization.

This module provides utilities for applying torch.compile() to neural networks
for 20-40% inference speedup on PyTorch 2.x.

Key features:
- Version checking for PyTorch 2.0+ requirement
- Graceful fallback if compilation fails
- Multiple compile modes for different use cases
- Support for dynamic batch sizes

Example:
    from _02_agents.neural.compile import maybe_compile_network

    network = BeastyBarNetwork(config)
    network = maybe_compile_network(
        network,
        compile_mode="reduce-overhead",
        dynamic=True,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

# Compile modes supported by torch.compile
CompileMode = Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]


def is_torch_compile_available() -> bool:
    """Check if torch.compile is available (PyTorch >= 2.0).

    Returns:
        True if torch.compile is available, False otherwise.
    """
    try:
        import torch

        # torch.compile was added in PyTorch 2.0
        version = torch.__version__
        major_version = int(version.split(".")[0])
        return major_version >= 2
    except (ImportError, ValueError, IndexError):
        return False


def _get_torch_version() -> str:
    """Get the current PyTorch version string.

    Internal helper function - not part of public API.

    Returns:
        PyTorch version string or "not installed" if unavailable.
    """
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "not installed"


def maybe_compile_network(
    network: "nn.Module",
    compile_mode: str | None = None,
    dynamic: bool = True,
    fullgraph: bool = False,
    disable_on_cpu: bool = True,
) -> "nn.Module":
    """Optionally apply torch.compile() to a network for inference speedup.

    This function provides a safe wrapper around torch.compile() that:
    - Checks PyTorch version compatibility (requires 2.0+)
    - Handles compilation failures gracefully
    - Supports multiple compilation modes
    - Optionally disables on CPU (where benefits are minimal)

    Args:
        network: Neural network module to potentially compile.
        compile_mode: Compilation mode. Options:
            - None: No compilation (returns network unchanged)
            - "default": Standard compilation, good balance
            - "reduce-overhead": Best for inference with small batches
            - "max-autotune": Maximum optimization, slower compile time
            - "max-autotune-no-cudagraphs": Max optimization without CUDA graphs
        dynamic: Whether to use dynamic shapes (handles variable batch sizes).
            Set to True for training/inference with varying batch sizes.
        fullgraph: Whether to require full graph compilation.
            Set to False for more flexibility with dynamic control flow.
        disable_on_cpu: If True, skip compilation on CPU devices where
            benefits are typically minimal or negative.

    Returns:
        Compiled network if compilation succeeds and is enabled,
        otherwise the original network unchanged.

    Example:
        >>> network = BeastyBarNetwork(config)
        >>> # For training (variable batch sizes)
        >>> network = maybe_compile_network(network, "reduce-overhead", dynamic=True)
        >>> # For inference (fixed batch sizes)
        >>> network = maybe_compile_network(network, "max-autotune", dynamic=False)
    """
    import torch

    # No compilation requested
    if compile_mode is None:
        return network

    # Check PyTorch version
    if not is_torch_compile_available():
        torch_version = _get_torch_version()
        logger.warning(
            f"torch.compile requires PyTorch >= 2.0, but found {torch_version}. "
            "Skipping compilation."
        )
        return network

    # Check if network is on CPU and compilation is disabled for CPU
    if disable_on_cpu:
        try:
            device = next(network.parameters()).device
            if device.type == "cpu":
                logger.info(
                    "Skipping torch.compile on CPU device (disable_on_cpu=True). "
                    "Set disable_on_cpu=False to force compilation."
                )
                return network
        except StopIteration:
            # No parameters in network, cannot determine device
            pass

    # Validate compile mode
    valid_modes = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
    if compile_mode not in valid_modes:
        logger.warning(
            f"Unknown compile_mode '{compile_mode}'. "
            f"Valid options: {valid_modes}. Skipping compilation."
        )
        return network

    # Attempt compilation
    try:
        compiled_network = torch.compile(
            network,
            mode=compile_mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )
        logger.info(
            f"Successfully compiled network with mode='{compile_mode}', "
            f"dynamic={dynamic}, fullgraph={fullgraph}"
        )
        return compiled_network

    except Exception as e:
        # Log the error but don't fail - return original network
        logger.warning(
            f"torch.compile failed with error: {e}. "
            "Continuing with uncompiled network."
        )
        return network


def compile_for_training(
    network: "nn.Module",
    enabled: bool = True,
) -> "nn.Module":
    """Compile network with settings optimized for training.

    Training-optimized settings:
    - mode="reduce-overhead": Good balance for training loop
    - dynamic=True: Handle variable batch sizes
    - fullgraph=False: Allow dynamic control flow

    Args:
        network: Neural network to compile.
        enabled: Whether to enable compilation.

    Returns:
        Compiled network if enabled and successful, otherwise original network.
    """
    if not enabled:
        return network

    return maybe_compile_network(
        network,
        compile_mode="reduce-overhead",
        dynamic=True,
        fullgraph=False,
        disable_on_cpu=True,
    )


def compile_for_inference(
    network: "nn.Module",
    enabled: bool = True,
    batch_size_fixed: bool = False,
) -> "nn.Module":
    """Compile network with settings optimized for inference.

    Inference-optimized settings:
    - mode="max-autotune" for fixed batch sizes
    - mode="reduce-overhead" for variable batch sizes
    - dynamic based on batch_size_fixed parameter

    Args:
        network: Neural network to compile.
        enabled: Whether to enable compilation.
        batch_size_fixed: If True, uses more aggressive optimization
            assuming batch size won't change.

    Returns:
        Compiled network if enabled and successful, otherwise original network.
    """
    if not enabled:
        return network

    if batch_size_fixed:
        return maybe_compile_network(
            network,
            compile_mode="max-autotune",
            dynamic=False,
            fullgraph=False,
            disable_on_cpu=True,
        )
    else:
        return maybe_compile_network(
            network,
            compile_mode="reduce-overhead",
            dynamic=True,
            fullgraph=False,
            disable_on_cpu=True,
        )


__all__ = [
    "CompileMode",
    "compile_for_inference",
    "compile_for_training",
    "is_torch_compile_available",
    "maybe_compile_network",
]
