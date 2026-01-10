"""Neural network utilities for model checkpointing, action sampling, and configuration.

This module provides:
- Model serialization (save/load checkpoints with versioning)
- Action sampling (stochastic and greedy selection from masked logits)
- Model factory utilities (configuration dataclass and convenience loaders)
- Device management and parameter counting utilities
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as functional

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Constants from action_space and observations modules
ACTION_DIM = 124  # From _01_simulator/action_space.py
OBSERVATION_DIM = 988  # From _01_simulator/observations.py


# ============================================================================
# Network Configuration
# ============================================================================


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for neural network architecture.

    All hyperparameters for the policy-value network are defined here.
    This dataclass is immutable (frozen) to ensure configs are not
    accidentally modified during training.

    Attributes:
        observation_dim: Input dimension (from observation encoding).
        action_dim: Output dimension for policy head.
        hidden_dim: Size of hidden layers in fusion network.
        num_heads: Number of attention heads in transformer layers.
        num_layers: Number of transformer encoder layers per zone.
        dropout: Dropout probability for regularization.
        species_embedding_dim: Dimension of learned species embeddings.
        card_feature_dim: Dimension of per-card feature vector.
        num_species: Number of distinct species (excluding 'unknown').
        max_queue_length: Maximum cards in queue zone.
        max_bar_length: Maximum cards in beasty_bar/thats_it zones.
        hand_size: Maximum cards in hand.
    """

    # Core dimensions
    observation_dim: int = OBSERVATION_DIM
    action_dim: int = ACTION_DIM

    # Network architecture
    # Defaults tuned for ~1.3M parameters (target: 500K-2M)
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.1
    species_embedding_dim: int = 32

    # Card encoding (for structured input processing)
    card_feature_dim: int = 17
    num_species: int = 12

    # Zone sizes
    max_queue_length: int = 5
    max_bar_length: int = 24
    hand_size: int = 4

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkConfig:
        """Create config from dictionary."""
        # Filter only known fields to handle forward compatibility
        import dataclasses

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


def default_config() -> NetworkConfig:
    """Return the default network configuration.

    Returns:
        NetworkConfig with sensible defaults for Beasty Bar training.
    """
    return NetworkConfig()


# ============================================================================
# Device Management
# ============================================================================


def get_device() -> torch.device:
    """Detect and return the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device for model placement.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(model: nn.Module, device: torch.device | str | None = None) -> nn.Module:
    """Move model to specified device.

    Args:
        model: PyTorch model to move.
        device: Target device. If None, auto-detects best device.

    Returns:
        Model on the specified device.
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    return model.to(device)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model.

    Args:
        model: PyTorch model to count parameters for.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# Model Serialization
# ============================================================================


@dataclass
class CheckpointData:
    """Container for checkpoint contents.

    Attributes:
        step: Training step number when checkpoint was saved.
        model_state: Model state dictionary.
        optimizer_state: Optimizer state dictionary (optional).
        config: Network configuration dictionary.
        metrics: Training metrics at checkpoint time (optional).
    """

    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    config: dict[str, Any]
    metrics: dict[str, Any] | None


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    path: str | Path,
    *,
    config: NetworkConfig | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save model checkpoint with training state.

    Checkpoints include:
    - Model weights
    - Optimizer state (if provided)
    - Training step number
    - Network configuration
    - Optional training metrics

    Args:
        model: Neural network model to save.
        optimizer: Optimizer to save state from (optional).
        step: Current training step for versioning.
        path: File path to save checkpoint (will add .pt if missing).
        config: Network configuration (optional, extracted from model if available).
        metrics: Dictionary of training metrics to include (optional).

    Returns:
        Path where checkpoint was saved.

    Raises:
        OSError: If unable to write to the specified path.
    """
    path = Path(path)

    # Ensure .pt extension
    if path.suffix != ".pt":
        path = path.with_suffix(".pt")

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint dictionary
    checkpoint: dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Save config if provided or try to extract from model
    if config is not None:
        checkpoint["config"] = config.to_dict()
    elif hasattr(model, "config") and isinstance(model.config, NetworkConfig):
        checkpoint["config"] = model.config.to_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    # Save with pickle protocol 4 for Python 3.8+ compatibility
    torch.save(checkpoint, path, pickle_protocol=4)

    logger.info(f"Saved checkpoint at step {step} to {path}")
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> CheckpointData:
    """Load model checkpoint and restore state.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to restore state to (optional).
        device: Device to load tensors to. If None, uses current model device.
        strict: Whether to strictly enforce state dict key matching.

    Returns:
        CheckpointData containing checkpoint contents.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint is incompatible with model.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Determine target device
    if device is None:
        # Try to infer from model
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Restore model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Restore optimizer state if provided
    optimizer_state = None
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_state = checkpoint["optimizer_state_dict"]

    step = checkpoint.get("step", 0)
    config = checkpoint.get("config", {})
    metrics = checkpoint.get("metrics")

    logger.info(f"Loaded checkpoint from step {step} at {path}")

    return CheckpointData(
        step=step,
        model_state=checkpoint["model_state_dict"],
        optimizer_state=optimizer_state,
        config=config,
        metrics=metrics,
    )


def load_network_from_checkpoint(
    path: str | Path,
    network_cls: type[nn.Module] | None = None,
    *,
    device: torch.device | str | None = None,
) -> tuple[nn.Module, NetworkConfig, int]:
    """Convenience function to load a network from checkpoint.

    This function creates a new network instance and loads weights from
    the checkpoint. Useful for inference or evaluation.

    Args:
        path: Path to checkpoint file.
        network_cls: Network class to instantiate. If None, attempts to
            import BeastyBarNetwork from network module.
        device: Device to place model on. If None, auto-detects.

    Returns:
        Tuple of (model, config, step) where:
            - model: Loaded network on specified device
            - config: NetworkConfig used for the model
            - step: Training step when checkpoint was saved

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ImportError: If network_cls is None and BeastyBarNetwork cannot be imported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint to extract config
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config_dict = checkpoint.get("config", {})
    # Handle nested network_config (PPO checkpoints) vs flat config (neural checkpoints)
    if "network_config" in config_dict:
        config_dict = config_dict["network_config"]
    config = NetworkConfig.from_dict(config_dict) if config_dict else default_config()

    # Import network class if not provided
    if network_cls is None:
        try:
            from _02_agents.neural.network import BeastyBarNetwork

            network_cls = BeastyBarNetwork
        except ImportError as e:
            raise ImportError(
                "Could not import BeastyBarNetwork. Either provide network_cls "
                "or ensure _02_agents.neural.network is available."
            ) from e

    # Create model with config
    model = network_cls(config)

    # Load weights
    checkpoint_data = load_checkpoint(path, model, device=device)

    # Move to device
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)

    return model, config, checkpoint_data.step


# ============================================================================
# Action Sampling
# ============================================================================


def sample_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    return_prob: bool = False,
) -> int | tuple[int, float]:
    """Sample action from masked policy logits.

    Applies the action mask to logits, scales by temperature, computes
    softmax probabilities, and samples from the resulting distribution.

    Args:
        logits: Raw policy logits of shape (action_dim,) or (1, action_dim).
        mask: Binary mask of shape (action_dim,) or (1, action_dim).
            1.0 = legal action, 0.0 = illegal action.
        temperature: Temperature for controlling exploration.
            Higher values increase randomness, lower values approach greedy.
            Must be positive.
        return_prob: If True, return tuple of (action, probability).
            Default False for backwards compatibility.

    Returns:
        If return_prob is False: Sampled action index (integer in [0, action_dim)).
        If return_prob is True: Tuple of (action_index, action_probability).

    Raises:
        ValueError: If temperature is not positive.
        ValueError: If no legal actions are available.

    Example:
        >>> logits = torch.randn(124)
        >>> mask = torch.zeros(124)
        >>> mask[0] = mask[5] = mask[10] = 1.0  # Only 3 legal actions
        >>> action = sample_action(logits, mask, temperature=1.0)
        >>> assert action in [0, 5, 10]
        >>> action, prob = sample_action(logits, mask, return_prob=True)
        >>> assert 0.0 <= prob <= 1.0
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Flatten to 1D if needed
    logits = logits.squeeze()
    mask = mask.squeeze()

    if logits.dim() != 1:
        raise ValueError(f"Expected 1D logits after squeeze, got shape {logits.shape}")

    # Check for legal actions
    if not mask.any():
        raise ValueError("No legal actions available (mask is all zeros)")

    # Apply mask: set illegal actions to very negative value
    masked_logits = torch.where(mask > 0, logits, torch.tensor(float("-inf"), device=logits.device))

    # Apply temperature scaling
    scaled_logits = masked_logits / temperature

    # Compute probabilities via softmax
    probs = functional.softmax(scaled_logits, dim=-1)

    # Handle NaN/Inf from unstable training - fall back to uniform over valid actions
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        probs = mask / mask.sum()

    # Sample from categorical distribution
    action_idx = torch.multinomial(probs, num_samples=1)
    action = int(action_idx.item())

    if return_prob:
        action_prob = float(probs[action].item())
        return action, action_prob

    return action


def greedy_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> int:
    """Return highest-logit legal action (deterministic selection).

    Args:
        logits: Raw policy logits of shape (action_dim,) or (1, action_dim).
        mask: Binary mask of shape (action_dim,) or (1, action_dim).
            1.0 = legal action, 0.0 = illegal action.

    Returns:
        Index of the legal action with highest logit.

    Raises:
        ValueError: If no legal actions are available.

    Example:
        >>> logits = torch.tensor([0.1, 0.5, 0.3, 0.9])
        >>> mask = torch.tensor([1.0, 1.0, 0.0, 1.0])  # actions 0, 1, 3 legal
        >>> action = greedy_action(logits, mask)
        >>> assert action == 3  # highest logit among legal actions
    """
    # Flatten to 1D if needed
    logits = logits.squeeze()
    mask = mask.squeeze()

    if logits.dim() != 1:
        raise ValueError(f"Expected 1D logits after squeeze, got shape {logits.shape}")

    # Check for legal actions
    if not mask.any():
        raise ValueError("No legal actions available (mask is all zeros)")

    # Apply mask: set illegal actions to very negative value
    masked_logits = torch.where(mask > 0, logits, torch.tensor(float("-inf"), device=logits.device))

    # Return argmax
    return int(torch.argmax(masked_logits).item())


def compute_action_probs(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute action probabilities from masked logits.

    Similar to sample_action but returns the full probability distribution
    instead of sampling. Useful for computing policy entropy or KL divergence.

    Args:
        logits: Raw policy logits of shape (..., action_dim).
        mask: Binary mask of shape (..., action_dim).
        temperature: Temperature for scaling.

    Returns:
        Probability tensor of same shape as logits with illegal actions
        having probability 0.

    Raises:
        ValueError: If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Apply mask: set illegal actions to very negative value
    masked_logits = torch.where(mask > 0, logits, torch.tensor(float("-inf"), device=logits.device))

    # Apply temperature and softmax
    scaled_logits = masked_logits / temperature
    probs = functional.softmax(scaled_logits, dim=-1)

    # Ensure illegal actions have exactly 0 probability
    probs = probs * mask

    # Renormalize to handle numerical issues
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    return probs


def batch_sample_actions(
    logits: torch.Tensor,
    masks: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample actions for a batch of states.

    Args:
        logits: Policy logits of shape (batch_size, action_dim).
        masks: Action masks of shape (batch_size, action_dim).
        temperature: Temperature for all samples.

    Returns:
        Tensor of shape (batch_size,) with sampled action indices.

    Raises:
        ValueError: If temperature is not positive.
        ValueError: If any state has no legal actions.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Check all states have legal actions
    legal_counts = masks.sum(dim=-1)
    if (legal_counts == 0).any():
        raise ValueError("Some states have no legal actions")

    # Apply mask
    masked_logits = torch.where(masks > 0, logits, torch.tensor(float("-inf"), device=logits.device))

    # Apply temperature and softmax
    scaled_logits = masked_logits / temperature
    probs = functional.softmax(scaled_logits, dim=-1)

    # Sample from each distribution
    actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return actions


def batch_greedy_actions(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Select greedy actions for a batch of states.

    Args:
        logits: Policy logits of shape (batch_size, action_dim).
        masks: Action masks of shape (batch_size, action_dim).

    Returns:
        Tensor of shape (batch_size,) with greedy action indices.

    Raises:
        ValueError: If any state has no legal actions.
    """
    # Check all states have legal actions
    legal_counts = masks.sum(dim=-1)
    if (legal_counts == 0).any():
        raise ValueError("Some states have no legal actions")

    # Apply mask
    masked_logits = torch.where(masks > 0, logits, torch.tensor(float("-inf"), device=logits.device))

    # Return argmax for each batch element
    return torch.argmax(masked_logits, dim=-1)


# ============================================================================
# Utility Functions
# ============================================================================


def seed_all(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_summary(model: nn.Module, config: NetworkConfig | None = None) -> str:
    """Generate a summary string for a model.

    Args:
        model: Model to summarize.
        config: Optional config to include in summary.

    Returns:
        Multi-line string with model summary.
    """
    lines = []
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append(f"Total parameters: {count_parameters(model, trainable_only=False):,}")
    lines.append(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    # Device info
    try:
        device = next(model.parameters()).device
        lines.append(f"Device: {device}")
    except StopIteration:
        lines.append("Device: (no parameters)")

    if config is not None:
        lines.append("Configuration:")
        for key, value in config.to_dict().items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


__all__ = [
    "ACTION_DIM",
    "OBSERVATION_DIM",
    "CheckpointData",
    "NetworkConfig",
    "batch_greedy_actions",
    "batch_sample_actions",
    "compute_action_probs",
    "count_parameters",
    "default_config",
    "get_device",
    "get_model_summary",
    "greedy_action",
    "load_checkpoint",
    "load_network_from_checkpoint",
    "move_to_device",
    "sample_action",
    "save_checkpoint",
    "seed_all",
]
