"""Checkpoint management for training state persistence.

This module provides the CheckpointManager class for saving and loading
training checkpoints, including model weights, optimizer state, and
training metadata.

Example:
    manager = CheckpointManager(checkpoint_dir="checkpoints/experiment1")

    # Save checkpoint
    manager.save_checkpoint(
        path="iter_000100.pt",
        network=network,
        optimizer=optimizer,
        iteration=100,
        metrics={"loss": 0.5},
    )

    # Load checkpoint
    state = manager.load_checkpoint("iter_000100.pt")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Conditional imports for PyTorch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if TYPE_CHECKING:
    from _02_agents.neural.network import BeastyBarNetwork
    from _03_training.opponent_pool import OpponentPool
    from _03_training.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


def _ensure_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for checkpointing. Install with: pip install torch")


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    Handles persistence of training state including:
    - Model weights
    - Optimizer state
    - Training iteration and counters
    - Metrics history
    - RNG states for reproducibility
    - Opponent pool state (if enabled)

    Attributes:
        checkpoint_dir: Directory for saving checkpoints.
        tracker: Optional experiment tracker for logging artifacts.

    Example:
        manager = CheckpointManager("checkpoints/run1")
        manager.save_checkpoint(
            "iter_100.pt",
            network=model,
            optimizer=opt,
            iteration=100,
        )
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints.
            tracker: Optional experiment tracker for logging artifacts.
        """
        _ensure_torch()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker

    def save_checkpoint(
        self,
        path: str | Path,
        network: BeastyBarNetwork,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        config: dict[str, Any] | None = None,
        total_games_played: int = 0,
        total_transitions_collected: int = 0,
        metrics_history: list[dict[str, float]] | None = None,
        opponent_pool: OpponentPool | None = None,
        is_final: bool = False,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            path: Checkpoint filename or full path.
            network: Neural network to save.
            optimizer: Optimizer to save.
            iteration: Current training iteration.
            config: Training configuration dictionary.
            total_games_played: Total games played so far.
            total_transitions_collected: Total transitions collected.
            metrics_history: History of training metrics.
            opponent_pool: Optional opponent pool to save.
            is_final: Whether this is the final checkpoint.

        Returns:
            Path to saved checkpoint file.
        """
        # Determine full path
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoint_dir / checkpoint_path

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Build checkpoint dictionary
        checkpoint: dict[str, Any] = {
            "iteration": iteration,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "total_games_played": total_games_played,
            "total_transitions_collected": total_transitions_collected,
            "metrics_history": metrics_history or [],
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
        }

        if config is not None:
            checkpoint["config"] = config

        # Save opponent pool state if provided
        if opponent_pool is not None:
            checkpoint["opponent_pool"] = self._serialize_opponent_pool(opponent_pool)

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
        logger.info(f"Saved training checkpoint to {checkpoint_path}")

        # Also save config as JSON for easy inspection
        if config is not None:
            config_path = checkpoint_path.with_suffix(".json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        # Log artifact if tracker is available
        if self.tracker is not None:
            artifact_name = "final" if is_final else f"iter_{iteration:06d}"
            self.tracker.log_artifact(str(checkpoint_path), artifact_name)

        return checkpoint_path

    def load_checkpoint(
        self,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint from disk.

        Args:
            path: Path to checkpoint file.
            device: Device to load tensors to.

        Returns:
            Dictionary containing checkpoint data.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoint_dir / checkpoint_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return checkpoint

    def restore_training_state(
        self,
        path: str | Path,
        network: BeastyBarNetwork,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str = "cpu",
        opponent_pool: OpponentPool | None = None,
    ) -> dict[str, Any]:
        """Restore full training state from checkpoint.

        Restores model weights, optimizer state, RNG states, and optionally
        opponent pool state.

        Args:
            path: Path to checkpoint file.
            network: Network to restore weights to.
            optimizer: Optimizer to restore state to.
            device: Device for loading.
            opponent_pool: Optional opponent pool to restore.

        Returns:
            Dictionary with restored metadata:
            - iteration: Training iteration
            - total_games_played: Games played counter
            - total_transitions_collected: Transitions counter
            - metrics_history: Metrics history list

        Raises:
            FileNotFoundError: If checkpoint does not exist.
        """
        checkpoint = self.load_checkpoint(path, device)

        # Restore model state
        network.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore RNG states for reproducibility
        if "rng_state" in checkpoint:
            torch_rng_state = checkpoint["rng_state"]["torch"]
            if torch_rng_state.device != torch.device("cpu"):
                torch_rng_state = torch_rng_state.cpu()
            torch.set_rng_state(torch_rng_state)
            np.random.set_state(checkpoint["rng_state"]["numpy"])

        # Restore opponent pool state if present
        if opponent_pool is not None and "opponent_pool" in checkpoint:
            self._restore_opponent_pool(opponent_pool, checkpoint["opponent_pool"])
            logger.info(f"Restored opponent pool with {len(opponent_pool.checkpoints)} checkpoints")

        logger.info(f"Restored training state from iteration {checkpoint['iteration']}")

        return {
            "iteration": checkpoint["iteration"],
            "total_games_played": checkpoint.get("total_games_played", 0),
            "total_transitions_collected": checkpoint.get("total_transitions_collected", 0),
            "metrics_history": checkpoint.get("metrics_history", []),
            "config": checkpoint.get("config"),
        }

    def _serialize_opponent_pool(self, opponent_pool: OpponentPool) -> dict[str, Any]:
        """Serialize opponent pool state for checkpointing.

        Args:
            opponent_pool: Opponent pool to serialize.

        Returns:
            Dictionary representation of pool state.
        """
        return {
            "checkpoints": [
                {
                    "state_dict": cp.state_dict,
                    "iteration": cp.iteration,
                    "win_rate": cp.win_rate,
                }
                for cp in opponent_pool.checkpoints
            ],
            "sample_counts": {t.name: c for t, c in opponent_pool._sample_counts.items()},
        }

    def _restore_opponent_pool(
        self,
        opponent_pool: OpponentPool,
        pool_state: dict[str, Any],
    ) -> None:
        """Restore opponent pool state from checkpoint data.

        Args:
            opponent_pool: Opponent pool to restore to.
            pool_state: Serialized pool state from checkpoint.
        """
        from _03_training.opponent_pool import CheckpointEntry, OpponentType

        # Restore checkpoints
        opponent_pool.checkpoints.clear()
        for cp_data in pool_state.get("checkpoints", []):
            opponent_pool.checkpoints.append(
                CheckpointEntry(
                    state_dict=cp_data["state_dict"],
                    iteration=cp_data["iteration"],
                    win_rate=cp_data.get("win_rate"),
                )
            )

        # Restore sample counts
        for type_name, count in pool_state.get("sample_counts", {}).items():
            try:
                opp_type = OpponentType[type_name]
                opponent_pool._sample_counts[opp_type] = count
            except KeyError:
                pass

    def list_checkpoints(self, pattern: str = "*.pt") -> list[Path]:
        """List all checkpoint files in the checkpoint directory.

        Args:
            pattern: Glob pattern for checkpoint files.

        Returns:
            List of checkpoint file paths, sorted by modification time.
        """
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime)

    def get_latest_checkpoint(self, pattern: str = "iter_*.pt") -> Path | None:
        """Get the most recent checkpoint file.

        Args:
            pattern: Glob pattern for checkpoint files.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints(pattern)
        return checkpoints[-1] if checkpoints else None

    def save_for_inference(
        self,
        path: str | Path,
        network: BeastyBarNetwork,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Save model weights only for inference/deployment (~5 MB).

        This saves ONLY what's needed for inference:
        - Model weights (state_dict)
        - Network config (to reconstruct architecture)

        Does NOT include (saving ~500 MB):
        - Optimizer state
        - Opponent pool checkpoints
        - Training metrics
        - RNG state

        Use this for deploying to production or uploading to HuggingFace.

        Args:
            path: Output filename or full path.
            network: Neural network to save.
            config: Optional network config dict. If None, tries to get from network.

        Returns:
            Path to saved inference checkpoint.

        Example:
            manager.save_for_inference("model.pt", network)
            # Creates ~5 MB file with just weights
        """
        # Determine full path
        inference_path = Path(path)
        if not inference_path.is_absolute():
            inference_path = self.checkpoint_dir / inference_path

        inference_path.parent.mkdir(parents=True, exist_ok=True)

        # Get config from network if not provided
        if config is None and hasattr(network, "config"):
            config = network.config.to_dict()

        # Build minimal inference checkpoint
        inference_checkpoint: dict[str, Any] = {
            "model_state_dict": network.state_dict(),
            "checkpoint_type": "inference",  # Mark as inference-only
        }

        if config is not None:
            inference_checkpoint["network_config"] = config

        # Save with compression
        torch.save(inference_checkpoint, inference_path, pickle_protocol=4)

        # Log size for user awareness
        size_mb = inference_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved inference checkpoint to {inference_path} ({size_mb:.1f} MB)")

        return inference_path


# ============================================================================
# Standalone Functions for Inference Export
# ============================================================================


def export_for_inference(
    source: str | Path,
    output: str | Path,
    device: str = "cpu",
) -> Path:
    """Extract model weights from a training checkpoint for inference.

    Converts a full training checkpoint (~500 MB) to an inference-only
    checkpoint (~5 MB) by removing:
    - Optimizer state
    - Opponent pool (past checkpoints)
    - Training metrics and history
    - RNG state

    Args:
        source: Path to full training checkpoint.
        output: Path for inference checkpoint output.
        device: Device for loading checkpoint.

    Returns:
        Path to saved inference checkpoint.

    Example:
        # Convert existing training checkpoint
        export_for_inference("checkpoints/final.pt", "model.pt")

        # Upload model.pt to HuggingFace (~5 MB instead of ~500 MB)
    """
    _ensure_torch()

    source_path = Path(source)
    output_path = Path(output)

    if not source_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    # Load full checkpoint
    checkpoint = torch.load(source_path, map_location=device, weights_only=False)

    # Extract only what's needed for inference
    inference_checkpoint: dict[str, Any] = {
        "model_state_dict": checkpoint["model_state_dict"],
        "checkpoint_type": "inference",
    }

    # Preserve network config if available
    if "config" in checkpoint and "network_config" in checkpoint["config"]:
        inference_checkpoint["network_config"] = checkpoint["config"]["network_config"]
    elif "network_config" in checkpoint:
        inference_checkpoint["network_config"] = checkpoint["network_config"]

    # Save minimal checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_checkpoint, output_path, pickle_protocol=4)

    # Log sizes for comparison
    source_mb = source_path.stat().st_size / (1024 * 1024)
    output_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Exported inference checkpoint: {source_mb:.1f} MB -> {output_mb:.1f} MB "
        f"(saved {source_mb - output_mb:.1f} MB)"
    )

    return output_path


def load_for_inference(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load model weights for inference.

    Works with both inference checkpoints and full training checkpoints.
    Returns only what's needed to reconstruct the network.

    Args:
        path: Path to checkpoint (inference or training).
        device: Device to load weights to.

    Returns:
        Tuple of (state_dict, network_config) where network_config may be None.

    Example:
        state_dict, config = load_for_inference("model.pt")
        network = BeastyBarNetwork(NetworkConfig.from_dict(config))
        network.load_state_dict(state_dict)
    """
    _ensure_torch()

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Try to find network config
    network_config = None
    if "network_config" in checkpoint:
        network_config = checkpoint["network_config"]
    elif "config" in checkpoint and isinstance(checkpoint["config"], dict):
        network_config = checkpoint["config"].get("network_config")

    return state_dict, network_config


__all__ = [
    "CheckpointManager",
    "export_for_inference",
    "load_for_inference",
]
