"""AlphaZero-style MCTS training for neural networks.

This module implements the AlphaZero training loop:
1. Generate self-play games using MCTS guided by current network
2. Train network to match MCTS-improved policies and game outcomes
3. Loss = policy_loss + value_loss - entropy_bonus
   - Policy loss: Cross-entropy between network policy and MCTS visit distribution
   - Value loss: MSE between network value and game outcome
   - Entropy bonus: Encourages exploration

Key differences from PPO training:
- No advantage estimation (GAE) - uses terminal game outcomes directly
- Policy target is MCTS visit distribution, not single action
- No clipped surrogate objective - uses standard cross-entropy
- Simpler and more stable than PPO for two-player games

Example:
    config = MCTSTrainerConfig(
        total_iterations=1000,
        games_per_iteration=100,
        mcts_config=MCTSConfig(num_simulations=800),
    )
    trainer = MCTSTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Conditional imports for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from _02_agents.neural.compile import maybe_compile_network
from _02_agents.neural.utils import NetworkConfig, get_device, seed_all
from _03_training.mcts_self_play import (
    DEFAULT_MCTS_CONFIG,
    MCTSConfig,
    MCTSTrainingData,
    compute_mcts_stats,
    generate_mcts_games_parallel,
    generate_mcts_games_with_opponents,
    mcts_trajectories_to_training_data,
)
from _03_training.opponent_pool import (
    OpponentConfig,
    OpponentPool,
    OpponentType,
    SampledOpponent,
)
from _03_training.tracking import ExperimentTracker, create_tracker

if TYPE_CHECKING:
    from _02_agents.neural.network import BeastyBarNetwork

logger = logging.getLogger(__name__)


def _ensure_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install with: pip install torch")


# ============================================================================
# MCTS Training Configuration
# ============================================================================


@dataclass
class MCTSTrainerConfig:
    """Configuration for AlphaZero-style MCTS training.

    Attributes:
        network_config: Configuration for neural network architecture.
        mcts_config: Configuration for MCTS search parameters.
        games_per_iteration: Number of self-play games per training iteration.
        total_iterations: Total number of training iterations.
        batch_size: Batch size for network training.
        epochs_per_iteration: Number of training epochs per iteration.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 weight decay regularization.
        value_loss_weight: Weight of value loss in total loss.
        policy_loss_weight: Weight of policy loss in total loss.
        entropy_bonus_weight: Weight of entropy bonus in total loss.
        max_grad_norm: Maximum gradient norm for clipping (0 = no clipping).
        checkpoint_frequency: Save checkpoint every N iterations.
        eval_frequency: Evaluate against baselines every N iterations.
        lr_warmup_iterations: Number of iterations for learning rate warmup.
        lr_decay: Learning rate decay type ("linear", "cosine", "none").
        seed: Random seed for reproducibility.
        device: Device for training ("cpu", "cuda", "mps", "auto").
        experiment_name: Name for experiment tracking.
        checkpoint_dir: Directory for saving checkpoints.
        log_frequency: Log metrics every N iterations.
        buffer_size: Maximum number of samples to keep in replay buffer.
        min_buffer_size: Minimum samples before training starts.
        warmstart_checkpoint: Path to PPO warm-start checkpoint to load.
        warmstart_reset_optimizer: Reset optimizer state after loading warm-start.
        warmstart_reset_iteration: Reset iteration counter after loading warm-start.
    """

    # Network configuration
    network_config: NetworkConfig = field(default_factory=NetworkConfig)

    # MCTS configuration (using the new MCTSConfig class)
    mcts_config: MCTSConfig = field(default_factory=lambda: DEFAULT_MCTS_CONFIG)

    # Self-play settings
    games_per_iteration: int = 64

    # Training schedule
    total_iterations: int = 200
    batch_size: int = 512
    epochs_per_iteration: int = 4

    # Optimization settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.5
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    entropy_bonus_weight: float = 0.01

    # Checkpointing and evaluation
    checkpoint_frequency: int = 10
    eval_frequency: int = 5

    # Learning rate schedule
    lr_warmup_iterations: int = 10
    lr_decay: str = "cosine"  # "linear", "cosine", "none"

    # Misc settings
    seed: int = 42
    device: str = "auto"  # "cpu", "cuda", "mps", "auto"
    experiment_name: str = "beastybar_mcts"
    checkpoint_dir: str = "checkpoints"
    log_frequency: int = 1

    # Replay buffer settings
    buffer_size: int = 100_000
    min_buffer_size: int = 5000

    # Opponent diversity (prevents self-play collapse)
    opponent_config: OpponentConfig = field(default_factory=OpponentConfig)
    checkpoint_to_pool_frequency: int = 20  # Add checkpoint to pool every N iterations

    # Warm-start from PPO checkpoint (for transitioning from PPO to MCTS training)
    warmstart_checkpoint: str | None = None  # Path to PPO checkpoint
    warmstart_reset_optimizer: bool = True  # Reset optimizer (different loss landscape)
    warmstart_reset_iteration: bool = True  # Reset iteration counter

    # Torch compile settings (PyTorch 2.0+)
    # Enables torch.compile() for 20-40% inference speedup
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        result = asdict(self)
        result["network_config"] = self.network_config.to_dict()
        result["mcts_config"] = self.mcts_config.to_dict()
        result["opponent_config"] = self.opponent_config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCTSTrainerConfig:
        """Create configuration from dictionary."""
        data = data.copy()

        # Handle nested network config
        if "network_config" in data:
            if isinstance(data["network_config"], dict):
                data["network_config"] = NetworkConfig.from_dict(data["network_config"])
        else:
            data["network_config"] = NetworkConfig()

        # Handle nested mcts config
        if "mcts_config" in data:
            if isinstance(data["mcts_config"], dict):
                data["mcts_config"] = MCTSConfig.from_dict(data["mcts_config"])
        else:
            data["mcts_config"] = DEFAULT_MCTS_CONFIG

        # Handle nested opponent config
        if "opponent_config" in data:
            if isinstance(data["opponent_config"], dict):
                data["opponent_config"] = OpponentConfig.from_dict(data["opponent_config"])
        else:
            data["opponent_config"] = OpponentConfig()

        # Filter to known fields
        import dataclasses

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MCTSTrainerConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.games_per_iteration <= 0:
            raise ValueError(f"games_per_iteration must be positive, got {self.games_per_iteration}")
        if self.total_iterations <= 0:
            raise ValueError(f"total_iterations must be positive, got {self.total_iterations}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.lr_decay not in ("linear", "cosine", "none"):
            raise ValueError(f"lr_decay must be 'linear', 'cosine', or 'none', got {self.lr_decay}")


# ============================================================================
# Learning Rate Scheduling
# ============================================================================


def get_learning_rate(
    iteration: int,
    total_iterations: int,
    base_lr: float,
    warmup_iterations: int,
    decay_type: str,
) -> float:
    """Compute learning rate for a given iteration.

    Same as trainer.py for consistency.
    """
    if iteration < warmup_iterations:
        # Linear warmup from 0 to base_lr
        return base_lr * (iteration + 1) / warmup_iterations

    # Post-warmup: apply decay
    if decay_type == "none":
        return base_lr

    # Calculate progress through decay phase
    decay_iterations = total_iterations - warmup_iterations
    decay_progress = (iteration - warmup_iterations) / max(decay_iterations, 1)
    decay_progress = min(decay_progress, 1.0)

    if decay_type == "linear":
        return base_lr * (1.0 - decay_progress)
    elif decay_type == "cosine":
        import math

        return base_lr * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all parameter groups in optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ============================================================================
# Loss Functions
# ============================================================================


def policy_loss(
    predicted_logits: torch.Tensor,
    target_policy: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss between predicted and target policies.

    Args:
        predicted_logits: Network policy logits, shape (batch, action_dim).
        target_policy: MCTS visit distribution, shape (batch, action_dim).
        action_mask: Legal action mask, shape (batch, action_dim).

    Returns:
        Scalar policy loss (cross-entropy).
    """
    _ensure_torch()

    # Apply action mask to logits
    masked_logits = torch.where(
        action_mask > 0,
        predicted_logits,
        torch.tensor(float("-inf"), device=predicted_logits.device),
    )

    # Compute log probabilities
    log_probs = F.log_softmax(masked_logits, dim=-1)

    # Handle -inf values (from masked actions)
    log_probs = torch.where(
        torch.isinf(log_probs),
        torch.zeros_like(log_probs),
        log_probs,
    )

    # Cross-entropy: -sum(target * log(pred))
    loss = -(target_policy * log_probs).sum(dim=-1).mean()

    return loss


def value_loss(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss between predicted and target values.

    Args:
        predicted_values: Network value predictions, shape (batch,).
        target_values: Game outcome values, shape (batch,).

    Returns:
        Scalar value loss (MSE).
    """
    _ensure_torch()
    return F.mse_loss(predicted_values, target_values)


def entropy_bonus(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute policy entropy for exploration bonus.

    Same as ppo.py for consistency.
    """
    _ensure_torch()

    # Apply action mask
    masked_logits = torch.where(
        action_mask > 0,
        logits,
        torch.tensor(float("-inf"), device=logits.device),
    )

    # Compute probabilities
    probs = F.softmax(masked_logits, dim=-1)
    log_probs = F.log_softmax(masked_logits, dim=-1)

    # Handle numerical issues
    log_probs = torch.where(
        torch.isinf(log_probs),
        torch.zeros_like(log_probs),
        log_probs,
    )

    # Entropy: H = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy.mean()


# ============================================================================
# MCTS Trainer Class
# ============================================================================


class MCTSTrainer:
    """AlphaZero-style MCTS trainer for neural networks.

    Orchestrates the complete training loop:
    1. Generate self-play games with MCTS
    2. Train network on collected data
    3. Checkpoint and evaluate periodically

    Attributes:
        config: Training configuration.
        network: Neural network model.
        optimizer: PyTorch optimizer.
        tracker: Experiment tracker for logging.
        device: Training device.
    """

    def __init__(
        self,
        config: MCTSTrainerConfig,
        tracker: ExperimentTracker | None = None,
        network: BeastyBarNetwork | None = None,
    ) -> None:
        """Initialize MCTS trainer.

        Args:
            config: Training configuration.
            tracker: Optional experiment tracker.
            network: Optional pre-initialized network.
        """
        _ensure_torch()
        config.validate()

        self.config = config
        self._iteration = 0
        self._total_games_played = 0
        self._total_steps_collected = 0
        self._training_start_time: float | None = None

        # Set random seeds
        seed_all(config.seed)

        # Determine device
        if config.device == "auto":
            self._device = get_device()
        else:
            self._device = torch.device(config.device)
        logger.info(f"Using device: {self._device}")

        # Create or use provided network
        if network is not None:
            self.network = network.to(self._device)
        else:
            from _02_agents.neural.network import BeastyBarNetwork

            self.network = BeastyBarNetwork(config.network_config).to(self._device)

        # Apply torch.compile if enabled (PyTorch 2.0+)
        # Compilation must happen after moving to device but before optimizer creation
        if config.torch_compile:
            self.network = maybe_compile_network(
                self.network,
                compile_mode=config.torch_compile_mode,
                dynamic=True,  # Handle variable batch sizes during training
            )

        # Create optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training buffer for replay
        self._training_buffer: MCTSTrainingData | None = None

        # Create or use provided tracker
        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = create_tracker(
                backend="console",
                project="beastybar",
                run_name=config.experiment_name,
                config=config.to_dict(),
            )

        # Create checkpoint directory
        self._checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics history
        self._metrics_history: list[dict[str, float]] = []

        # Initialize opponent pool for diverse training
        self._opponent_pool = OpponentPool(
            config=config.opponent_config,
            seed=config.seed,
        )

        # Network for checkpoint opponents (loaded on demand)
        self._opponent_network: BeastyBarNetwork | None = None

        logger.info(f"MCTS Trainer initialized with {self.network.count_parameters():,} parameters")
        logger.info(f"MCTS config: {config.mcts_config.num_simulations} simulations, "
                   f"temp={config.mcts_config.temperature}, "
                   f"temp_drop={config.mcts_config.temperature_drop_move}")
        logger.info(f"Opponent diversity: self={config.opponent_config.current_weight:.0%}, "
                   f"checkpoint={config.opponent_config.checkpoint_weight:.0%}, "
                   f"random={config.opponent_config.random_weight:.0%}, "
                   f"heuristic={config.opponent_config.heuristic_weight:.0%}")

        # Load warm-start checkpoint if configured
        if config.warmstart_checkpoint:
            self.load_warmstart_checkpoint(
                checkpoint_path=config.warmstart_checkpoint,
                reset_optimizer=config.warmstart_reset_optimizer,
                reset_iteration=config.warmstart_reset_iteration,
            )

    @property
    def current_iteration(self) -> int:
        """Return current training iteration."""
        return self._iteration

    @property
    def device(self) -> torch.device:
        """Return training device."""
        return self._device

    def load_warmstart_checkpoint(
        self,
        checkpoint_path: str | Path,
        reset_optimizer: bool = True,
        reset_iteration: bool = True,
    ) -> dict[str, Any]:
        """Load a PPO warm-start checkpoint into the MCTS trainer.

        This method allows transitioning from PPO training to MCTS refinement
        by loading a pre-trained value network. The optimizer state is typically
        reset because the MCTS loss landscape differs from PPO.

        Args:
            checkpoint_path: Path to the PPO checkpoint file.
            reset_optimizer: If True, reset optimizer state (recommended for
                different loss landscape between PPO and MCTS).
            reset_iteration: If True, reset iteration counter to 0.

        Returns:
            Dictionary with loading statistics:
                - source_iteration: Original iteration from checkpoint
                - parameters_loaded: Number of parameters loaded
                - architecture_match: Whether architecture matched exactly
                - value_mean: Mean of value head output (for validation)
                - value_std: Std of value head output (for validation)

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            RuntimeError: If checkpoint architecture does not match.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading warm-start checkpoint from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

        # Extract model state (handle both MCTS and PPO checkpoint formats)
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        elif "network_state_dict" in checkpoint:
            model_state = checkpoint["network_state_dict"]
        else:
            raise RuntimeError(
                f"Checkpoint does not contain model state. "
                f"Keys found: {list(checkpoint.keys())}"
            )

        # Validate architecture by checking state dict keys
        current_keys = set(self.network.state_dict().keys())
        checkpoint_keys = set(model_state.keys())

        missing_keys = current_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - current_keys
        architecture_match = len(missing_keys) == 0 and len(unexpected_keys) == 0

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        if not architecture_match:
            logger.warning(
                "Architecture mismatch detected. Attempting partial load with strict=False."
            )

        # Load model state
        try:
            self.network.load_state_dict(model_state, strict=architecture_match)
        except RuntimeError as e:
            # Try non-strict loading if strict fails
            if architecture_match:
                logger.warning(f"Strict loading failed: {e}. Retrying with strict=False.")
                self.network.load_state_dict(model_state, strict=False)
            else:
                raise

        # Extract source iteration for logging
        source_iteration = checkpoint.get("iteration", checkpoint.get("step", -1))

        # Reset optimizer state if requested
        if reset_optimizer:
            logger.info("Resetting optimizer state for MCTS loss landscape")
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            # Try to load optimizer state if available
            if "optimizer_state_dict" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    logger.info("Loaded optimizer state from checkpoint")
                except ValueError as e:
                    logger.warning(f"Could not load optimizer state: {e}")

        # Reset iteration counter if requested
        if reset_iteration:
            self._iteration = 0
            logger.info("Reset iteration counter to 0")
        else:
            self._iteration = source_iteration if source_iteration >= 0 else 0

        # Validate network by computing statistics on random inputs
        stats = self._validate_network_after_load()

        # Log summary
        num_params = sum(p.numel() for p in self.network.parameters())
        logger.info(
            f"Warm-start loaded successfully:\n"
            f"  Source iteration: {source_iteration}\n"
            f"  Parameters: {num_params:,}\n"
            f"  Architecture match: {architecture_match}\n"
            f"  Value head stats: mean={stats['value_mean']:.4f}, std={stats['value_std']:.4f}"
        )

        return {
            "source_iteration": source_iteration,
            "parameters_loaded": num_params,
            "architecture_match": architecture_match,
            "value_mean": stats["value_mean"],
            "value_std": stats["value_std"],
        }

    def _validate_network_after_load(self) -> dict[str, float]:
        """Validate network produces reasonable outputs after loading.

        Runs a forward pass with random inputs to check value head statistics.

        Returns:
            Dictionary with value_mean and value_std.
        """
        self.network.eval()
        with torch.no_grad():
            # Generate random observations (batch of 32)
            batch_size = 32
            obs_dim = self.config.network_config.observation_dim
            action_dim = self.config.network_config.action_dim

            random_obs = torch.randn(batch_size, obs_dim, device=self._device)
            random_mask = torch.ones(batch_size, action_dim, device=self._device)

            # Forward pass
            _, values = self.network(random_obs, random_mask)
            values = values.squeeze(-1)

            return {
                "value_mean": values.mean().item(),
                "value_std": values.std().item(),
            }

    def _get_current_lr(self) -> float:
        """Get learning rate for current iteration."""
        return get_learning_rate(
            iteration=self._iteration,
            total_iterations=self.config.total_iterations,
            base_lr=self.config.learning_rate,
            warmup_iterations=self.config.lr_warmup_iterations,
            decay_type=self.config.lr_decay,
        )

    def _compute_parallel_games(self) -> int:
        """Compute optimal number of parallel games based on configuration.

        Higher values improve GPU utilization but require more memory.

        Returns:
            Number of games to run in parallel.
        """
        return min(32, max(8, self.config.games_per_iteration // 2))

    def _generate_games_for_opponent(
        self, sampled: SampledOpponent, parallel_games: int
    ) -> list:
        """Generate games against a specific opponent type.

        Args:
            sampled: Sampled opponent from pool.
            parallel_games: Number of games to run in parallel.

        Returns:
            List of game trajectories.
        """
        if sampled.opponent_type == OpponentType.CURRENT:
            # Pure self-play: both players use current network
            return generate_mcts_games_parallel(
                network=self.network,
                num_games=self.config.games_per_iteration,
                config=self.config.mcts_config,
                device=self._device,
                parallel_games=parallel_games,
            )
        elif sampled.opponent_type == OpponentType.CHECKPOINT:
            # Play against past checkpoint
            opponent_network = self._get_or_create_opponent_network(sampled)
            return generate_mcts_games_with_opponents(
                network=self.network,
                num_games=self.config.games_per_iteration,
                opponent_network=opponent_network,
                config=self.config.mcts_config,
                device=self._device,
                parallel_games=parallel_games,
            )
        else:
            # Play against simple agent (random or heuristic)
            return generate_mcts_games_with_opponents(
                network=self.network,
                num_games=self.config.games_per_iteration,
                opponent_agent=sampled.agent,
                config=self.config.mcts_config,
                device=self._device,
                parallel_games=parallel_games,
            )

    def _log_generation_stats(self, trajectories: list, gen_time: float) -> None:
        """Compute and log game generation statistics.

        Args:
            trajectories: List of game trajectories.
            gen_time: Time taken to generate games.
        """
        stats = compute_mcts_stats(trajectories)
        logger.info(
            f"Generated {stats.games_played} games in {gen_time:.1f}s "
            f"({stats.total_steps} samples, "
            f"P0={stats.p0_win_rate:.1%}, "
            f"avg_len={stats.avg_game_length:.1f})"
        )
        self._total_games_played += stats.games_played
        self._total_steps_collected += stats.total_steps

    def _generate_mcts_games(self) -> MCTSTrainingData:
        """Generate self-play games using MCTS and convert to training data.

        Uses parallel game generation with BatchMCTS for maximum GPU efficiency.
        Uses OpponentPool for diverse training opponents to prevent self-play collapse.
        Expected speedup: 10-50x over sequential generation on GPU.

        Returns:
            MCTSTrainingData containing observations, policies, and values.
        """
        # Set network to evaluation mode for inference
        self.network.eval()

        parallel_games = self._compute_parallel_games()
        logger.info(
            f"Generating {self.config.games_per_iteration} MCTS games "
            f"(parallel={parallel_games})..."
        )

        gen_start = time.time()

        # Sample opponent from pool for diversity
        sampled = self._opponent_pool.sample_opponent()
        logger.info(f"Opponent: {sampled.name}")

        # Generate games
        trajectories = self._generate_games_for_opponent(sampled, parallel_games)

        # Log stats and update counters
        gen_time = time.time() - gen_start
        self._log_generation_stats(trajectories, gen_time)

        # Convert to training data
        return mcts_trajectories_to_training_data(trajectories)

    def _get_or_create_opponent_network(
        self, sampled: SampledOpponent,
    ) -> BeastyBarNetwork:
        """Get or create opponent network from checkpoint.

        Reuses network instance when possible to reduce memory allocation.

        Args:
            sampled: Sampled opponent with checkpoint state.

        Returns:
            Network loaded with checkpoint weights.
        """
        from _02_agents.neural.network import BeastyBarNetwork

        if self._opponent_network is None:
            # Create new opponent network with same config
            self._opponent_network = BeastyBarNetwork(self.config.network_config)
            self._opponent_network = self._opponent_network.to(self._device)

        # Load checkpoint weights
        if sampled.network_state is not None:
            self._opponent_network.load_state_dict(sampled.network_state)
        self._opponent_network.eval()

        return self._opponent_network

    def _add_to_buffer(self, new_data: MCTSTrainingData) -> None:
        """Add new training data to buffer, managing size limit.

        Args:
            new_data: New training data to add.
        """
        if self._training_buffer is None or len(self._training_buffer) == 0:
            self._training_buffer = new_data
            return

        # Combine with existing data
        combined_obs = np.concatenate([self._training_buffer.observations, new_data.observations])
        combined_masks = np.concatenate([self._training_buffer.action_masks, new_data.action_masks])
        combined_policies = np.concatenate(
            [self._training_buffer.policy_targets, new_data.policy_targets]
        )
        combined_values = np.concatenate(
            [self._training_buffer.value_targets, new_data.value_targets]
        )

        # Trim to buffer size (keep most recent)
        if len(combined_obs) > self.config.buffer_size:
            start_idx = len(combined_obs) - self.config.buffer_size
            combined_obs = combined_obs[start_idx:]
            combined_masks = combined_masks[start_idx:]
            combined_policies = combined_policies[start_idx:]
            combined_values = combined_values[start_idx:]

        self._training_buffer = MCTSTrainingData(
            observations=combined_obs,
            action_masks=combined_masks,
            policy_targets=combined_policies,
            value_targets=combined_values,
        )

    def _prepare_training_tensors(
        self, training_data: MCTSTrainingData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert training data to shuffled tensors on device.

        Args:
            training_data: MCTSTrainingData containing samples.

        Returns:
            Tuple of (observations, action_masks, policy_targets, value_targets) tensors.
        """
        # Shuffle data
        training_data = training_data.shuffle()

        # Convert to tensors
        observations = torch.from_numpy(training_data.observations).float().to(self._device)
        action_masks = torch.from_numpy(training_data.action_masks).float().to(self._device)
        policy_targets = torch.from_numpy(training_data.policy_targets).float().to(self._device)
        value_targets = torch.from_numpy(training_data.value_targets).float().to(self._device)

        return observations, action_masks, policy_targets, value_targets

    def _compute_mcts_losses(
        self,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
        policy_targets: torch.Tensor,
        value_targets: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute MCTS training losses.

        Args:
            policy_logits: Network policy output.
            values: Network value output (squeezed).
            policy_targets: Target MCTS policy distribution.
            value_targets: Target game outcome values.
            action_masks: Legal action masks.

        Returns:
            Tuple of (policy_loss, value_loss, entropy_bonus, total_loss).
        """
        p_loss = policy_loss(
            predicted_logits=policy_logits,
            target_policy=policy_targets,
            action_mask=action_masks,
        )

        v_loss = value_loss(
            predicted_values=values,
            target_values=value_targets,
        )

        ent_bonus = entropy_bonus(
            logits=policy_logits,
            action_mask=action_masks,
        )

        # Combined loss
        total_loss = (
            self.config.policy_loss_weight * p_loss
            + self.config.value_loss_weight * v_loss
            - self.config.entropy_bonus_weight * ent_bonus
        )

        return p_loss, v_loss, ent_bonus, total_loss

    def _apply_mcts_gradient_step(self, total_loss: torch.Tensor) -> None:
        """Apply gradient update with optional clipping.

        Args:
            total_loss: Loss tensor to backpropagate.
        """
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

    def _run_mcts_training_loop(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        policy_targets: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> tuple[dict[str, float], int]:
        """Run training epochs over the data.

        Args:
            observations: Observation tensor.
            action_masks: Action mask tensor.
            policy_targets: Policy target tensor.
            value_targets: Value target tensor.

        Returns:
            Tuple of (accumulated_metrics, num_updates).
        """
        num_samples = len(observations)
        metrics_accum = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        num_updates = 0

        for _epoch in range(self.config.epochs_per_iteration):
            # Shuffle indices for this epoch
            indices = torch.randperm(num_samples, device=self._device)

            # Iterate through minibatches
            for start_idx in range(0, num_samples, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # Extract minibatch
                obs_batch = observations[batch_indices]
                mask_batch = action_masks[batch_indices]
                policy_batch = policy_targets[batch_indices]
                value_batch = value_targets[batch_indices]

                # Forward pass
                policy_logits, values = self.network(obs_batch, mask_batch)
                values = values.squeeze(-1)

                # Compute losses
                p_loss, v_loss, ent_bonus, total_loss = self._compute_mcts_losses(
                    policy_logits, values, policy_batch, value_batch, mask_batch
                )

                # Optimization step
                self._apply_mcts_gradient_step(total_loss)

                # Accumulate metrics
                metrics_accum["policy_loss"] += p_loss.item()
                metrics_accum["value_loss"] += v_loss.item()
                metrics_accum["entropy"] += ent_bonus.item()
                metrics_accum["total_loss"] += total_loss.item()
                num_updates += 1

        return metrics_accum, num_updates

    def _train_on_data(self, training_data: MCTSTrainingData) -> dict[str, float]:
        """Train network on collected MCTS data.

        Args:
            training_data: MCTSTrainingData containing samples.

        Returns:
            Dictionary of training metrics.
        """
        num_samples = len(training_data)
        if num_samples < self.config.min_buffer_size:
            logger.warning(
                f"Buffer size ({num_samples}) below minimum "
                f"({self.config.min_buffer_size}), skipping training"
            )
            return {}

        # Ensure network is in training mode
        self.network.train()

        # Update learning rate
        current_lr = self._get_current_lr()
        set_learning_rate(self.optimizer, current_lr)

        # Prepare tensors
        observations, action_masks, policy_targets, value_targets = self._prepare_training_tensors(training_data)

        # Run training loop
        metrics_accum, num_updates = self._run_mcts_training_loop(
            observations, action_masks, policy_targets, value_targets
        )

        # Average metrics
        if num_updates > 0:
            for key in metrics_accum:
                metrics_accum[key] /= num_updates

        # Add additional metrics
        metrics_accum["learning_rate"] = current_lr
        metrics_accum["num_updates"] = float(num_updates)
        metrics_accum["buffer_size"] = float(num_samples)

        return metrics_accum

    def train_iteration(self) -> dict[str, float]:
        """Execute a single training iteration.

        Returns:
            Dictionary of metrics from this iteration.
        """
        iteration_start = time.time()
        metrics: dict[str, float] = {"iteration": float(self._iteration)}

        # Generate MCTS self-play games
        gen_start = time.time()
        new_data = self._generate_mcts_games()
        gen_time = time.time() - gen_start

        # Add to buffer
        self._add_to_buffer(new_data)

        # Train network on buffer
        train_start = time.time()
        train_metrics = self._train_on_data(self._training_buffer)
        train_time = time.time() - train_start

        # Combine metrics
        metrics.update(train_metrics)
        metrics["self_play/games_generated"] = float(self.config.games_per_iteration)
        metrics["self_play/samples_collected"] = float(len(new_data))
        metrics["self_play/generation_time"] = gen_time
        metrics["train/training_time"] = train_time
        metrics["total_games_played"] = float(self._total_games_played)
        metrics["total_steps_collected"] = float(self._total_steps_collected)

        iteration_time = time.time() - iteration_start
        metrics["iteration_time"] = iteration_time

        # Store in history
        self._metrics_history.append(metrics)

        return metrics

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(f"Starting MCTS training for {self.config.total_iterations} iterations")
        self._training_start_time = time.time()

        # Log hyperparameters
        self.tracker.log_hyperparameters(self.config.to_dict())

        try:
            while self._iteration < self.config.total_iterations:
                # Training iteration
                metrics = self.train_iteration()

                # Log metrics
                if self._iteration % self.config.log_frequency == 0:
                    self.tracker.log_metrics(metrics, step=self._iteration)

                    # Console logging
                    elapsed = time.time() - self._training_start_time
                    eta = elapsed / (self._iteration + 1) * (
                        self.config.total_iterations - self._iteration - 1
                    )
                    logger.info(
                        f"Iteration {self._iteration}/{self.config.total_iterations} | "
                        f"Loss: {metrics.get('total_loss', 0.0):.4f} | "
                        f"Policy: {metrics.get('policy_loss', 0.0):.4f} | "
                        f"Value: {metrics.get('value_loss', 0.0):.4f} | "
                        f"LR: {metrics.get('learning_rate', 0.0):.2e} | "
                        f"ETA: {eta / 60:.1f}min"
                    )

                # Checkpoint
                if (self._iteration + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint()

                # Add checkpoint to opponent pool for diversity training
                if (self._iteration + 1) % self.config.checkpoint_to_pool_frequency == 0:
                    self._add_to_opponent_pool()

                # Evaluation
                if (self._iteration + 1) % self.config.eval_frequency == 0:
                    self._run_evaluation()

                self._iteration += 1

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            # Final checkpoint
            self._save_checkpoint(is_final=True)
            self.tracker.finish()

            total_time = time.time() - self._training_start_time
            logger.info(f"Training completed in {total_time / 60:.1f} minutes")
            logger.info(f"Total iterations: {self._iteration}")
            logger.info(f"Total games: {self._total_games_played}")

    def _save_checkpoint(self, is_final: bool = False) -> Path:
        """Save training checkpoint.

        Args:
            is_final: Whether this is the final checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_name = "final" if is_final else f"iter_{self._iteration:06d}"
        checkpoint_path = self._checkpoint_dir / f"{checkpoint_name}.pt"

        save_checkpoint(self, str(checkpoint_path))

        # Log artifact
        self.tracker.log_artifact(str(checkpoint_path), checkpoint_name)

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def _add_to_opponent_pool(self) -> None:
        """Add current network state to opponent pool.

        This enables training against past versions of the network,
        preventing self-play collapse and catastrophic forgetting.
        """
        # Deep copy network state for opponent pool
        state_dict = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
        self._opponent_pool.add_checkpoint(
            state_dict=state_dict,
            iteration=self._iteration,
        )
        logger.info(
            f"Added iteration {self._iteration} to opponent pool "
            f"(size: {len(self._opponent_pool.checkpoints)}/"
            f"{self.config.opponent_config.max_checkpoints})"
        )

    def _run_evaluation(self) -> dict[str, float]:
        """Run evaluation against baseline opponents.

        Delegates to the shared evaluation module for consistency.

        Returns:
            Dictionary of evaluation metrics.
        """
        from _03_training.evaluation import run_evaluation

        logger.info(f"Running evaluation at iteration {self._iteration}")

        # Set network to eval mode for evaluation
        self.network.eval()

        metrics = run_evaluation(
            network=self.network,
            device=self._device,
            tracker=self.tracker,
            step=self._iteration,
            games_per_opponent=50,
            opponents=["random", "heuristic"],
            play_both_sides=True,
            mode="greedy",
        )

        # Log summary to console
        logger.info(
            "Evaluation results: "
            + ", ".join(
                f"{k.split('/')[-2]}={v:.1%}"
                for k, v in metrics.items()
                if k.endswith("/win_rate")
            )
        )

        return metrics


# ============================================================================
# Checkpoint Functions
# ============================================================================


def save_checkpoint(trainer: MCTSTrainer, path: str) -> None:
    """Save full training state for resumption.

    Args:
        trainer: MCTSTrainer instance to save.
        path: File path for checkpoint.
    """
    _ensure_torch()

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "iteration": trainer.current_iteration,
        "model_state_dict": trainer.network.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "config": trainer.config.to_dict(),
        "total_games_played": trainer._total_games_played,
        "total_steps_collected": trainer._total_steps_collected,
        "metrics_history": trainer._metrics_history,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        },
    }

    torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
    logger.info(f"Saved training checkpoint to {checkpoint_path}")

    # Save config as JSON
    config_path = checkpoint_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(trainer.config.to_dict(), f, indent=2)


def load_checkpoint(path: str, trainer: MCTSTrainer) -> None:
    """Resume training from checkpoint.

    Args:
        path: Path to checkpoint file.
        trainer: MCTSTrainer instance to restore into.
    """
    _ensure_torch()

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)

    # Restore model state
    trainer.network.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore training state
    trainer._iteration = checkpoint["iteration"]
    trainer._total_games_played = checkpoint.get("total_games_played", 0)
    trainer._total_steps_collected = checkpoint.get("total_steps_collected", 0)
    trainer._metrics_history = checkpoint.get("metrics_history", [])

    # Restore RNG states
    if "rng_state" in checkpoint:
        torch_rng_state = checkpoint["rng_state"]["torch"]
        if torch_rng_state.device != torch.device("cpu"):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint["rng_state"]["numpy"])

    logger.info(f"Restored training from iteration {trainer._iteration}")


def create_trainer_from_checkpoint(
    path: str,
    tracker: ExperimentTracker | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> MCTSTrainer:
    """Create a new MCTSTrainer instance from a checkpoint.

    Args:
        path: Path to checkpoint file.
        tracker: Optional experiment tracker.
        config_overrides: Optional dictionary of config parameters to override.

    Returns:
        MCTSTrainer instance restored from checkpoint.
    """
    _ensure_torch()

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint to extract config
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint.get("config", {})

    # Apply overrides
    if config_overrides:
        config_dict.update(config_overrides)

    config = MCTSTrainerConfig.from_dict(config_dict)

    # Create trainer with config
    trainer = MCTSTrainer(config, tracker=tracker)

    # Restore full state
    load_checkpoint(path, trainer)

    return trainer


__all__ = [
    "TORCH_AVAILABLE",
    "MCTSTrainer",
    "MCTSTrainerConfig",
    "create_trainer_from_checkpoint",
    "entropy_bonus",
    "get_learning_rate",
    "load_checkpoint",
    "policy_loss",
    "save_checkpoint",
    "set_learning_rate",
    "value_loss",
]
