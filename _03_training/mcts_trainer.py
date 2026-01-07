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
        mcts_simulations=800,
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

from _02_agents.neural.utils import NetworkConfig, get_device, seed_all
from _03_training.mcts_self_play import (
    MCTSTrajectory,
    generate_mcts_games,
    mcts_policy_to_array,
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
        games_per_iteration: Number of self-play games per training iteration.
        mcts_simulations: Number of MCTS simulations per move.
        temperature: Temperature for action sampling (1.0 = proportional, 0 = greedy).
        c_puct: PUCT exploration constant for MCTS search.
        total_iterations: Total number of training iterations.
        batch_size: Batch size for network training.
        epochs_per_iteration: Number of training epochs per iteration.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 weight decay regularization.
        value_loss_weight: Weight of value loss in total loss.
        policy_loss_weight: Weight of policy loss in total loss.
        entropy_bonus_weight: Weight of entropy bonus in total loss.
        checkpoint_frequency: Save checkpoint every N iterations.
        eval_frequency: Evaluate against baselines every N iterations.
        lr_warmup_iterations: Number of iterations for learning rate warmup.
        lr_decay: Learning rate decay type ("linear", "cosine", "none").
        seed: Random seed for reproducibility.
        device: Device for training ("cpu", "cuda", "mps", "auto").
        experiment_name: Name for experiment tracking.
        checkpoint_dir: Directory for saving checkpoints.
        log_frequency: Log metrics every N iterations.
    """

    # Network configuration
    network_config: NetworkConfig = field(default_factory=NetworkConfig)

    # MCTS self-play settings
    games_per_iteration: int = 100
    mcts_simulations: int = 800
    temperature: float = 1.0
    c_puct: float = 1.0

    # Training schedule
    total_iterations: int = 1000
    batch_size: int = 256
    epochs_per_iteration: int = 10

    # Optimization settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping (0 = no clipping)
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    entropy_bonus_weight: float = 0.01

    # Checkpointing and evaluation
    checkpoint_frequency: int = 50
    eval_frequency: int = 10

    # Learning rate schedule
    lr_warmup_iterations: int = 10
    lr_decay: str = "linear"  # "linear", "cosine", "none"

    # Misc settings
    seed: int = 42
    device: str = "auto"  # "cpu", "cuda", "mps", "auto"
    experiment_name: str = "beastybar_mcts"
    checkpoint_dir: str = "checkpoints"
    log_frequency: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        result = asdict(self)
        result["network_config"] = self.network_config.to_dict()
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

        # Filter to known fields
        import dataclasses

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.games_per_iteration <= 0:
            raise ValueError(f"games_per_iteration must be positive, got {self.games_per_iteration}")
        if self.mcts_simulations <= 0:
            raise ValueError(f"mcts_simulations must be positive, got {self.mcts_simulations}")
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

        # Create optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

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

        logger.info(f"MCTS Trainer initialized with {self.network.count_parameters():,} parameters")

    @property
    def current_iteration(self) -> int:
        """Return current training iteration."""
        return self._iteration

    @property
    def device(self) -> torch.device:
        """Return training device."""
        return self._device

    def _get_current_lr(self) -> float:
        """Get learning rate for current iteration."""
        return get_learning_rate(
            iteration=self._iteration,
            total_iterations=self.config.total_iterations,
            base_lr=self.config.learning_rate,
            warmup_iterations=self.config.lr_warmup_iterations,
            decay_type=self.config.lr_decay,
        )

    def _generate_mcts_games(self) -> list[MCTSTrajectory]:
        """Generate self-play games using MCTS.

        Returns:
            List of MCTSTrajectory objects.
        """
        # Set network to evaluation mode for inference
        self.network.eval()

        trajectories = generate_mcts_games(
            network=self.network,
            num_games=self.config.games_per_iteration,
            num_simulations=self.config.mcts_simulations,
            temperature=self.config.temperature,
            device=self._device,
            c_puct=self.config.c_puct,
        )

        self._total_games_played += self.config.games_per_iteration
        return trajectories

    def _trajectories_to_training_data(
        self,
        trajectories: list[MCTSTrajectory],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert trajectories to training tensors.

        Args:
            trajectories: List of game trajectories.

        Returns:
            Tuple of (observations, target_policies, target_values, action_masks).
        """
        observations_list = []
        policies_list = []
        values_list = []
        masks_list = []

        for traj in trajectories:
            # Process both players' steps
            for player in [0, 1]:
                steps = traj.get_steps(player)
                for step in steps:
                    observations_list.append(step.observation)
                    # Convert MCTS policy dict to array
                    policy_array = mcts_policy_to_array(
                        step.mcts_policy,
                        self.config.network_config.action_dim,
                    )
                    policies_list.append(policy_array)
                    values_list.append(step.value)
                    masks_list.append(step.action_mask)

        # Convert to tensors
        observations = torch.from_numpy(np.array(observations_list)).float().to(self._device)
        target_policies = torch.from_numpy(np.array(policies_list)).float().to(self._device)
        target_values = torch.from_numpy(np.array(values_list)).float().to(self._device)
        action_masks = torch.from_numpy(np.array(masks_list)).float().to(self._device)

        return observations, target_policies, target_values, action_masks

    def _train_on_data(
        self,
        observations: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> dict[str, float]:
        """Train network on collected data.

        Args:
            observations: Observation tensors, shape (N, obs_dim).
            target_policies: Target MCTS policies, shape (N, action_dim).
            target_values: Target game outcomes, shape (N,).
            action_masks: Action masks, shape (N, action_dim).

        Returns:
            Dictionary of training metrics.
        """
        num_samples = observations.shape[0]
        if num_samples == 0:
            return {}

        # Ensure network is in training mode (dropout, batch norm, etc.)
        self.network.train()

        # Update learning rate
        current_lr = self._get_current_lr()
        set_learning_rate(self.optimizer, current_lr)

        # Training metrics accumulator
        metrics_accum = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        num_updates = 0

        # Multiple epochs over the data
        for _epoch in range(self.config.epochs_per_iteration):
            # Shuffle data
            indices = torch.randperm(num_samples)

            # Iterate through minibatches
            for start_idx in range(0, num_samples, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # Extract minibatch
                obs_batch = observations[batch_indices]
                policy_batch = target_policies[batch_indices]
                value_batch = target_values[batch_indices]
                mask_batch = action_masks[batch_indices]

                # Forward pass
                policy_logits, values = self.network(obs_batch, mask_batch)
                values = values.squeeze(-1)  # (batch,)

                # Compute losses
                p_loss = policy_loss(
                    predicted_logits=policy_logits,
                    target_policy=policy_batch,
                    action_mask=mask_batch,
                )

                v_loss = value_loss(
                    predicted_values=values,
                    target_values=value_batch,
                )

                ent_bonus = entropy_bonus(
                    logits=policy_logits,
                    action_mask=mask_batch,
                )

                # Combined loss
                total_loss = (
                    self.config.policy_loss_weight * p_loss
                    + self.config.value_loss_weight * v_loss
                    - self.config.entropy_bonus_weight * ent_bonus
                )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping for stability
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()

                # Accumulate metrics
                metrics_accum["policy_loss"] += p_loss.item()
                metrics_accum["value_loss"] += v_loss.item()
                metrics_accum["entropy"] += ent_bonus.item()
                metrics_accum["total_loss"] += total_loss.item()
                num_updates += 1

        # Average metrics
        if num_updates > 0:
            for key in metrics_accum:
                metrics_accum[key] /= num_updates

        # Add additional metrics
        metrics_accum["learning_rate"] = current_lr
        metrics_accum["num_updates"] = float(num_updates)
        metrics_accum["num_samples"] = float(num_samples)

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
        trajectories = self._generate_mcts_games()
        gen_time = time.time() - gen_start

        # Convert to training data
        observations, target_policies, target_values, action_masks = (
            self._trajectories_to_training_data(trajectories)
        )
        num_steps = observations.shape[0]
        self._total_steps_collected += num_steps

        # Train network
        train_start = time.time()
        train_metrics = self._train_on_data(
            observations, target_policies, target_values, action_masks
        )
        train_time = time.time() - train_start

        # Combine metrics
        metrics.update(train_metrics)
        metrics["self_play/games_generated"] = float(self.config.games_per_iteration)
        metrics["self_play/steps_collected"] = float(num_steps)
        metrics["self_play/generation_time"] = gen_time
        metrics["train/training_time"] = train_time
        metrics["total_games_played"] = float(self._total_games_played)
        metrics["total_steps_collected"] = float(self._total_steps_collected)

        # Game statistics
        total_moves = sum(t.game_length for t in trajectories)
        avg_game_length = total_moves / len(trajectories) if trajectories else 0
        p0_wins = sum(1 for t in trajectories if t.winner == 0)
        p1_wins = sum(1 for t in trajectories if t.winner == 1)
        draws = sum(1 for t in trajectories if t.winner is None)

        metrics["self_play/avg_game_length"] = avg_game_length
        metrics["self_play/p0_win_rate"] = p0_wins / len(trajectories) if trajectories else 0
        metrics["self_play/p1_win_rate"] = p1_wins / len(trajectories) if trajectories else 0
        metrics["self_play/draw_rate"] = draws / len(trajectories) if trajectories else 0

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

    def _run_evaluation(self) -> dict[str, float]:
        """Run evaluation against baseline opponents.

        Returns:
            Dictionary of evaluation metrics.
        """
        from _02_agents.neural.agent import NeuralAgent
        from _03_training.evaluation import (
            EvaluationConfig,
            evaluate_agent,
            log_evaluation_results,
        )

        logger.info(f"Running evaluation at iteration {self._iteration}")

        # Create agent from current network
        agent = NeuralAgent(
            model=self.network,
            device=self._device,
            mode="greedy",  # Use greedy for evaluation
        )

        # Configure evaluation
        eval_config = EvaluationConfig(
            games_per_opponent=50,
            opponents=["random", "heuristic"],
            play_both_sides=True,
        )

        # Run evaluation
        results = evaluate_agent(agent, eval_config, device=self._device)

        # Log results
        log_evaluation_results(self.tracker, results, step=self._iteration)

        # Extract metrics
        metrics: dict[str, float] = {}
        for result in results:
            prefix = f"eval/{result.opponent_name}"
            metrics[f"{prefix}/win_rate"] = result.win_rate
            metrics[f"{prefix}/avg_margin"] = result.avg_point_margin
            metrics[f"{prefix}/games"] = float(result.games_played)

        logger.info(
            "Evaluation results: "
            + ", ".join(f"{r.opponent_name}={r.win_rate:.1%}" for r in results)
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
