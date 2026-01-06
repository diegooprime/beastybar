"""Training orchestration for Beasty Bar neural network self-play.

This module provides:
- TrainingConfig: Complete configuration dataclass for training
- Trainer: Main training orchestrator with self-play, PPO updates, and checkpointing
- Learning rate scheduling with warmup and decay
- Gradient accumulation for memory-efficient training
- Full checkpoint/resumption support

Example:
    config = TrainingConfig(
        total_iterations=1000,
        games_per_iteration=256,
        checkpoint_frequency=50,
    )
    trainer = Trainer(config)
    trainer.train()
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Conditional imports for PyTorch
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from _02_agents.neural.utils import NetworkConfig, get_device, seed_all
from _03_training.ppo import PPOBatch, PPOConfig, iterate_minibatches
from _03_training.replay_buffer import ReplayBuffer, Transition
from _03_training.tracking import ExperimentTracker, create_tracker

if TYPE_CHECKING:
    from _02_agents.neural.network import BeastyBarNetwork

logger = logging.getLogger(__name__)


def _ensure_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install with: pip install torch")


# ============================================================================
# Training Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Complete configuration for neural network training.

    This dataclass consolidates all hyperparameters for:
    - Network architecture (via NetworkConfig)
    - PPO algorithm (via PPOConfig)
    - Self-play game generation
    - Training schedule and checkpointing
    - Learning rate schedule
    - Device and reproducibility settings

    Attributes:
        network_config: Configuration for neural network architecture.
        ppo_config: Configuration for PPO algorithm.
        games_per_iteration: Number of self-play games to generate per iteration.
        self_play_temperature: Temperature for action sampling during self-play.
        total_iterations: Total number of training iterations.
        checkpoint_frequency: Save checkpoint every N iterations.
        eval_frequency: Evaluate against baselines every N iterations.
        lr_warmup_iterations: Number of iterations for learning rate warmup.
        lr_decay: Learning rate decay type ("linear", "cosine", "none").
        seed: Random seed for reproducibility.
        device: Device for training ("cpu", "cuda", "mps", "auto").
        experiment_name: Name for experiment tracking.
        checkpoint_dir: Directory for saving checkpoints.
        buffer_size: Maximum size of replay buffer.
        gradient_accumulation_steps: Number of minibatches to accumulate before update.
        log_frequency: Log metrics every N iterations.
        min_buffer_size: Minimum buffer samples before training starts.
    """

    # Network configuration
    network_config: NetworkConfig = field(default_factory=NetworkConfig)

    # PPO configuration
    ppo_config: PPOConfig = field(default_factory=PPOConfig)

    # Self-play settings
    games_per_iteration: int = 256
    self_play_temperature: float = 1.0

    # Training schedule
    total_iterations: int = 1000
    checkpoint_frequency: int = 50
    eval_frequency: int = 10

    # Learning rate schedule
    lr_warmup_iterations: int = 10
    lr_decay: str = "linear"  # "linear", "cosine", "none"

    # Misc settings
    seed: int = 42
    device: str = "auto"  # "cpu", "cuda", "mps", "auto"
    experiment_name: str = "beastybar_neural"
    checkpoint_dir: str = "checkpoints"

    # Buffer settings
    buffer_size: int = 100_000
    min_buffer_size: int = 1000

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Logging
    log_frequency: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of all configuration parameters.
        """
        result = asdict(self)
        # Convert nested configs properly
        result["network_config"] = self.network_config.to_dict()
        result["ppo_config"] = asdict(self.ppo_config)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create configuration from dictionary.

        Args:
            data: Dictionary containing configuration parameters.

        Returns:
            TrainingConfig instance.
        """
        data = data.copy()

        # Handle nested configs
        if "network_config" in data:
            if isinstance(data["network_config"], dict):
                data["network_config"] = NetworkConfig.from_dict(data["network_config"])
        else:
            data["network_config"] = NetworkConfig()

        if "ppo_config" in data:
            if isinstance(data["ppo_config"], dict):
                data["ppo_config"] = PPOConfig(**data["ppo_config"])
        else:
            data["ppo_config"] = PPOConfig()

        # Filter to known fields
        import dataclasses

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.games_per_iteration <= 0:
            raise ValueError(f"games_per_iteration must be positive, got {self.games_per_iteration}")
        if self.total_iterations <= 0:
            raise ValueError(f"total_iterations must be positive, got {self.total_iterations}")
        if self.checkpoint_frequency <= 0:
            raise ValueError(f"checkpoint_frequency must be positive, got {self.checkpoint_frequency}")
        if self.lr_decay not in ("linear", "cosine", "none"):
            raise ValueError(f"lr_decay must be 'linear', 'cosine', or 'none', got {self.lr_decay}")
        if self.self_play_temperature <= 0:
            raise ValueError(f"self_play_temperature must be positive, got {self.self_play_temperature}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )


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

    Supports warmup and multiple decay strategies:
    - Warmup: Linear ramp from 0 to base_lr over warmup_iterations
    - Linear decay: Linear decrease from base_lr to 0
    - Cosine decay: Cosine annealing from base_lr to 0
    - None: Constant learning rate

    Args:
        iteration: Current training iteration (0-indexed).
        total_iterations: Total number of training iterations.
        base_lr: Base learning rate after warmup.
        warmup_iterations: Number of warmup iterations.
        decay_type: Type of decay ("linear", "cosine", "none").

    Returns:
        Learning rate for the given iteration.

    Raises:
        ValueError: If decay_type is not recognized.

    Example:
        >>> lr = get_learning_rate(50, 1000, 3e-4, 10, "linear")
        >>> print(f"{lr:.6f}")
        0.000285
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
    decay_progress = min(decay_progress, 1.0)  # Clamp to [0, 1]

    if decay_type == "linear":
        return base_lr * (1.0 - decay_progress)
    elif decay_type == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all parameter groups in optimizer.

    Args:
        optimizer: PyTorch optimizer to update.
        lr: New learning rate value.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ============================================================================
# Trainer Class
# ============================================================================


class Trainer:
    """Main training orchestrator for neural network self-play.

    Handles the complete training loop:
    1. Generate self-play games
    2. Store experiences in replay buffer
    3. Train network with PPO on collected experiences
    4. Log metrics and checkpoints
    5. Periodic evaluation

    The trainer supports:
    - Full checkpoint/resumption
    - Gradient accumulation for memory efficiency
    - Learning rate scheduling
    - Multiple experiment tracking backends

    Attributes:
        config: Training configuration.
        network: Neural network model.
        optimizer: PyTorch optimizer.
        replay_buffer: Experience replay buffer.
        tracker: Experiment tracker for logging.
        device: Training device.

    Example:
        config = TrainingConfig(total_iterations=100)
        trainer = Trainer(config)
        trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        tracker: ExperimentTracker | None = None,
        network: BeastyBarNetwork | None = None,
    ) -> None:
        """Initialize trainer with configuration.

        Args:
            config: Training configuration.
            tracker: Optional experiment tracker. If None, creates console tracker.
            network: Optional pre-initialized network. If None, creates new network.

        Raises:
            ImportError: If PyTorch is not available.
            ValueError: If configuration is invalid.
        """
        _ensure_torch()
        config.validate()

        self.config = config
        self._iteration = 0
        self._total_games_played = 0
        self._total_transitions_collected = 0
        self._training_start_time: float | None = None

        # Set random seeds for reproducibility
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

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.ppo_config.learning_rate,
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=config.buffer_size,
            observation_dim=config.network_config.observation_dim,
            action_dim=config.network_config.action_dim,
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

        # Metrics history for current training session
        self._metrics_history: list[dict[str, float]] = []

        logger.info(f"Trainer initialized with {self.network.count_parameters():,} parameters")

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
            base_lr=self.config.ppo_config.learning_rate,
            warmup_iterations=self.config.lr_warmup_iterations,
            decay_type=self.config.lr_decay,
        )

    def _generate_self_play_games(self) -> tuple[list[Transition], list[list[Transition]]]:
        """Generate self-play games and return transitions with trajectory info.

        Uses the actual self-play module to generate real game trajectories.

        Returns:
            Tuple of (all_transitions, trajectory_list) where trajectory_list
            contains separate lists of transitions for each trajectory (for GAE).
        """
        from _03_training.self_play import (
            generate_games,
            trajectory_to_player_transitions,
        )

        # Generate actual self-play games
        trajectories = generate_games(
            network=self.network,
            num_games=self.config.games_per_iteration,
            temperature=self.config.self_play_temperature,
            device=self._device,
        )

        # Convert to transitions, keeping trajectory boundaries for GAE
        all_transitions: list[Transition] = []
        trajectory_list: list[list[Transition]] = []

        for trajectory in trajectories:
            # Each player's trajectory is a separate sequence for GAE
            for player in [0, 1]:
                player_transitions = trajectory_to_player_transitions(trajectory, player)
                if player_transitions:
                    trajectory_list.append(player_transitions)
                    all_transitions.extend(player_transitions)

        self._total_games_played += self.config.games_per_iteration
        return all_transitions, trajectory_list

    def _train_on_buffer(
        self, trajectory_list: list[list[Transition]] | None = None
    ) -> dict[str, float]:
        """Train network on experiences in replay buffer.

        Performs PPO updates with gradient accumulation support.
        If trajectory_list is provided, computes GAE properly per-trajectory.

        Args:
            trajectory_list: Optional list of trajectory lists for proper GAE computation.

        Returns:
            Dictionary of training metrics.
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            logger.warning(
                f"Buffer size ({len(self.replay_buffer)}) below minimum "
                f"({self.config.min_buffer_size}), skipping training"
            )
            return {}

        from _03_training.ppo import compute_gae

        # If we have trajectory info, compute GAE properly per-trajectory
        if trajectory_list is not None and len(trajectory_list) > 0:
            all_observations = []
            all_actions = []
            all_action_probs = []
            all_values = []
            all_action_masks = []
            all_advantages = []
            all_returns = []

            for traj in trajectory_list:
                if not traj:
                    continue

                # Extract arrays for this trajectory
                obs_arr = np.array([t.observation for t in traj], dtype=np.float32)
                acts_arr = np.array([t.action for t in traj], dtype=np.int64)
                probs_arr = np.array([t.action_prob for t in traj], dtype=np.float32)
                vals_arr = np.array([t.value for t in traj], dtype=np.float32)
                masks_arr = np.array([t.action_mask for t in traj], dtype=np.float32)
                rewards_arr = np.array([t.reward for t in traj], dtype=np.float32)
                dones_arr = np.array([t.done for t in traj], dtype=np.float32)

                # Compute GAE for this trajectory
                # Bootstrap value is 0 since trajectory ends at game termination
                values_with_bootstrap = np.append(vals_arr, 0.0)
                advantages, returns = compute_gae(
                    rewards=rewards_arr,
                    values=values_with_bootstrap,
                    dones=dones_arr,
                    gamma=self.config.ppo_config.gamma,
                    gae_lambda=self.config.ppo_config.gae_lambda,
                )

                all_observations.append(obs_arr)
                all_actions.append(acts_arr)
                all_action_probs.append(probs_arr)
                all_values.append(vals_arr)
                all_action_masks.append(masks_arr)
                all_advantages.append(advantages)
                all_returns.append(returns)

            # Concatenate all trajectories
            observations = np.concatenate(all_observations, axis=0)
            actions = np.concatenate(all_actions, axis=0)
            action_probs = np.concatenate(all_action_probs, axis=0)
            values = np.concatenate(all_values, axis=0)
            action_masks = np.concatenate(all_action_masks, axis=0)
            advantages = np.concatenate(all_advantages, axis=0)
            returns = np.concatenate(all_returns, axis=0)
        else:
            # Fallback: sample from buffer (less accurate GAE)
            batch = self.replay_buffer.sample_all()
            observations = batch.observations
            actions = batch.actions
            action_probs = batch.action_probs
            values = batch.values
            action_masks = batch.action_masks

            # Simplified GAE (treats as single trajectory - less accurate)
            values_with_bootstrap = np.append(values, 0.0)
            advantages, returns = compute_gae(
                rewards=batch.rewards,
                values=values_with_bootstrap,
                dones=batch.dones.astype(np.float32),
                gamma=self.config.ppo_config.gamma,
                gae_lambda=self.config.ppo_config.gae_lambda,
            )

        # Create PPOBatch
        # Note: action_probs are raw probabilities, convert to log probs
        ppo_batch = PPOBatch(
            observations=torch.from_numpy(observations).float().to(self._device),
            actions=torch.from_numpy(actions).long().to(self._device),
            old_log_probs=torch.log(
                torch.from_numpy(action_probs).float().clamp(min=1e-8)
            ).to(self._device),
            old_values=torch.from_numpy(values).float().to(self._device),
            action_masks=torch.from_numpy(action_masks).float().to(self._device),
            advantages=torch.from_numpy(advantages).float().to(self._device),
            returns=torch.from_numpy(returns).float().to(self._device),
        )

        # Training with gradient accumulation
        metrics_accum: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        num_updates = 0

        # Update learning rate
        current_lr = self._get_current_lr()
        set_learning_rate(self.optimizer, current_lr)

        # Multiple PPO epochs
        for _epoch in range(self.config.ppo_config.ppo_epochs):
            # Iterate through minibatches with accumulation
            accumulated_steps = 0
            self.optimizer.zero_grad()

            for minibatch in iterate_minibatches(
                ppo_batch, self.config.ppo_config.minibatch_size, shuffle=True
            ):
                # Forward pass
                policy_logits, values = self.network(minibatch.observations)
                values = values.squeeze(-1)

                # Normalize advantages
                advantages_norm = minibatch.advantages
                if self.config.ppo_config.normalize_advantages:
                    advantages_norm = (advantages_norm - advantages_norm.mean()) / (
                        advantages_norm.std() + 1e-8
                    )

                # Compute losses using functions from ppo.py
                from _03_training.ppo import entropy_bonus, policy_loss, value_loss

                p_loss, approx_kl, clip_frac = policy_loss(
                    logits=policy_logits,
                    actions=minibatch.actions,
                    old_log_probs=minibatch.old_log_probs,
                    advantages=advantages_norm,
                    action_masks=minibatch.action_masks,
                    clip_epsilon=self.config.ppo_config.clip_epsilon,
                )

                v_loss = value_loss(
                    predicted_values=values,
                    returns=minibatch.returns,
                    old_values=minibatch.old_values,
                    clip_epsilon=self.config.ppo_config.clip_epsilon,
                    clip_value=self.config.ppo_config.clip_value,
                )

                ent_bonus = entropy_bonus(
                    logits=policy_logits,
                    action_masks=minibatch.action_masks,
                )

                # Combined loss
                total_loss = (
                    p_loss
                    + self.config.ppo_config.value_coef * v_loss
                    - self.config.ppo_config.entropy_coef * ent_bonus
                )

                # Scale loss for gradient accumulation
                scaled_loss = total_loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()

                accumulated_steps += 1

                # Accumulate metrics
                metrics_accum["policy_loss"] += p_loss.item()
                metrics_accum["value_loss"] += v_loss.item()
                metrics_accum["entropy"] += ent_bonus.item()
                metrics_accum["total_loss"] += total_loss.item()
                metrics_accum["approx_kl"] += approx_kl.item()
                metrics_accum["clip_fraction"] += clip_frac.item()
                num_updates += 1

                # Apply gradients after accumulation steps
                if accumulated_steps >= self.config.gradient_accumulation_steps:
                    # Gradient clipping
                    if self.config.ppo_config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.network.parameters(),
                            self.config.ppo_config.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulated_steps = 0

            # Handle remaining accumulated gradients
            if accumulated_steps > 0:
                if self.config.ppo_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config.ppo_config.max_grad_norm,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Average metrics
        if num_updates > 0:
            for key in metrics_accum:
                metrics_accum[key] /= num_updates

        # Add additional metrics
        metrics_accum["learning_rate"] = current_lr
        metrics_accum["num_updates"] = float(num_updates)
        metrics_accum["buffer_size"] = float(len(self.replay_buffer))

        return metrics_accum

    def train_iteration(self) -> dict[str, float]:
        """Execute a single training iteration.

        One iteration consists of:
        1. Generate self-play games
        2. Add experiences to replay buffer
        3. Train PPO on buffer (with proper per-trajectory GAE)
        4. Log metrics

        Returns:
            Dictionary of metrics from this iteration.
        """
        iteration_start = time.time()
        metrics: dict[str, float] = {"iteration": float(self._iteration)}

        # Generate self-play games
        gen_start = time.time()
        transitions, trajectory_list = self._generate_self_play_games()
        gen_time = time.time() - gen_start

        # Add to replay buffer
        self.replay_buffer.add_batch(transitions)
        self._total_transitions_collected += len(transitions)

        # Train on buffer with proper trajectory info for GAE
        train_start = time.time()
        train_metrics = self._train_on_buffer(trajectory_list)
        train_time = time.time() - train_start

        # Combine metrics
        metrics.update(train_metrics)
        metrics["self_play/games_generated"] = float(self.config.games_per_iteration)
        metrics["self_play/transitions_collected"] = float(len(transitions))
        metrics["self_play/generation_time"] = gen_time
        metrics["train/training_time"] = train_time
        metrics["total_games_played"] = float(self._total_games_played)
        metrics["total_transitions"] = float(self._total_transitions_collected)

        iteration_time = time.time() - iteration_start
        metrics["iteration_time"] = iteration_time

        # Store in history
        self._metrics_history.append(metrics)

        return metrics

    def train(self) -> None:
        """Run the full training loop.

        Executes training iterations until total_iterations is reached.
        Handles checkpointing, logging, and evaluation.
        """
        logger.info(f"Starting training for {self.config.total_iterations} iterations")
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

                    # Log to console
                    elapsed = time.time() - self._training_start_time
                    eta = elapsed / (self._iteration + 1) * (
                        self.config.total_iterations - self._iteration - 1
                    )
                    logger.info(
                        f"Iteration {self._iteration}/{self.config.total_iterations} | "
                        f"Loss: {metrics.get('total_loss', 0.0):.4f} | "
                        f"LR: {metrics.get('learning_rate', 0.0):.2e} | "
                        f"ETA: {eta / 60:.1f}min"
                    )

                # Checkpoint
                if (self._iteration + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint()

                # Evaluation (placeholder - will integrate with evaluation.py)
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

        save_training_checkpoint(self, str(checkpoint_path))

        # Log artifact
        self.tracker.log_artifact(str(checkpoint_path), checkpoint_name)

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def _run_evaluation(self) -> dict[str, float]:
        """Run evaluation against baseline opponents.

        Evaluates the current network against random and heuristic agents.

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
            games_per_opponent=50,  # Quick evaluation
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


def save_training_checkpoint(trainer: Trainer, path: str) -> None:
    """Save full training state for resumption.

    Saves:
    - Model weights
    - Optimizer state
    - Training iteration
    - Configuration
    - Replay buffer state (optional, can be large)
    - Metrics history

    Args:
        trainer: Trainer instance to save.
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
        "total_transitions_collected": trainer._total_transitions_collected,
        "metrics_history": trainer._metrics_history,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        },
    }

    # Optionally save buffer state (can be large)
    # For full resumption, buffer contents are needed
    buffer_state = {
        "size": len(trainer.replay_buffer),
        "position": trainer.replay_buffer._position,
        "total_added": trainer.replay_buffer._total_added,
    }
    checkpoint["buffer_state"] = buffer_state

    torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
    logger.info(f"Saved training checkpoint to {checkpoint_path}")

    # Also save config as JSON for easy inspection
    config_path = checkpoint_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(trainer.config.to_dict(), f, indent=2)


def load_training_checkpoint(path: str, trainer: Trainer) -> None:
    """Resume training from checkpoint.

    Restores:
    - Model weights
    - Optimizer state
    - Training iteration
    - Metrics history
    - RNG states (for reproducibility)

    Args:
        path: Path to checkpoint file.
        trainer: Trainer instance to restore into.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
        RuntimeError: If checkpoint is incompatible.
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
    trainer._total_transitions_collected = checkpoint.get("total_transitions_collected", 0)
    trainer._metrics_history = checkpoint.get("metrics_history", [])

    # Restore RNG states for reproducibility
    if "rng_state" in checkpoint:
        # RNG state must be on CPU
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
) -> Trainer:
    """Create a new Trainer instance from a checkpoint.

    Useful for resuming training with potentially modified configuration.

    Args:
        path: Path to checkpoint file.
        tracker: Optional experiment tracker.
        config_overrides: Optional dictionary of config parameters to override.

    Returns:
        Trainer instance restored from checkpoint.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
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

    config = TrainingConfig.from_dict(config_dict)

    # Create trainer with config
    trainer = Trainer(config, tracker=tracker)

    # Restore full state
    load_training_checkpoint(path, trainer)

    return trainer


__all__ = [
    "TORCH_AVAILABLE",
    "Trainer",
    "TrainingConfig",
    "create_trainer_from_checkpoint",
    "get_learning_rate",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "set_learning_rate",
]
