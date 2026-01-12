"""PPO warm-start training for bootstrapping value networks before MCTS training.

This module provides PPO-based training to warm-start the value function before
transitioning to MCTS-based training. The cold-start problem in MCTS training
occurs when the value network provides no useful signal, leading to training
collapse. This warm-start phase trains the network to:

1. Beat random agents (baseline competence)
2. Compete with heuristic agents (quality signal)
3. Develop reasonable value estimates for game states

Once target win rates are achieved, the trained weights can be used to
initialize MCTS training for improved performance.

Example:
    # Train from scratch
    python -m _03_training.ppo_warmstart --config configs/ppo_warmstart.yaml

    # Resume from checkpoint
    python -m _03_training.ppo_warmstart --resume checkpoints/ppo_warmstart/latest.pt

    # Then use for MCTS training
    python -m _03_training.mcts_trainer --warmstart checkpoints/ppo_warmstart/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Conditional PyTorch import
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

if TYPE_CHECKING:
    from _02_agents.neural.network import BeastyBarNetwork

from _01_simulator.action_space import ACTION_DIM
from _01_simulator.observations import OBSERVATION_DIM
from _02_agents.neural.compile import maybe_compile_network
from _02_agents.neural.utils import NetworkConfig, get_device, seed_all
from _03_training.evaluation import (
    EvaluationConfig,
    create_evaluation_report,
    evaluate_agent,
    log_evaluation_results,
)
from _03_training.opponent_pool import (
    OpponentConfig,
    OpponentPool,
    OpponentType,
    create_opponent_network,
)
from _03_training.ppo import PPOBatch, PPOConfig, compute_gae, iterate_minibatches
from _03_training.replay_buffer import ReplayBuffer, Transition
from _03_training.self_play import (
    generate_games,
    trajectory_to_player_transitions,
)
from _03_training.tracking import ExperimentTracker, create_tracker
from _03_training.utils import inference_mode

logger = logging.getLogger(__name__)


def _ensure_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install with: pip install torch")


@dataclass
class PPOWarmstartConfig:
    """Configuration for PPO warm-start training.

    This config focuses on training a value network to beat baseline agents
    before using it for MCTS-enhanced training.

    Attributes:
        network_config: Neural network architecture configuration.
        ppo_config: PPO algorithm hyperparameters.
        games_per_iteration: Number of self-play games per training iteration.
        total_iterations: Maximum number of training iterations.
        target_win_rate_random: Stop early if we exceed this win rate vs random.
        target_win_rate_heuristic: Stop early if we exceed this win rate vs heuristic.
        eval_frequency: Evaluate against baselines every N iterations.
        eval_games: Number of games per evaluation opponent.
        checkpoint_frequency: Save checkpoint every N iterations.
        self_play_temperature: Temperature for action sampling during self-play.
        seed: Random seed for reproducibility.
        device: Training device ("cpu", "cuda", "mps", "auto").
        experiment_name: Name for experiment tracking and checkpoints.
        checkpoint_dir: Directory for saving checkpoints.
        lr_warmup_iterations: Number of iterations for learning rate warmup.
        lr_decay: Learning rate decay type ("linear", "cosine", "none").
        use_opponent_pool: Enable opponent diversity during training.
        opponent_config: Configuration for opponent pool sampling.
        buffer_size: Maximum replay buffer size.
        min_buffer_size: Minimum samples before training starts.
        shaped_rewards: Use score-margin shaped rewards.
        log_frequency: Log metrics every N iterations.
    """

    # Network configuration
    network_config: NetworkConfig = field(default_factory=NetworkConfig)

    # PPO configuration
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=4,
            minibatch_size=64,
            normalize_advantages=True,
            clip_value=True,
        )
    )

    # Training schedule
    games_per_iteration: int = 256
    total_iterations: int = 1000

    # Early stopping targets (win rate thresholds)
    target_win_rate_random: float = 0.60
    target_win_rate_heuristic: float = 0.20

    # Evaluation settings
    eval_frequency: int = 10
    eval_games: int = 100

    # Checkpointing
    checkpoint_frequency: int = 50

    # Self-play settings
    self_play_temperature: float = 1.0
    shaped_rewards: bool = False

    # Misc settings
    seed: int = 42
    device: str = "auto"
    experiment_name: str = "ppo_warmstart"
    checkpoint_dir: str = "checkpoints"

    # Learning rate schedule
    lr_warmup_iterations: int = 10
    lr_decay: str = "linear"

    # Opponent diversity
    use_opponent_pool: bool = True
    opponent_config: OpponentConfig = field(
        default_factory=lambda: OpponentConfig(
            current_weight=0.6,
            checkpoint_weight=0.2,
            random_weight=0.1,
            heuristic_weight=0.1,
            max_checkpoints=10,
        )
    )
    pool_checkpoint_frequency: int = 50

    # Buffer settings
    buffer_size: int = 100_000
    min_buffer_size: int = 1000

    # Logging
    log_frequency: int = 1

    # Torch compile settings (PyTorch 2.0+)
    # Enables torch.compile() for 20-40% inference speedup
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        result = asdict(self)
        result["network_config"] = self.network_config.to_dict()
        result["ppo_config"] = asdict(self.ppo_config)
        result["opponent_config"] = self.opponent_config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PPOWarmstartConfig:
        """Create configuration from dictionary."""
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
    def from_yaml(cls, path: str | Path) -> PPOWarmstartConfig:
        """Load configuration from YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.games_per_iteration <= 0:
            raise ValueError(f"games_per_iteration must be positive, got {self.games_per_iteration}")
        if self.total_iterations <= 0:
            raise ValueError(f"total_iterations must be positive, got {self.total_iterations}")
        if not (0 <= self.target_win_rate_random <= 1):
            raise ValueError(f"target_win_rate_random must be in [0,1], got {self.target_win_rate_random}")
        if not (0 <= self.target_win_rate_heuristic <= 1):
            raise ValueError(f"target_win_rate_heuristic must be in [0,1], got {self.target_win_rate_heuristic}")
        if self.self_play_temperature <= 0:
            raise ValueError(f"self_play_temperature must be positive, got {self.self_play_temperature}")
        if self.lr_decay not in ("linear", "cosine", "none"):
            raise ValueError(f"lr_decay must be 'linear', 'cosine', or 'none', got {self.lr_decay}")


def get_learning_rate(
    iteration: int,
    total_iterations: int,
    base_lr: float,
    warmup_iterations: int,
    decay_type: str,
    min_lr: float = 1e-6,
) -> float:
    """Compute learning rate for a given iteration.

    Supports warmup and multiple decay strategies.
    """
    if iteration < warmup_iterations:
        warmup_progress = (iteration + 1) / warmup_iterations
        return min_lr + (base_lr - min_lr) * warmup_progress

    if decay_type == "none":
        return base_lr

    decay_iterations = total_iterations - warmup_iterations
    decay_progress = (iteration - warmup_iterations) / max(decay_iterations, 1)
    decay_progress = min(decay_progress, 1.0)

    if decay_type == "linear":
        return min_lr + (base_lr - min_lr) * (1.0 - decay_progress)
    elif decay_type == "cosine":
        import math

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr + (base_lr - min_lr) * cosine_factor
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all parameter groups in optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class PPOWarmstartTrainer:
    """PPO trainer for warm-starting value networks before MCTS training.

    This trainer focuses on developing basic game competence through self-play
    with opponent diversity. It monitors win rates against baseline agents
    and supports early stopping when targets are achieved.

    The trained model can then be used to initialize MCTS training, avoiding
    the cold-start problem where MCTS receives no useful signal from the
    value network.

    Attributes:
        config: Training configuration.
        network: Neural network model.
        optimizer: PyTorch optimizer.
        tracker: Experiment tracking interface.
        device: Training device.
    """

    def __init__(
        self,
        config: PPOWarmstartConfig,
        tracker: ExperimentTracker | None = None,
        network: BeastyBarNetwork | None = None,
    ) -> None:
        """Initialize the warm-start trainer.

        Args:
            config: Training configuration.
            tracker: Optional experiment tracker. Creates console tracker if None.
            network: Optional pre-initialized network. Creates new network if None.
        """
        _ensure_torch()
        config.validate()

        self.config = config
        self._iteration = 0
        self._total_games_played = 0
        self._total_transitions_collected = 0
        self._training_start_time: float | None = None

        # Best model tracking (by win rate, not loss)
        self._best_win_rate_random: float = 0.0
        self._best_win_rate_heuristic: float = 0.0
        self._best_combined_score: float = 0.0

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

        # Create experiment tracker
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
        self._eval_history: list[dict[str, float]] = []

        # Opponent pool for diversity
        if config.use_opponent_pool:
            self.opponent_pool: OpponentPool | None = OpponentPool(
                config=config.opponent_config,
                seed=config.seed,
            )
            logger.info(
                f"Opponent pool enabled: "
                f"current={config.opponent_config.current_weight:.0%}, "
                f"checkpoint={config.opponent_config.checkpoint_weight:.0%}, "
                f"random={config.opponent_config.random_weight:.0%}, "
                f"heuristic={config.opponent_config.heuristic_weight:.0%}"
            )
        else:
            self.opponent_pool = None
            logger.info("Opponent pool disabled - using pure self-play")

        logger.info(f"PPOWarmstartTrainer initialized with {self.network.count_parameters():,} parameters")

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

    def _generate_self_play_games(self) -> tuple[list[Transition], list[list[Transition]], str]:
        """Generate self-play games and return transitions.

        Returns:
            Tuple of (all_transitions, trajectory_list, opponent_name).
        """
        from _02_agents.neural.network import BeastyBarNetwork

        # Sample opponent from pool if enabled
        opponent_agent = None
        opponent_network = None
        opponent_name = "self"
        collect_both_players = True

        if self.opponent_pool is not None:
            sampled = self.opponent_pool.sample_opponent()
            opponent_name = sampled.name

            if sampled.opponent_type == OpponentType.CURRENT:
                pass  # Standard self-play
            elif sampled.opponent_type == OpponentType.CHECKPOINT:
                opponent_network = create_opponent_network(
                    sampled=sampled,
                    network_class=BeastyBarNetwork,
                    network_config=self.config.network_config,
                    device=self._device,
                )
                collect_both_players = False
            elif sampled.opponent_type in (OpponentType.RANDOM, OpponentType.HEURISTIC):
                opponent_agent = sampled.agent
                collect_both_players = False

        # Generate games - use inference_mode to ensure network is in eval mode
        with inference_mode(self.network):
            trajectories = generate_games(
                network=self.network,
                num_games=self.config.games_per_iteration,
                temperature=self.config.self_play_temperature,
                device=self._device,
                shaped_rewards=self.config.shaped_rewards,
                opponent=opponent_agent,
                opponent_network=opponent_network,
            )

        # Convert to transitions
        # Note: trajectory_list is kept as a list for variable-length trajectories
        trajectory_list: list[list[Transition]] = []

        for trajectory in trajectories:
            players_to_collect = [0, 1] if collect_both_players else [0]

            for player in players_to_collect:
                player_transitions = trajectory_to_player_transitions(trajectory, player)
                if player_transitions:
                    trajectory_list.append(player_transitions)

        # Flatten trajectory_list into all_transitions
        # More efficient than repeated list.extend() in loop
        all_transitions: list[Transition] = [t for traj in trajectory_list for t in traj]

        self._total_games_played += self.config.games_per_iteration
        return all_transitions, trajectory_list, opponent_name

    def _prepare_gae_from_trajectories(
        self, trajectory_list: list[list[Transition]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Prepare training arrays from trajectories with GAE computation.

        Pre-allocates arrays for efficiency and computes GAE per trajectory.

        Args:
            trajectory_list: List of trajectories to process.

        Returns:
            Tuple of (observations, actions, action_probs, values, action_masks, advantages, returns)
            or None if no valid trajectories.
        """
        # Pre-calculate total number of steps across all trajectories
        total_steps = sum(len(traj) for traj in trajectory_list if traj)

        if total_steps == 0:
            return None

        # Pre-allocate arrays with exact sizes
        observations = np.empty((total_steps, OBSERVATION_DIM), dtype=np.float32)
        actions = np.empty(total_steps, dtype=np.int64)
        action_probs = np.empty(total_steps, dtype=np.float32)
        values = np.empty(total_steps, dtype=np.float32)
        action_masks = np.empty((total_steps, ACTION_DIM), dtype=np.float32)
        advantages = np.empty(total_steps, dtype=np.float32)
        returns = np.empty(total_steps, dtype=np.float32)

        # Fill arrays in-place
        offset = 0
        for traj in trajectory_list:
            if not traj:
                continue

            n = len(traj)
            end_idx = offset + n

            # Extract trajectory data into temporary arrays
            obs_arr = np.array([t.observation for t in traj], dtype=np.float32)
            acts_arr = np.array([t.action for t in traj], dtype=np.int64)
            probs_arr = np.array([t.action_prob for t in traj], dtype=np.float32)
            vals_arr = np.array([t.value for t in traj], dtype=np.float32)
            masks_arr = np.array([t.action_mask for t in traj], dtype=np.float32)
            rewards_arr = np.array([t.reward for t in traj], dtype=np.float32)
            dones_arr = np.array([t.done for t in traj], dtype=np.float32)

            # Compute GAE for this trajectory
            values_with_bootstrap = np.append(vals_arr, 0.0)
            traj_advantages, traj_returns = compute_gae(
                rewards=rewards_arr,
                values=values_with_bootstrap,
                dones=dones_arr,
                gamma=self.config.ppo_config.gamma,
                gae_lambda=self.config.ppo_config.gae_lambda,
            )

            # Fill pre-allocated arrays in-place (no memory reallocation)
            observations[offset:end_idx] = obs_arr
            actions[offset:end_idx] = acts_arr
            action_probs[offset:end_idx] = probs_arr
            values[offset:end_idx] = vals_arr
            action_masks[offset:end_idx] = masks_arr
            advantages[offset:end_idx] = traj_advantages
            returns[offset:end_idx] = traj_returns

            offset = end_idx

        return observations, actions, action_probs, values, action_masks, advantages, returns

    def _prepare_gae_from_buffer(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training arrays from replay buffer with GAE computation.

        Fallback when no trajectory list is provided.

        Returns:
            Tuple of (observations, actions, action_probs, values, action_masks, advantages, returns).
        """
        batch = self.replay_buffer.sample_all()
        observations = batch.observations
        actions = batch.actions
        action_probs = batch.action_probs
        values = batch.values
        action_masks = batch.action_masks

        values_with_bootstrap = np.append(values, 0.0)
        advantages, returns = compute_gae(
            rewards=batch.rewards,
            values=values_with_bootstrap,
            dones=batch.dones.astype(np.float32),
            gamma=self.config.ppo_config.gamma,
            gae_lambda=self.config.ppo_config.gae_lambda,
        )

        return observations, actions, action_probs, values, action_masks, advantages, returns

    def _create_ppo_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        action_probs: np.ndarray,
        values: np.ndarray,
        action_masks: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> PPOBatch:
        """Convert numpy arrays to PPOBatch tensors on device.

        Args:
            observations: Observation array.
            actions: Action array.
            action_probs: Action probability array.
            values: Value array.
            action_masks: Action mask array.
            advantages: Advantage array.
            returns: Return array.

        Returns:
            PPOBatch with tensors on training device.
        """
        return PPOBatch(
            observations=torch.from_numpy(observations).float().to(self._device),
            actions=torch.from_numpy(actions).long().to(self._device),
            old_log_probs=torch.log(torch.from_numpy(action_probs).float().clamp(min=1e-8)).to(self._device),
            old_values=torch.from_numpy(values).float().to(self._device),
            action_masks=torch.from_numpy(action_masks).float().to(self._device),
            advantages=torch.from_numpy(advantages).float().to(self._device),
            returns=torch.from_numpy(returns).float().to(self._device),
        )

    def _compute_ppo_losses(
        self,
        minibatch: PPOBatch,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """Compute PPO losses for a minibatch.

        Args:
            minibatch: PPOBatch with training data.
            policy_logits: Network policy output.
            values: Network value output (squeezed).

        Returns:
            Tuple of (policy_loss, value_loss, entropy_bonus, total_loss, approx_kl, clip_fraction).
        """
        from _03_training.ppo import entropy_bonus, policy_loss, value_loss

        # Normalize advantages
        advantages_norm = minibatch.advantages
        if self.config.ppo_config.normalize_advantages:
            advantages_norm = (advantages_norm - advantages_norm.mean()) / (advantages_norm.std() + 1e-8)

        # Compute losses
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

        return p_loss, v_loss, ent_bonus, total_loss, approx_kl.item(), clip_frac.item()

    def _apply_gradient_step(self, total_loss: torch.Tensor) -> None:
        """Apply gradient update with optional clipping.

        Args:
            total_loss: Loss tensor to backpropagate.
        """
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.ppo_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.ppo_config.max_grad_norm,
            )

        self.optimizer.step()

    def _run_ppo_training_loop(self, ppo_batch: PPOBatch) -> tuple[dict[str, float], int]:
        """Run PPO training epochs over the batch.

        Args:
            ppo_batch: PPOBatch with all training data.

        Returns:
            Tuple of (accumulated_metrics, num_updates).
        """
        metrics_accum: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        num_updates = 0

        for _epoch in range(self.config.ppo_config.ppo_epochs):
            for minibatch in iterate_minibatches(ppo_batch, self.config.ppo_config.minibatch_size, shuffle=True):
                # Forward pass
                policy_logits, values = self.network(minibatch.observations)
                values = values.squeeze(-1)

                # Compute losses
                p_loss, v_loss, ent_bonus, total_loss, approx_kl, clip_frac = self._compute_ppo_losses(
                    minibatch, policy_logits, values
                )

                # Optimization step
                self._apply_gradient_step(total_loss)

                # Accumulate metrics
                metrics_accum["policy_loss"] += p_loss.item()
                metrics_accum["value_loss"] += v_loss.item()
                metrics_accum["entropy"] += ent_bonus.item()
                metrics_accum["total_loss"] += total_loss.item()
                metrics_accum["approx_kl"] += approx_kl
                metrics_accum["clip_fraction"] += clip_frac
                num_updates += 1

        return metrics_accum, num_updates

    def _train_on_buffer(self, trajectory_list: list[list[Transition]] | None = None) -> dict[str, float]:
        """Train network on experiences in replay buffer.

        Args:
            trajectory_list: Optional list of trajectories for proper GAE computation.

        Returns:
            Dictionary of training metrics.
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            logger.warning(
                f"Buffer size ({len(self.replay_buffer)}) below minimum "
                f"({self.config.min_buffer_size}), skipping training"
            )
            return {}

        # Prepare training data with GAE computation
        if trajectory_list is not None and len(trajectory_list) > 0:
            gae_result = self._prepare_gae_from_trajectories(trajectory_list)
            if gae_result is None:
                logger.warning("No valid trajectories found, skipping training")
                return {}
            observations, actions, action_probs, values, action_masks, advantages, returns = gae_result
        else:
            observations, actions, action_probs, values, action_masks, advantages, returns = (
                self._prepare_gae_from_buffer()
            )

        # Create PPOBatch tensors
        ppo_batch = self._create_ppo_batch(
            observations, actions, action_probs, values, action_masks, advantages, returns
        )

        # Update learning rate
        current_lr = self._get_current_lr()
        set_learning_rate(self.optimizer, current_lr)

        # Ensure network is in training mode
        self.network.train()

        # Run PPO training loop
        metrics_accum, num_updates = self._run_ppo_training_loop(ppo_batch)

        # Average metrics
        if num_updates > 0:
            for key in metrics_accum:
                metrics_accum[key] /= num_updates

        metrics_accum["learning_rate"] = current_lr
        metrics_accum["num_updates"] = float(num_updates)
        metrics_accum["buffer_size"] = float(len(self.replay_buffer))

        return metrics_accum

    def _evaluate(self) -> tuple[dict[str, float], bool]:
        """Evaluate current network against baseline agents.

        Returns:
            Tuple of (metrics_dict, should_stop_early).
        """
        from _02_agents.neural.agent import NeuralAgent

        logger.info(f"Evaluating at iteration {self._iteration}")

        # Evaluate with network in eval mode
        with inference_mode(self.network):
            # Create agent from current network
            agent = NeuralAgent(
                model=self.network,
                device=self._device,
                mode="greedy",
            )

            # Evaluate
            eval_config = EvaluationConfig(
                games_per_opponent=self.config.eval_games,
                opponents=["random", "heuristic"],
                play_both_sides=True,
            )

            results = evaluate_agent(agent, eval_config, device=self._device)

        # Extract win rates
        win_rate_random = 0.0
        win_rate_heuristic = 0.0

        metrics: dict[str, float] = {}
        for result in results:
            prefix = f"eval/{result.opponent_name}"
            metrics[f"{prefix}/win_rate"] = result.win_rate
            metrics[f"{prefix}/avg_margin"] = result.avg_point_margin
            metrics[f"{prefix}/games"] = float(result.games_played)

            if result.opponent_name == "random":
                win_rate_random = result.win_rate
            elif result.opponent_name == "heuristic":
                win_rate_heuristic = result.win_rate

        # Log results
        log_evaluation_results(self.tracker, results, step=self._iteration)

        # Print evaluation report
        report = create_evaluation_report(results, iteration=self._iteration)
        logger.info("\n" + report)

        # Track best model (combined score: weighted average of win rates)
        combined_score = 0.7 * win_rate_random + 0.3 * win_rate_heuristic

        if combined_score > self._best_combined_score:
            self._best_combined_score = combined_score
            self._best_win_rate_random = win_rate_random
            self._best_win_rate_heuristic = win_rate_heuristic
            self._save_checkpoint(is_best=True)
            logger.info(f"New best model! Random: {win_rate_random:.1%}, Heuristic: {win_rate_heuristic:.1%}")

        # Check early stopping conditions
        should_stop = (
            win_rate_random >= self.config.target_win_rate_random
            and win_rate_heuristic >= self.config.target_win_rate_heuristic
        )

        if should_stop:
            logger.info(
                f"Early stopping triggered! Target win rates achieved: "
                f"Random {win_rate_random:.1%} >= {self.config.target_win_rate_random:.1%}, "
                f"Heuristic {win_rate_heuristic:.1%} >= {self.config.target_win_rate_heuristic:.1%}"
            )

        # Store evaluation history
        eval_metrics = {
            "iteration": float(self._iteration),
            "win_rate_random": win_rate_random,
            "win_rate_heuristic": win_rate_heuristic,
            "combined_score": combined_score,
        }
        self._eval_history.append(eval_metrics)

        return metrics, should_stop

    def train_iteration(self) -> dict[str, float]:
        """Execute a single training iteration.

        Returns:
            Dictionary of metrics from this iteration.
        """
        iteration_start = time.time()
        metrics: dict[str, float] = {"iteration": float(self._iteration)}

        # Generate self-play games
        gen_start = time.time()
        transitions, trajectory_list, _opponent_name = self._generate_self_play_games()
        gen_time = time.time() - gen_start

        # Add to replay buffer
        self.replay_buffer.add_batch(transitions)
        self._total_transitions_collected += len(transitions)

        # Train on buffer
        train_start = time.time()
        train_metrics = self._train_on_buffer(trajectory_list)
        train_time = time.time() - train_start

        # Add checkpoint to opponent pool periodically
        if self.opponent_pool is not None and (self._iteration + 1) % self.config.pool_checkpoint_frequency == 0:
            self.opponent_pool.add_checkpoint(
                state_dict=self.network.state_dict(),
                iteration=self._iteration,
            )

        # Combine metrics
        metrics.update(train_metrics)
        metrics["self_play/games_generated"] = float(self.config.games_per_iteration)
        metrics["self_play/transitions_collected"] = float(len(transitions))
        metrics["self_play/generation_time"] = gen_time
        metrics["train/training_time"] = train_time
        metrics["total_games_played"] = float(self._total_games_played)
        metrics["total_transitions"] = float(self._total_transitions_collected)

        if self.opponent_pool is not None:
            pool_stats = self.opponent_pool.get_statistics()
            metrics["opponent_pool/size"] = float(pool_stats["checkpoint_pool_size"])
            metrics["opponent_pool/samples"] = float(pool_stats["total_samples"])

        iteration_time = time.time() - iteration_start
        metrics["iteration_time"] = iteration_time

        self._metrics_history.append(metrics)
        return metrics

    def train(self) -> None:
        """Run the full training loop.

        Executes training iterations until:
        1. Total iterations reached, OR
        2. Target win rates achieved (early stopping)
        """
        logger.info(f"Starting PPO warm-start training for up to {self.config.total_iterations} iterations")
        logger.info(
            f"Early stopping targets: "
            f"Random >= {self.config.target_win_rate_random:.0%}, "
            f"Heuristic >= {self.config.target_win_rate_heuristic:.0%}"
        )

        self._training_start_time = time.time()
        self.tracker.log_hyperparameters(self.config.to_dict())

        try:
            while self._iteration < self.config.total_iterations:
                # Training iteration
                metrics = self.train_iteration()

                # Log metrics
                if self._iteration % self.config.log_frequency == 0:
                    self.tracker.log_metrics(metrics, step=self._iteration)

                    elapsed = time.time() - self._training_start_time
                    eta = elapsed / (self._iteration + 1) * (self.config.total_iterations - self._iteration - 1)
                    logger.info(
                        f"Iteration {self._iteration}/{self.config.total_iterations} | "
                        f"Loss: {metrics.get('total_loss', 0.0):.4f} | "
                        f"LR: {metrics.get('learning_rate', 0.0):.2e} | "
                        f"ETA: {eta / 60:.1f}min"
                    )

                # Checkpoint
                if (self._iteration + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint()

                # Evaluation
                if (self._iteration + 1) % self.config.eval_frequency == 0:
                    _eval_metrics, should_stop = self._evaluate()
                    if should_stop:
                        logger.info("Early stopping - target win rates achieved!")
                        break

                self._iteration += 1

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            # Final checkpoint and evaluation
            self._save_checkpoint(is_final=True)
            self._evaluate()
            self.tracker.finish()

            total_time = time.time() - self._training_start_time
            logger.info(f"Training completed in {total_time / 60:.1f} minutes")
            logger.info(f"Total iterations: {self._iteration}")
            logger.info(f"Total games: {self._total_games_played}")
            logger.info(
                f"Best model: Random {self._best_win_rate_random:.1%}, Heuristic {self._best_win_rate_heuristic:.1%}"
            )

    def _save_checkpoint(
        self,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        """Save training checkpoint.

        Args:
            is_best: Whether this is the best model so far.
            is_final: Whether this is the final checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        if is_final:
            checkpoint_name = "final"
        elif is_best:
            checkpoint_name = "best"
        else:
            checkpoint_name = f"iter_{self._iteration:06d}"

        checkpoint_path = self._checkpoint_dir / f"{checkpoint_name}.pt"

        checkpoint: dict[str, Any] = {
            "iteration": self._iteration,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "total_games_played": self._total_games_played,
            "total_transitions_collected": self._total_transitions_collected,
            "metrics_history": self._metrics_history,
            "eval_history": self._eval_history,
            "best_win_rate_random": self._best_win_rate_random,
            "best_win_rate_heuristic": self._best_win_rate_heuristic,
            "best_combined_score": self._best_combined_score,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
        }

        # Save opponent pool state
        if self.opponent_pool is not None:
            checkpoint["opponent_pool"] = {
                "checkpoints": [
                    {
                        "state_dict": cp.state_dict,
                        "iteration": cp.iteration,
                        "win_rate": cp.win_rate,
                    }
                    for cp in self.opponent_pool.checkpoints
                ],
                "sample_counts": {t.name: c for t, c in self.opponent_pool._sample_counts.items()},
            }

        torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
        self.tracker.log_artifact(str(checkpoint_path), checkpoint_name)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Also save config as JSON
        config_path = checkpoint_path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        return checkpoint_path


def load_warmstart_checkpoint(
    path: str | Path,
    config_overrides: dict[str, Any] | None = None,
) -> PPOWarmstartTrainer:
    """Load trainer from checkpoint for resuming training.

    Args:
        path: Path to checkpoint file.
        config_overrides: Optional config parameters to override.

    Returns:
        PPOWarmstartTrainer restored from checkpoint.
    """
    _ensure_torch()

    from _03_training.opponent_pool import CheckpointEntry

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load config
    config_dict = checkpoint.get("config", {})
    if config_overrides:
        config_dict.update(config_overrides)
    config = PPOWarmstartConfig.from_dict(config_dict)

    # Create trainer
    trainer = PPOWarmstartTrainer(config)

    # Restore model state
    trainer.network.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore training state
    trainer._iteration = checkpoint["iteration"]
    trainer._total_games_played = checkpoint.get("total_games_played", 0)
    trainer._total_transitions_collected = checkpoint.get("total_transitions_collected", 0)
    trainer._metrics_history = checkpoint.get("metrics_history", [])
    trainer._eval_history = checkpoint.get("eval_history", [])
    trainer._best_win_rate_random = checkpoint.get("best_win_rate_random", 0.0)
    trainer._best_win_rate_heuristic = checkpoint.get("best_win_rate_heuristic", 0.0)
    trainer._best_combined_score = checkpoint.get("best_combined_score", 0.0)

    # Restore RNG states
    if "rng_state" in checkpoint:
        torch_rng_state = checkpoint["rng_state"]["torch"]
        if torch_rng_state.device != torch.device("cpu"):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint["rng_state"]["numpy"])

    # Restore opponent pool
    if trainer.opponent_pool is not None and "opponent_pool" in checkpoint:
        pool_state = checkpoint["opponent_pool"]
        trainer.opponent_pool.checkpoints.clear()
        for cp_data in pool_state.get("checkpoints", []):
            trainer.opponent_pool.checkpoints.append(
                CheckpointEntry(
                    state_dict=cp_data["state_dict"],
                    iteration=cp_data["iteration"],
                    win_rate=cp_data.get("win_rate"),
                )
            )
        for type_name, count in pool_state.get("sample_counts", {}).items():
            try:
                opp_type = OpponentType[type_name]
                trainer.opponent_pool._sample_counts[opp_type] = count
            except KeyError:
                pass

    logger.info(f"Restored training from iteration {trainer._iteration}")
    return trainer


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="PPO warm-start training for bootstrapping value networks")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_warmstart.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment name",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="console",
        help="Tracking backend (console, wandb, tensorboard)",
    )
    return parser


def _configure_logging() -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _create_trainer_from_resume(args: argparse.Namespace) -> PPOWarmstartTrainer:
    """Create trainer by resuming from checkpoint.

    Args:
        args: Parsed command line arguments.

    Returns:
        PPOWarmstartTrainer restored from checkpoint.
    """
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.experiment_name:
        overrides["experiment_name"] = args.experiment_name

    return load_warmstart_checkpoint(args.resume, config_overrides=overrides)


def _create_trainer_from_config(args: argparse.Namespace) -> PPOWarmstartTrainer:
    """Create trainer from config file.

    Args:
        args: Parsed command line arguments.

    Returns:
        New PPOWarmstartTrainer instance.
    """
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = PPOWarmstartConfig.from_yaml(config_path)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = PPOWarmstartConfig()

    # Apply overrides
    if args.device:
        config.device = args.device
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    # Create tracker
    tracker = create_tracker(
        backend=args.tracker,
        project="beastybar",
        run_name=config.experiment_name,
        config=config.to_dict(),
    )

    return PPOWarmstartTrainer(config, tracker=tracker)


def main() -> None:
    """Main entry point for PPO warm-start training."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    _configure_logging()

    trainer = _create_trainer_from_resume(args) if args.resume else _create_trainer_from_config(args)

    trainer.train()


if __name__ == "__main__":
    main()


__all__ = [
    "PPOWarmstartConfig",
    "PPOWarmstartTrainer",
    "load_warmstart_checkpoint",
]
