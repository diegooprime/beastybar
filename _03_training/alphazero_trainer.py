"""AlphaZero-style trainer for Beasty Bar neural network.

This module implements the pure AlphaZero training loop as specified in
ROADMAP_TO_SUPERHUMAN.md Section 4:

1. Generate training data using MCTS search (not PPO rollouts)
2. Policy targets = normalized MCTS visit counts
3. Value targets = game outcome (Monte Carlo returns)
4. Cross-entropy policy loss, MSE value loss
5. No entropy bonus needed (MCTS provides exploration)

Key differences from PPO training:
- No advantage estimation (GAE) - uses terminal game outcomes directly
- Policy target is MCTS visit distribution, not single action
- No clipped surrogate objective - uses standard cross-entropy
- No entropy bonus - MCTS exploration replaces entropy regularization
- Simpler and more stable than PPO for two-player games

Example:
    config = AlphaZeroConfig(
        total_iterations=2000,
        games_per_iteration=100,
        num_simulations=200,
    )
    trainer = AlphaZeroTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from _01_simulator.action_space import ACTION_DIM
from _01_simulator.observations import OBSERVATION_DIM
from _02_agents.neural.compile import maybe_compile_network
from _02_agents.neural.network_v2 import NetworkConfigV2
from _02_agents.neural.utils import NetworkConfig, get_device, seed_all
from _03_training.checkpoint_manager import CheckpointManager
from _03_training.tracking import ExperimentTracker, create_tracker

# Optional tablebase import
try:
    from _02_agents.tablebase import EndgameTablebase, GameTheoreticValue
    TABLEBASE_AVAILABLE = True
except ImportError:
    TABLEBASE_AVAILABLE = False
    EndgameTablebase = None
    GameTheoreticValue = None

if TYPE_CHECKING:
    from _02_agents.neural.network import BeastyBarNetwork

logger = logging.getLogger(__name__)


def _ensure_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install with: pip install torch")


# ============================================================================
# Training Example Data Structure
# ============================================================================


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """Single training example from MCTS self-play.

    Following AlphaZero, each example contains:
    - observation: Game state observation tensor
    - action_mask: Legal action mask
    - mcts_policy: Normalized MCTS visit counts (policy target)
    - value: Game outcome from this player's perspective (value target)

    Attributes:
        observation: State observation tensor of shape (OBSERVATION_DIM,).
        action_mask: Legal action mask of shape (ACTION_DIM,).
        mcts_policy: MCTS visit distribution of shape (ACTION_DIM,).
        value: Game outcome in [-1, 1] where 1=win, -1=loss, 0=draw.
    """

    observation: np.ndarray  # (OBSERVATION_DIM,)
    action_mask: np.ndarray  # (ACTION_DIM,)
    mcts_policy: np.ndarray  # (ACTION_DIM,)
    value: float


# ============================================================================
# AlphaZero Configuration
# ============================================================================


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero-style training.

    This configuration follows the AlphaZero approach:
    - MCTS for policy improvement and exploration
    - Game outcomes for value targets
    - Cross-entropy policy loss + MSE value loss
    - No entropy bonus (MCTS provides exploration)

    Attributes:
        network_config: Configuration for neural network architecture.
        num_simulations: Number of MCTS simulations per move.
        c_puct: Exploration constant for PUCT formula.
        dirichlet_alpha: Dirichlet noise concentration parameter.
        dirichlet_epsilon: Mixing weight for Dirichlet noise at root.
        temperature: Initial temperature for action sampling from visits.
        temperature_drop_move: Move after which temperature drops to ~0.
        final_temperature: Temperature after drop move.
        games_per_iteration: Number of self-play games per training iteration.
        total_iterations: Total number of training iterations.
        batch_size: Batch size for network training.
        epochs_per_iteration: Number of training epochs per iteration.
        learning_rate: Base learning rate for optimizer.
        weight_decay: L2 regularization weight decay.
        max_grad_norm: Maximum gradient norm for clipping (0=no clipping).
        value_loss_weight: Weight of value loss in total loss.
        checkpoint_frequency: Save checkpoint every N iterations.
        eval_frequency: Evaluate against baselines every N iterations.
        eval_games_per_opponent: Games per opponent during evaluation.
        lr_warmup_iterations: Number of iterations for LR warmup.
        lr_decay: Learning rate decay type ("linear", "cosine", "none").
        seed: Random seed for reproducibility.
        device: Device for training ("cpu", "cuda", "mps", "auto").
        experiment_name: Name for experiment tracking.
        checkpoint_dir: Directory for saving checkpoints.
        log_frequency: Log metrics every N iterations.
        buffer_size: Maximum number of samples in replay buffer.
        min_buffer_size: Minimum samples before training starts.
        parallel_games: Number of games to run in parallel for GPU efficiency.
        torch_compile: Enable torch.compile() for inference speedup.
        torch_compile_mode: Mode for torch.compile().
    """

    # Network configuration
    network_config: NetworkConfig | NetworkConfigV2 = field(default_factory=NetworkConfig)
    network_version: str = "v1"  # "v1" or "v2"

    # MCTS configuration
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_drop_move: int = 15
    final_temperature: float = 0.1

    # Self-play settings
    games_per_iteration: int = 100
    parallel_games: int = 16

    # Training schedule
    total_iterations: int = 2000
    batch_size: int = 512
    epochs_per_iteration: int = 4

    # Optimization settings (no entropy bonus - MCTS provides exploration)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.5
    value_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.1  # Weight for auxiliary losses (V2 only)

    # Checkpointing and evaluation
    checkpoint_frequency: int = 50
    eval_frequency: int = 25
    eval_games_per_opponent: int = 200
    eval_opponents: list[str] = field(
        default_factory=lambda: [
            "random",
            "heuristic",
            "outcome_heuristic",
        ]
    )

    # Learning rate schedule
    lr_warmup_iterations: int = 10
    lr_decay: str = "cosine"  # "linear", "cosine", "none"

    # Misc settings
    seed: int = 42
    device: str = "auto"
    experiment_name: str = "beastybar_alphazero"
    checkpoint_dir: str = "checkpoints"
    log_frequency: int = 1

    # Replay buffer settings
    buffer_size: int = 500_000
    min_buffer_size: int = 10_000

    # Torch compile settings
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"

    # Tablebase integration (for perfect endgame play and value targets)
    tablebase_path: str | None = None  # Path to tablebase file
    use_tablebase_values: bool = True  # Use tablebase values as training targets
    use_tablebase_play: bool = True    # Use tablebase moves in endgame during self-play

    def get_temperature(self, move_number: int) -> float:
        """Get temperature for a given move number.

        Args:
            move_number: Current move number (0-indexed).

        Returns:
            Temperature to use for action selection.
        """
        if self.temperature_drop_move <= 0:
            return self.temperature
        if move_number >= self.temperature_drop_move:
            return self.final_temperature
        return self.temperature

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        result = asdict(self)
        result["network_config"] = self.network_config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlphaZeroConfig:
        """Create configuration from dictionary."""
        data = data.copy()

        # Determine network version
        network_version = data.get("network_version", "v1")

        if "network_config" in data:
            if isinstance(data["network_config"], dict):
                if network_version == "v2":
                    data["network_config"] = NetworkConfigV2.from_dict(data["network_config"])
                else:
                    data["network_config"] = NetworkConfig.from_dict(data["network_config"])
        else:
            if network_version == "v2":
                data["network_config"] = NetworkConfigV2()
            else:
                data["network_config"] = NetworkConfig()

        # Filter to known fields
        import dataclasses

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AlphaZeroConfig:
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
        if self.num_simulations <= 0:
            raise ValueError(f"num_simulations must be positive, got {self.num_simulations}")
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
    min_lr: float = 1e-6,
) -> float:
    """Compute learning rate for a given iteration.

    Supports warmup and multiple decay strategies.

    Args:
        iteration: Current training iteration (0-indexed).
        total_iterations: Total number of training iterations.
        base_lr: Base learning rate after warmup.
        warmup_iterations: Number of warmup iterations.
        decay_type: Type of decay ("linear", "cosine", "none").
        min_lr: Minimum learning rate floor.

    Returns:
        Learning rate for the given iteration.
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
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr + (base_lr - min_lr) * cosine_factor
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all parameter groups in optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ============================================================================
# AlphaZero Loss Functions
# ============================================================================


def policy_loss_cross_entropy(
    predicted_logits: torch.Tensor,
    target_policy: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss between predicted and MCTS target policies.

    AlphaZero policy loss: L_p = -sum(pi_mcts * log(pi_network))

    Args:
        predicted_logits: Network policy logits, shape (batch, ACTION_DIM).
        target_policy: MCTS visit distribution, shape (batch, ACTION_DIM).
        action_mask: Legal action mask, shape (batch, ACTION_DIM).

    Returns:
        Scalar policy loss.
    """
    _ensure_torch()

    # Apply mask to logits - set illegal actions to -inf
    masked_logits = torch.where(
        action_mask > 0,
        predicted_logits,
        torch.tensor(float("-inf"), device=predicted_logits.device),
    )

    # Compute log softmax probabilities
    log_probs = F.log_softmax(masked_logits, dim=-1)

    # Handle -inf values from masked actions
    log_probs = torch.where(
        torch.isinf(log_probs),
        torch.zeros_like(log_probs),
        log_probs,
    )

    # Cross-entropy: -sum(target * log(pred))
    loss = -(target_policy * log_probs).sum(dim=-1).mean()

    return loss


def value_loss_mse(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss between predicted values and game outcomes.

    AlphaZero value loss: L_v = (z - v)^2 where z is game outcome.

    Args:
        predicted_values: Network value predictions, shape (batch,) or (batch, 1).
        target_values: Game outcome values in [-1, 1], shape (batch,).

    Returns:
        Scalar value loss.
    """
    _ensure_torch()
    # Ensure both tensors have same shape to avoid broadcasting warnings
    predicted_values = predicted_values.view(-1)
    target_values = target_values.view(-1)
    return F.mse_loss(predicted_values, target_values)


# ============================================================================
# Replay Buffer for Training Examples
# ============================================================================


class AlphaZeroBuffer:
    """Circular replay buffer for AlphaZero training examples.

    Stores training examples from MCTS self-play games and provides
    efficient batched sampling for training.

    Attributes:
        max_size: Maximum number of examples to store.
        observation_dim: Dimension of observation vectors.
        action_dim: Dimension of action mask and policy vectors.
    """

    def __init__(
        self,
        max_size: int = 500_000,
        observation_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        """Initialize replay buffer with pre-allocated arrays.

        Args:
            max_size: Maximum capacity of buffer.
            observation_dim: Dimension of observation vectors.
            action_dim: Dimension of action/policy vectors.
        """
        self.max_size = max_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Pre-allocate storage arrays
        self._observations = np.zeros((max_size, observation_dim), dtype=np.float32)
        self._action_masks = np.zeros((max_size, action_dim), dtype=np.float32)
        self._mcts_policies = np.zeros((max_size, action_dim), dtype=np.float32)
        self._values = np.zeros(max_size, dtype=np.float32)

        # Circular buffer state
        self._position = 0
        self._size = 0
        self._total_added = 0

    def __len__(self) -> int:
        """Return current number of stored examples."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Return True if buffer is at capacity."""
        return self._size == self.max_size

    def add(self, example: TrainingExample) -> None:
        """Add a single training example to the buffer.

        Args:
            example: Training example to store.
        """
        idx = self._position

        self._observations[idx] = example.observation
        self._action_masks[idx] = example.action_mask
        self._mcts_policies[idx] = example.mcts_policy
        self._values[idx] = example.value

        self._position = (self._position + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)
        self._total_added += 1

    def add_batch(self, examples: list[TrainingExample]) -> None:
        """Add multiple training examples to the buffer.

        Args:
            examples: List of training examples to store.
        """
        for example in examples:
            self.add(example)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of training examples.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            Tuple of (observations, action_masks, mcts_policies, values).
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        if batch_size > self._size:
            raise ValueError(f"batch_size ({batch_size}) exceeds buffer size ({self._size})")

        indices = np.random.choice(self._size, size=batch_size, replace=False)

        return (
            self._observations[indices].copy(),
            self._action_masks[indices].copy(),
            self._mcts_policies[indices].copy(),
            self._values[indices].copy(),
        )

    def sample_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return all examples in buffer.

        Returns:
            Tuple of (observations, action_masks, mcts_policies, values).
        """
        return (
            self._observations[: self._size].copy(),
            self._action_masks[: self._size].copy(),
            self._mcts_policies[: self._size].copy(),
            self._values[: self._size].copy(),
        )

    def clear(self) -> None:
        """Reset buffer to empty state."""
        self._position = 0
        self._size = 0


# ============================================================================
# AlphaZero Trainer
# ============================================================================


class AlphaZeroTrainer:
    """AlphaZero-style trainer for Beasty Bar neural network.

    Implements the AlphaZero training loop:
    1. Generate self-play games using MCTS with neural network guidance
    2. Store (state, mcts_policy, game_outcome) tuples
    3. Train network to match MCTS policies and predict game outcomes
    4. No entropy bonus - MCTS provides sufficient exploration

    Attributes:
        config: Training configuration.
        network: Neural network model.
        optimizer: PyTorch optimizer.
        replay_buffer: Buffer for training examples.
        tracker: Experiment tracker for logging.
        device: Training device.
    """

    def __init__(
        self,
        config: AlphaZeroConfig,
        tracker: ExperimentTracker | None = None,
        network: BeastyBarNetwork | None = None,
    ) -> None:
        """Initialize AlphaZero trainer.

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
        self._total_examples_collected = 0
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
            if config.network_version == "v2":
                from _02_agents.neural.network_v2 import BeastyBarNetworkV2

                self.network = BeastyBarNetworkV2(config.network_config).to(self._device)
            else:
                from _02_agents.neural.network import BeastyBarNetwork

                self.network = BeastyBarNetwork(config.network_config).to(self._device)

        # Apply torch.compile if enabled
        if config.torch_compile:
            self.network = maybe_compile_network(
                self.network,
                compile_mode=config.torch_compile_mode,
                dynamic=True,
            )

        # Create optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Create replay buffer
        self.replay_buffer = AlphaZeroBuffer(
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

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self._checkpoint_dir,
            tracker=self.tracker,
        )

        # Metrics history
        self._metrics_history: list[dict[str, float]] = []

        # BatchMCTS instance for efficient game generation (created on first use)
        self._batch_mcts = None

        # Load tablebase if configured
        self._tablebase = None
        if config.tablebase_path and TABLEBASE_AVAILABLE:
            try:
                self._tablebase = EndgameTablebase.load(config.tablebase_path)
                logger.info(f"Loaded tablebase with {len(self._tablebase.positions):,} positions")
            except Exception as e:
                logger.warning(f"Failed to load tablebase from {config.tablebase_path}: {e}")

        logger.info(f"AlphaZero Trainer initialized with {self.network.count_parameters():,} parameters")
        logger.info(
            f"MCTS: {config.num_simulations} simulations, "
            f"c_puct={config.c_puct}, temp={config.temperature}"
        )

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

    def _get_batch_mcts(self):
        """Get or create BatchMCTS instance."""
        if self._batch_mcts is None:
            from _02_agents.mcts.batch_mcts import BatchMCTS

            self._batch_mcts = BatchMCTS(
                network=self.network,
                num_simulations=self.config.num_simulations,
                c_puct=self.config.c_puct,
                dirichlet_alpha=self.config.dirichlet_alpha,
                dirichlet_epsilon=self.config.dirichlet_epsilon,
                batch_size=max(16, self.config.parallel_games * 2),
                device=self._device,
            )
        return self._batch_mcts

    def generate_training_data(self, num_games: int) -> list[TrainingExample]:
        """Generate training data using parallel MCTS self-play.

        Runs multiple games in parallel for efficient GPU utilization:
        1. Maintain pool of active games (up to parallel_games)
        2. Batch MCTS search across all active game states
        3. Apply actions and collect training examples
        4. Start new games as others complete

        If tablebase is loaded:
        - Use tablebase moves in endgame positions (perfect play)
        - Use tablebase values as training targets (ground truth)

        Args:
            num_games: Number of self-play games to generate.

        Returns:
            List of TrainingExample objects with MCTS policies and game outcomes.
        """
        from _01_simulator import simulate
        from _01_simulator.action_space import (
            ACTION_DIM,
            action_index,
            index_to_action,
            legal_action_mask_tensor,
        )
        from _01_simulator.observations import state_to_tensor
        from _03_training.game_utils import compute_rewards, compute_winner
        from _03_training.utils import inference_mode

        examples: list[TrainingExample] = []
        batch_mcts = self._get_batch_mcts()
        tablebase_hits = 0
        tablebase_value_hits = 0

        # Track parallel games
        parallel_games = min(self.config.parallel_games, num_games)
        games_started = 0
        games_completed = 0

        # Active game state tracking
        # Each entry: (state, game_examples, move_count, game_id)
        # game_examples: list of (obs, mask, mcts_policy, player, tablebase_value)
        active_games: list[tuple] = []

        def start_new_game() -> tuple:
            """Initialize a new game."""
            nonlocal games_started
            seed = np.random.randint(0, 2**31)
            state = simulate.new_game(seed)
            games_started += 1
            return (state, [], 0, games_started - 1)

        def finalize_game(game_state, game_examples: list, game_id: int) -> None:
            """Compute outcomes and create training examples for completed game."""
            nonlocal games_completed, tablebase_value_hits

            scores = simulate.score(game_state)
            final_scores = (scores[0], scores[1])
            winner = compute_winner(final_scores)
            value_p0, value_p1 = compute_rewards(winner, final_scores, shaped=False)

            for obs, mask, policy, player, tb_value in game_examples:
                if tb_value is not None:
                    value = tb_value
                    tablebase_value_hits += 1
                else:
                    value = value_p0 if player == 0 else value_p1

                examples.append(
                    TrainingExample(
                        observation=obs,
                        action_mask=mask,
                        mcts_policy=policy,
                        value=value,
                    )
                )

            games_completed += 1
            if games_completed % max(1, num_games // 10) == 0:
                logger.info(f"Generated {games_completed}/{num_games} games")

        # Initialize first batch of games
        while len(active_games) < parallel_games and games_started < num_games:
            active_games.append(start_new_game())

        # Main loop: process all games in parallel batches
        with inference_mode(self.network):
            while active_games:
                # Separate games into: tablebase positions, MCTS positions, terminal
                mcts_games = []  # Games needing MCTS
                mcts_indices = []  # Original indices in active_games

                for idx, (state, game_examples, move_count, game_id) in enumerate(active_games):
                    if simulate.is_terminal(state) or move_count >= 200:
                        continue  # Will be finalized below

                    player = state.active_player

                    # Check tablebase first
                    use_tablebase = (
                        self._tablebase is not None
                        and self.config.use_tablebase_play
                        and self._tablebase.is_endgame_position(state)
                    )

                    if use_tablebase:
                        tablebase_hits += 1
                        legal = list(simulate.legal_actions(state, player))
                        optimal_action = self._tablebase.get_optimal_action(state, legal)

                        if optimal_action is not None:
                            # Create one-hot policy
                            action_idx = action_index(optimal_action)
                            mcts_policy = np.zeros(ACTION_DIM, dtype=np.float32)
                            mcts_policy[action_idx] = 1.0

                            # Get tablebase value
                            tablebase_value = None
                            if self.config.use_tablebase_values:
                                entry = self._tablebase.solve(state, player)
                                if entry.value != GameTheoreticValue.UNKNOWN:
                                    tablebase_value = float(entry.value.value)

                            # Store example and apply action
                            obs = state_to_tensor(state, player)
                            mask = legal_action_mask_tensor(state, player)
                            game_examples.append((obs, mask, mcts_policy, player, tablebase_value))

                            new_state = simulate.apply(state, optimal_action)
                            active_games[idx] = (new_state, game_examples, move_count + 1, game_id)
                            continue

                    # Needs MCTS search
                    mcts_games.append((state, player, game_examples, move_count, game_id, idx))
                    mcts_indices.append(idx)

                # Batch MCTS search for all games that need it
                if mcts_games:
                    states_batch = [g[0] for g in mcts_games]
                    # Use first game's player perspective (both players evaluated separately)
                    # Actually we need per-state perspective
                    players_batch = [g[1] for g in mcts_games]

                    # Group by player for batched search (same perspective)
                    p0_indices = [i for i, p in enumerate(players_batch) if p == 0]
                    p1_indices = [i for i, p in enumerate(players_batch) if p == 1]

                    visit_distributions = [None] * len(mcts_games)

                    # Search player 0 batch
                    if p0_indices:
                        p0_states = [states_batch[i] for i in p0_indices]
                        p0_results = batch_mcts.search_batch(
                            states=p0_states,
                            perspective=0,
                            add_root_noise=True,
                        )
                        for i, result in zip(p0_indices, p0_results, strict=True):
                            visit_distributions[i] = result

                    # Search player 1 batch
                    if p1_indices:
                        p1_states = [states_batch[i] for i in p1_indices]
                        p1_results = batch_mcts.search_batch(
                            states=p1_states,
                            perspective=1,
                            add_root_noise=True,
                        )
                        for i, result in zip(p1_indices, p1_results, strict=True):
                            visit_distributions[i] = result

                    # Process MCTS results and update games
                    for i, (state, player, game_examples, move_count, game_id, orig_idx) in enumerate(mcts_games):
                        visit_distribution = visit_distributions[i]

                        if not visit_distribution:
                            # Fallback: uniform over legal actions
                            mask = legal_action_mask_tensor(state, player)
                            legal_actions = np.where(mask > 0)[0]
                            if len(legal_actions) == 0:
                                # Terminal-like state, mark for finalization
                                continue
                            visit_distribution = {
                                int(a): 1.0 / len(legal_actions) for a in legal_actions
                            }

                        # Convert to dense policy
                        mcts_policy = np.zeros(ACTION_DIM, dtype=np.float32)
                        for action_idx, prob in visit_distribution.items():
                            if 0 <= action_idx < ACTION_DIM:
                                mcts_policy[action_idx] = prob

                        # Check tablebase value even if not using tablebase for moves
                        tablebase_value = None
                        if (
                            self._tablebase is not None
                            and self.config.use_tablebase_values
                            and self._tablebase.is_endgame_position(state)
                        ):
                            entry = self._tablebase.solve(state, player)
                            if entry.value != GameTheoreticValue.UNKNOWN:
                                tablebase_value = float(entry.value.value)

                        # Store example
                        obs = state_to_tensor(state, player)
                        mask = legal_action_mask_tensor(state, player)
                        game_examples.append((obs, mask, mcts_policy, player, tablebase_value))

                        # Sample action with temperature
                        temperature = self.config.get_temperature(move_count)
                        action_idx = self._sample_from_policy(visit_distribution, temperature)
                        action = index_to_action(action_idx)
                        new_state = simulate.apply(state, action)

                        active_games[orig_idx] = (new_state, game_examples, move_count + 1, game_id)

                # Finalize completed games and start new ones
                new_active_games = []
                for state, game_examples, move_count, game_id in active_games:
                    if simulate.is_terminal(state) or move_count >= 200:
                        finalize_game(state, game_examples, game_id)
                        # Start new game if needed
                        if games_started < num_games:
                            new_active_games.append(start_new_game())
                    else:
                        new_active_games.append((state, game_examples, move_count, game_id))

                active_games = new_active_games

        self._total_games_played += num_games
        self._total_examples_collected += len(examples)

        if tablebase_hits > 0:
            logger.info(
                f"Tablebase: {tablebase_hits} moves played, "
                f"{tablebase_value_hits} value targets from tablebase"
            )

        return examples

    def _sample_from_policy(
        self, visit_distribution: dict[int, float], temperature: float
    ) -> int:
        """Sample action from MCTS visit distribution with temperature.

        Uses AlphaZero-style temperature: pi(a) ~ N(a)^(1/tau)

        Args:
            visit_distribution: Action index to visit probability mapping.
            temperature: Sampling temperature (lower = more greedy).

        Returns:
            Selected action index.
        """
        actions = list(visit_distribution.keys())
        probs = np.array([visit_distribution[a] for a in actions])

        if temperature <= 0.01:
            # Greedy selection
            return actions[int(np.argmax(probs))]

        # Apply temperature
        if temperature != 1.0:
            scaled = probs ** (1.0 / temperature)
            probs = scaled / (scaled.sum() + 1e-10)

        # Handle numerical issues
        probs = np.clip(probs, 0, 1)
        probs = probs / (probs.sum() + 1e-10)

        return actions[np.random.choice(len(actions), p=probs)]

    def train_step(self, batch_size: int | None = None) -> dict[str, float]:
        """Train on a batch from the replay buffer.

        Performs one gradient update step with:
        - Cross-entropy policy loss against MCTS targets
        - MSE value loss against game outcomes
        - No entropy bonus (MCTS provides exploration)

        Args:
            batch_size: Number of examples to sample. Uses config default if None.

        Returns:
            Dictionary of training metrics.
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        if len(self.replay_buffer) < self.config.min_buffer_size:
            logger.warning(
                f"Buffer size ({len(self.replay_buffer)}) below minimum "
                f"({self.config.min_buffer_size}), skipping training"
            )
            return {}

        # Sample batch
        observations, action_masks, mcts_policies, values = self.replay_buffer.sample(
            min(batch_size, len(self.replay_buffer))
        )

        # Convert to tensors
        obs_tensor = torch.from_numpy(observations).to(self._device)
        mask_tensor = torch.from_numpy(action_masks).to(self._device)
        policy_tensor = torch.from_numpy(mcts_policies).to(self._device)
        value_tensor = torch.from_numpy(values).to(self._device)

        # Set network to training mode
        self.network.train()

        # Forward pass
        policy_logits, pred_values = self.network(obs_tensor, mask_tensor)
        pred_values = pred_values.squeeze(-1)

        # Compute losses (no entropy bonus - MCTS provides exploration)
        p_loss = policy_loss_cross_entropy(policy_logits, policy_tensor, mask_tensor)
        v_loss = value_loss_mse(pred_values, value_tensor)

        # Combined loss
        total_loss = p_loss + self.config.value_loss_weight * v_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

        return {
            "policy_loss": p_loss.item(),
            "value_loss": v_loss.item(),
            "total_loss": total_loss.item(),
        }

    def train_on_buffer(self) -> dict[str, float]:
        """Train for multiple epochs on the entire buffer.

        Returns:
            Dictionary of averaged training metrics.
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return {}

        # Update learning rate
        current_lr = self._get_current_lr()
        set_learning_rate(self.optimizer, current_lr)

        # Set network to training mode
        self.network.train()

        # Get all data from buffer
        observations, action_masks, mcts_policies, values = self.replay_buffer.sample_all()
        num_samples = len(observations)

        # Convert to tensors
        obs_tensor = torch.from_numpy(observations).to(self._device)
        mask_tensor = torch.from_numpy(action_masks).to(self._device)
        policy_tensor = torch.from_numpy(mcts_policies).to(self._device)
        value_tensor = torch.from_numpy(values).to(self._device)

        # Train for multiple epochs
        metrics_accum = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
        num_updates = 0

        for _epoch in range(self.config.epochs_per_iteration):
            # Shuffle indices
            indices = torch.randperm(num_samples, device=self._device)

            # Iterate through minibatches
            for start_idx in range(0, num_samples, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # Extract minibatch
                obs_batch = obs_tensor[batch_indices]
                mask_batch = mask_tensor[batch_indices]
                policy_batch = policy_tensor[batch_indices]
                value_batch = value_tensor[batch_indices]

                # Forward pass
                policy_logits, pred_values = self.network(obs_batch, mask_batch)
                pred_values = pred_values.squeeze(-1)

                # Compute losses (no entropy bonus)
                p_loss = policy_loss_cross_entropy(policy_logits, policy_batch, mask_batch)
                v_loss = value_loss_mse(pred_values, value_batch)
                total_loss = p_loss + self.config.value_loss_weight * v_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()

                # Accumulate metrics
                metrics_accum["policy_loss"] += p_loss.item()
                metrics_accum["value_loss"] += v_loss.item()
                metrics_accum["total_loss"] += total_loss.item()
                num_updates += 1

        # Average metrics
        if num_updates > 0:
            for key in metrics_accum:
                metrics_accum[key] /= num_updates

        metrics_accum["learning_rate"] = current_lr
        metrics_accum["num_updates"] = float(num_updates)
        metrics_accum["buffer_size"] = float(len(self.replay_buffer))

        return metrics_accum

    def train_iteration(self) -> dict[str, float]:
        """Execute a single training iteration.

        One iteration consists of:
        1. Generate self-play games using MCTS
        2. Add examples to replay buffer
        3. Train network on buffer
        4. Log metrics

        Returns:
            Dictionary of metrics from this iteration.
        """
        iteration_start = time.time()
        metrics: dict[str, float] = {"iteration": float(self._iteration)}

        # Generate MCTS self-play games
        gen_start = time.time()
        examples = self.generate_training_data(self.config.games_per_iteration)
        gen_time = time.time() - gen_start

        # Add to replay buffer
        self.replay_buffer.add_batch(examples)

        # Train on buffer
        train_start = time.time()
        train_metrics = self.train_on_buffer()
        train_time = time.time() - train_start

        # Combine metrics
        metrics.update(train_metrics)
        metrics["self_play/games_generated"] = float(self.config.games_per_iteration)
        metrics["self_play/examples_collected"] = float(len(examples))
        metrics["self_play/generation_time"] = gen_time
        metrics["train/training_time"] = train_time
        metrics["total_games_played"] = float(self._total_games_played)
        metrics["total_examples_collected"] = float(self._total_examples_collected)

        iteration_time = time.time() - iteration_start
        metrics["iteration_time"] = iteration_time

        # Store in history
        self._metrics_history.append(metrics)

        return metrics

    def train(self) -> None:
        """Run the full training loop with async game generation.

        Uses ThreadPoolExecutor to overlap game generation (CPU) with
        training (GPU) for better hardware utilization.
        """
        logger.info(f"Starting AlphaZero training for {self.config.total_iterations} iterations")
        logger.info("Using async game generation for CPU/GPU overlap")
        self._training_start_time = time.time()

        # Log hyperparameters
        self.tracker.log_hyperparameters(self.config.to_dict())

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Start first generation
                future: Future | None = executor.submit(
                    self.generate_training_data, self.config.games_per_iteration
                )

                while self._iteration < self.config.total_iterations:
                    iteration_start = time.time()

                    # Wait for current generation to complete
                    gen_start = time.time()
                    examples = future.result() if future else []
                    gen_time = time.time() - gen_start

                    # Start next generation in background (while we train)
                    if self._iteration + 1 < self.config.total_iterations:
                        future = executor.submit(
                            self.generate_training_data, self.config.games_per_iteration
                        )
                    else:
                        future = None

                    # Add examples to buffer
                    self.replay_buffer.add_batch(examples)

                    # Train on buffer (GPU work - overlaps with CPU generation)
                    train_start = time.time()
                    train_metrics = self.train_on_buffer()
                    train_time = time.time() - train_start

                    # Build metrics
                    iteration_time = time.time() - iteration_start
                    metrics: dict[str, float] = {"iteration": float(self._iteration)}
                    metrics.update(train_metrics)
                    metrics["self_play/games_generated"] = float(self.config.games_per_iteration)
                    metrics["self_play/examples_collected"] = float(len(examples))
                    metrics["self_play/generation_time"] = gen_time
                    metrics["train/training_time"] = train_time
                    metrics["total_games_played"] = float(self._total_games_played)
                    metrics["total_examples_collected"] = float(self._total_examples_collected)
                    metrics["iteration_time"] = iteration_time

                    self._metrics_history.append(metrics)

                    # Log metrics
                    if self._iteration % self.config.log_frequency == 0:
                        self.tracker.log_metrics(metrics, step=self._iteration)

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

            # Inference checkpoint
            inference_path = self.checkpoint_manager.save_for_inference(
                path="model_inference.pt",
                network=self.network,
                config=self.config.network_config.to_dict(),
            )
            logger.info(f"Inference checkpoint saved: {inference_path}")

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
        checkpoint_name = "final.pt" if is_final else f"iter_{self._iteration:06d}.pt"
        checkpoint_path = self._checkpoint_dir / checkpoint_name

        checkpoint: dict[str, Any] = {
            "iteration": self._iteration,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "total_games_played": self._total_games_played,
            "total_examples_collected": self._total_examples_collected,
            "metrics_history": self._metrics_history,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
        }

        torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save config as JSON
        config_path = checkpoint_path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Log artifact
        self.tracker.log_artifact(str(checkpoint_path), checkpoint_name)

        return checkpoint_path

    def _run_evaluation(self) -> dict[str, float]:
        """Run evaluation against baseline opponents.

        Returns:
            Dictionary of evaluation metrics.
        """
        from _03_training.evaluation import run_evaluation
        from _03_training.utils import inference_mode

        logger.info(f"Running evaluation at iteration {self._iteration}")

        with inference_mode(self.network):
            metrics = run_evaluation(
                network=self.network,
                device=self._device,
                tracker=self.tracker,
                step=self._iteration,
                games_per_opponent=self.config.eval_games_per_opponent,
                opponents=self.config.eval_opponents,
                play_both_sides=True,
                mode="greedy",
            )

        # Log summary
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


def save_checkpoint(trainer: AlphaZeroTrainer, path: str) -> None:
    """Save full training state for resumption.

    Args:
        trainer: AlphaZeroTrainer instance to save.
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
        "total_examples_collected": trainer._total_examples_collected,
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


def load_checkpoint(path: str, trainer: AlphaZeroTrainer) -> None:
    """Resume training from checkpoint.

    Args:
        path: Path to checkpoint file.
        trainer: AlphaZeroTrainer instance to restore into.
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
    trainer._total_examples_collected = checkpoint.get("total_examples_collected", 0)
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
) -> AlphaZeroTrainer:
    """Create a new AlphaZeroTrainer instance from a checkpoint.

    Args:
        path: Path to checkpoint file.
        tracker: Optional experiment tracker.
        config_overrides: Optional dictionary of config parameters to override.

    Returns:
        AlphaZeroTrainer instance restored from checkpoint.
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

    config = AlphaZeroConfig.from_dict(config_dict)

    # Create trainer with config
    trainer = AlphaZeroTrainer(config, tracker=tracker)

    # Restore full state
    load_checkpoint(path, trainer)

    return trainer


__all__ = [
    "TORCH_AVAILABLE",
    "AlphaZeroBuffer",
    "AlphaZeroConfig",
    "AlphaZeroTrainer",
    "TrainingExample",
    "create_trainer_from_checkpoint",
    "get_learning_rate",
    "load_checkpoint",
    "policy_loss_cross_entropy",
    "save_checkpoint",
    "set_learning_rate",
    "value_loss_mse",
]
