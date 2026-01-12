"""MCTS-Enhanced PPO Training with AlphaZero-style policy targets.

This module integrates Monte Carlo Tree Search with PPO training following
the AlphaZero recipe:
- Uses MCTS visit counts as policy targets instead of raw network policy
- Trains on game outcomes (Monte Carlo returns) instead of GAE
- Leverages batch MCTS for efficient parallel game generation

Key differences from standard PPO:
1. Policy targets: MCTS visit distribution (improved policy) vs network policy
2. Training signal: Cross-entropy with MCTS targets vs policy gradient
3. Value targets: Game outcomes vs GAE estimates
4. Exploration: MCTS provides natural exploration vs entropy bonus

References:
    - AlphaZero: https://arxiv.org/abs/1712.01815
    - MuZero: https://arxiv.org/abs/1911.08265
    - ROADMAP_TO_SUPERHUMAN.md Section 5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# Conditional PyTorch import with graceful fallback
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

from _01_simulator import engine, state
from _01_simulator.action_space import (
    ACTION_DIM,
    index_to_action,
    legal_action_mask_tensor,
)
from _01_simulator.observations import state_to_tensor
from _02_agents.mcts.batch_mcts import BatchMCTS

if TYPE_CHECKING:
    from _02_agents.base import Agent
    from _02_agents.neural.network import BeastyBarNetwork

logger = logging.getLogger(__name__)


def _ensure_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MCTS-PPO training. Install with: pip install torch"
        )


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class MCTSPPOConfig:
    """Configuration for MCTS-enhanced PPO training.

    Attributes:
        mcts_simulations: Number of MCTS simulations per move during training.
        c_puct: PUCT exploration constant.
        dirichlet_alpha: Dirichlet noise concentration for root exploration.
        dirichlet_epsilon: Mixing weight for Dirichlet noise.
        mcts_batch_size: Number of leaves to batch for neural network evaluation.
        temperature: Temperature for MCTS action sampling (1.0 = proportional to visits).
        temperature_threshold: After this many moves, use argmax instead of sampling.
        learning_rate: Optimizer learning rate.
        value_coef: Weight for value loss in total loss.
        l2_reg: L2 regularization coefficient.
        train_batch_size: Batch size for training updates.
        epochs_per_iteration: Number of training epochs per data collection iteration.
        games_per_iteration: Number of games to generate per training iteration.
        buffer_size: Maximum training examples to keep.
        min_buffer_size: Minimum examples before training starts.
    """

    # MCTS settings
    mcts_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    mcts_batch_size: int = 8
    virtual_loss: float = 3.0

    # Temperature schedule
    temperature: float = 1.0
    temperature_threshold: int = 30  # After 30 moves, use greedy

    # Training settings
    learning_rate: float = 2e-4
    value_coef: float = 1.0
    l2_reg: float = 1e-4
    train_batch_size: int = 2048
    epochs_per_iteration: int = 4
    games_per_iteration: int = 256
    max_grad_norm: float = 1.0

    # Buffer settings
    buffer_size: int = 500_000
    min_buffer_size: int = 10_000

    # Device
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mcts_simulations": self.mcts_simulations,
            "c_puct": self.c_puct,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            "mcts_batch_size": self.mcts_batch_size,
            "virtual_loss": self.virtual_loss,
            "temperature": self.temperature,
            "temperature_threshold": self.temperature_threshold,
            "learning_rate": self.learning_rate,
            "value_coef": self.value_coef,
            "l2_reg": self.l2_reg,
            "train_batch_size": self.train_batch_size,
            "epochs_per_iteration": self.epochs_per_iteration,
            "games_per_iteration": self.games_per_iteration,
            "max_grad_norm": self.max_grad_norm,
            "buffer_size": self.buffer_size,
            "min_buffer_size": self.min_buffer_size,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCTSPPOConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# Training Example Data Structure
# ============================================================================


@dataclass
class MCTSTrainingExample:
    """Single training example from MCTS self-play.

    Attributes:
        observation: Encoded game state observation.
        action_mask: Legal action mask.
        mcts_policy: MCTS-improved policy (visit count distribution).
        value: Game outcome from this player's perspective (-1, 0, or 1).
        player: Which player generated this example.
    """

    observation: np.ndarray
    action_mask: np.ndarray
    mcts_policy: np.ndarray  # (ACTION_DIM,) probability distribution
    value: float
    player: int


@dataclass
class MCTSTrainingBuffer:
    """Replay buffer for MCTS training examples.

    Stores training examples with MCTS policy targets and game outcomes.
    Uses FIFO eviction when buffer is full.
    """

    max_size: int
    examples: list[MCTSTrainingExample] = field(default_factory=list)

    def add(self, example: MCTSTrainingExample) -> None:
        """Add single example to buffer."""
        self.examples.append(example)
        if len(self.examples) > self.max_size:
            # FIFO eviction
            self.examples.pop(0)

    def add_batch(self, examples: list[MCTSTrainingExample]) -> None:
        """Add batch of examples to buffer."""
        for example in examples:
            self.add(example)

    def sample(self, batch_size: int) -> list[MCTSTrainingExample]:
        """Sample random batch of examples."""
        if len(self.examples) < batch_size:
            return self.examples.copy()
        indices = np.random.choice(len(self.examples), batch_size, replace=False)
        return [self.examples[i] for i in indices]

    def get_all(self) -> list[MCTSTrainingExample]:
        """Get all examples."""
        return self.examples.copy()

    def clear(self) -> None:
        """Clear all examples."""
        self.examples.clear()

    def __len__(self) -> int:
        """Return number of examples in buffer."""
        return len(self.examples)


# ============================================================================
# MCTS-Enhanced PPO Trainer
# ============================================================================


class MCTSEnhancedPPO:
    """PPO training with MCTS policy targets (AlphaZero-style).

    This trainer wraps existing PPO infrastructure but uses MCTS to generate
    improved policy targets during game generation. The key insight from
    AlphaZero is that MCTS provides a natural policy improvement operator:
    the visit count distribution from search is a better policy than the
    raw network output.

    Training loop:
    1. Generate games using MCTS for action selection
    2. Store (state, MCTS_policy, game_outcome) triples
    3. Train network to predict MCTS_policy and game_outcome
    4. Repeat

    The trained network learns to predict what MCTS would output, effectively
    distilling the search into the network.

    Attributes:
        network: Policy-value neural network.
        config: MCTS-PPO configuration.
        mcts: Batch MCTS instance for efficient parallel search.
        buffer: Training example buffer.
        optimizer: PyTorch optimizer.
        device: Training device.

    Example:
        trainer = MCTSEnhancedPPO(network, config)
        for iteration in range(1000):
            metrics = trainer.train_iteration()
            print(f"Iter {iteration}: {metrics}")
    """

    def __init__(
        self,
        network: BeastyBarNetwork,
        config: MCTSPPOConfig | None = None,
    ) -> None:
        """Initialize MCTS-enhanced PPO trainer.

        Args:
            network: Policy-value neural network to train.
            config: Training configuration. Uses defaults if None.
        """
        _ensure_torch()

        self.config = config or MCTSPPOConfig()
        self.network = network

        # Determine device
        if self.config.device == "auto":
            try:
                self._device = next(network.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self.config.device)

        self.network = self.network.to(self._device)

        # Create batch MCTS for efficient game generation
        self.mcts = BatchMCTS(
            network=self.network,
            num_simulations=self.config.mcts_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            virtual_loss=self.config.virtual_loss,
            batch_size=self.config.mcts_batch_size,
            device=self._device,
        )

        # Training buffer
        self.buffer = MCTSTrainingBuffer(max_size=self.config.buffer_size)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_reg,
        )

        # Statistics
        self._iteration = 0
        self._total_games = 0
        self._total_steps = 0

        logger.info(
            f"MCTSEnhancedPPO initialized: "
            f"device={self._device}, "
            f"mcts_sims={self.config.mcts_simulations}, "
            f"batch_size={self.config.train_batch_size}"
        )

    @property
    def device(self) -> torch.device:
        """Return training device."""
        return self._device

    # ========================================================================
    # Game Generation with MCTS
    # ========================================================================

    def generate_games(
        self,
        num_games: int,
        opponent: Agent | None = None,
    ) -> list[MCTSTrainingExample]:
        """Generate training data from MCTS self-play games.

        Each game generates (state, mcts_policy, outcome) training examples.
        The MCTS policy provides improved targets compared to raw network output.

        Args:
            num_games: Number of games to generate.
            opponent: Optional opponent for player 1. If None, uses MCTS self-play.

        Returns:
            List of training examples from all games.
        """
        all_examples: list[MCTSTrainingExample] = []

        # Generate games in parallel batches for efficiency
        games_remaining = num_games
        while games_remaining > 0:
            batch_size = min(games_remaining, 16)  # Process up to 16 games at once
            batch_examples = self._generate_game_batch(batch_size, opponent)
            all_examples.extend(batch_examples)
            games_remaining -= batch_size
            self._total_games += batch_size

        logger.debug(
            f"Generated {num_games} games, {len(all_examples)} training examples"
        )
        return all_examples

    def _generate_game_batch(
        self,
        batch_size: int,
        opponent: Agent | None = None,
    ) -> list[MCTSTrainingExample]:
        """Generate a batch of games efficiently using parallel MCTS.

        Args:
            batch_size: Number of games to generate in parallel.
            opponent: Optional opponent for player 1.

        Returns:
            Training examples from all games in batch.
        """
        # Initialize game states
        games: list[_GameInProgress] = []
        for _i in range(batch_size):
            seed = np.random.randint(0, 2**31)
            game_state = engine.initial_state(seed=seed)
            games.append(
                _GameInProgress(
                    state=game_state,
                    seed=seed,
                    pending_examples=[],
                    move_count=0,
                )
            )

        # Play games until all complete
        self.network.eval()
        with torch.no_grad():
            while any(not g.is_done for g in games):
                # Collect active games
                active_games = [g for g in games if not g.is_done]
                [g.state for g in active_games]

                # Determine which player is active (all same in 2-player alternating game)
                # Group by active player for efficient MCTS
                player_groups: dict[int, list[tuple[int, _GameInProgress]]] = {0: [], 1: []}
                for idx, game in enumerate(active_games):
                    player = game.state.active_player
                    player_groups[player].append((idx, game))

                # Process each player group
                for player, group in player_groups.items():
                    if not group:
                        continue

                    group_states = [g.state for _, g in group]

                    # Use MCTS for player 0 or self-play
                    # Use opponent for player 1 if provided
                    if player == 1 and opponent is not None:
                        # Opponent moves (no training data collected)
                        for _, game in group:
                            legal = engine.legal_actions(game.state)
                            action = opponent.select_action(game.state, legal)
                            game.state = engine.step(game.state, action)
                            game.move_count += 1
                    else:
                        # MCTS search for batch of states
                        mcts_policies = self.mcts.search_batch(
                            states=group_states,
                            perspective=player,
                            temperature=self._get_temperature(group[0][1].move_count),
                            add_root_noise=True,
                        )

                        # Process each game in group
                        for (_, game), mcts_policy in zip(group, mcts_policies, strict=False):
                            # Create training example
                            obs = state_to_tensor(game.state, player)
                            mask = legal_action_mask_tensor(game.state, player)

                            # Convert mcts_policy dict to dense array
                            policy_array = np.zeros(ACTION_DIM, dtype=np.float32)
                            for action_idx, prob in mcts_policy.items():
                                policy_array[action_idx] = prob

                            example = MCTSTrainingExample(
                                observation=obs,
                                action_mask=mask,
                                mcts_policy=policy_array,
                                value=0.0,  # Filled in after game ends
                                player=player,
                            )
                            game.pending_examples.append(example)

                            # Sample action from MCTS policy
                            action_idx = self._sample_from_policy(
                                mcts_policy,
                                self._get_temperature(game.move_count),
                            )
                            action = index_to_action(action_idx)
                            game.state = engine.step(game.state, action)
                            game.move_count += 1
                            self._total_steps += 1

        # Collect all examples and fill in game outcomes
        all_examples: list[MCTSTrainingExample] = []
        for game in games:
            # Compute final scores and determine value
            scores = engine.score(game.state)
            p0_value = self._compute_game_value(scores[0], scores[1])

            # Fill in values for each example
            for example in game.pending_examples:
                # Value from this player's perspective
                if example.player == 0:
                    example.value = p0_value
                else:
                    example.value = -p0_value
                all_examples.append(example)

        return all_examples

    def _get_temperature(self, move_count: int) -> float:
        """Get temperature for a given move count.

        Early game uses higher temperature for exploration.
        Late game uses lower temperature for exploitation.
        """
        if move_count >= self.config.temperature_threshold:
            return 0.0  # Greedy (argmax)
        return self.config.temperature

    def _sample_from_policy(
        self,
        policy: dict[int, float],
        temperature: float,
    ) -> int:
        """Sample action from MCTS policy distribution.

        Args:
            policy: Dict mapping action indices to probabilities.
            temperature: Sampling temperature (0 = greedy, 1 = proportional).

        Returns:
            Sampled action index.
        """
        if not policy:
            raise ValueError("Cannot sample from empty policy")

        actions = list(policy.keys())
        probs = np.array([policy[a] for a in actions], dtype=np.float64)

        if temperature == 0.0 or len(actions) == 1:
            # Greedy selection
            return actions[np.argmax(probs)]

        # Apply temperature
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)

        # Normalize
        probs = probs / probs.sum()

        # Sample
        return actions[np.random.choice(len(actions), p=probs)]

    def _compute_game_value(self, my_score: int, opp_score: int) -> float:
        """Compute game value with margin bonus.

        Returns value in [-1, 1] with margin-based scaling.
        """
        if my_score > opp_score:
            # Win with margin bonus
            margin = my_score - opp_score
            max_margin = 36
            margin_bonus = 0.2 * min(1.0, margin / max_margin)
            return 0.8 + margin_bonus
        elif my_score < opp_score:
            # Loss with margin penalty
            margin = opp_score - my_score
            max_margin = 36
            margin_penalty = 0.2 * min(1.0, margin / max_margin)
            return -0.8 - margin_penalty
        else:
            return 0.0  # Draw

    # ========================================================================
    # Training
    # ========================================================================

    def train_on_buffer(self) -> dict[str, float]:
        """Train network on current buffer contents.

        Performs multiple epochs of training on shuffled buffer data.
        Uses cross-entropy loss for policy and MSE for value.

        Returns:
            Dictionary of training metrics.
        """
        if len(self.buffer) < self.config.min_buffer_size:
            logger.warning(
                f"Buffer size ({len(self.buffer)}) below minimum "
                f"({self.config.min_buffer_size}), skipping training"
            )
            return {}

        self.network.train()

        # Prepare training data
        examples = self.buffer.get_all()
        n_examples = len(examples)

        # Stack into tensors
        observations = torch.from_numpy(
            np.stack([e.observation for e in examples])
        ).float().to(self._device)

        action_masks = torch.from_numpy(
            np.stack([e.action_mask for e in examples])
        ).float().to(self._device)

        target_policies = torch.from_numpy(
            np.stack([e.mcts_policy for e in examples])
        ).float().to(self._device)

        target_values = torch.tensor(
            [e.value for e in examples], dtype=torch.float32
        ).to(self._device)

        # Training loop
        metrics_sum = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "policy_entropy": 0.0,
        }
        num_updates = 0

        for _epoch in range(self.config.epochs_per_iteration):
            # Shuffle indices
            indices = np.random.permutation(n_examples)

            # Iterate through minibatches
            for start in range(0, n_examples, self.config.train_batch_size):
                end = min(start + self.config.train_batch_size, n_examples)
                batch_indices = indices[start:end]

                # Get batch
                batch_obs = observations[batch_indices]
                batch_masks = action_masks[batch_indices]
                batch_target_policy = target_policies[batch_indices]
                batch_target_value = target_values[batch_indices]

                # Forward pass
                policy_logits, pred_values = self.network(batch_obs, batch_masks)
                pred_values = pred_values.squeeze(-1)

                # Policy loss: cross-entropy with MCTS targets
                # Mask illegal actions
                masked_logits = policy_logits.clone()
                masked_logits[batch_masks == 0] = float("-inf")

                log_probs = F.log_softmax(masked_logits, dim=-1)

                # Cross-entropy: -sum(target * log(pred))
                policy_loss = -(batch_target_policy * log_probs).sum(dim=-1).mean()

                # Value loss: MSE
                value_loss = F.mse_loss(pred_values, batch_target_value)

                # Total loss
                total_loss = policy_loss + self.config.value_coef * value_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()

                # Compute entropy for monitoring
                with torch.no_grad():
                    probs = F.softmax(masked_logits, dim=-1)
                    entropy = -(probs * log_probs.clamp(min=-100)).sum(dim=-1).mean()

                # Accumulate metrics
                metrics_sum["policy_loss"] += policy_loss.item()
                metrics_sum["value_loss"] += value_loss.item()
                metrics_sum["total_loss"] += total_loss.item()
                metrics_sum["policy_entropy"] += entropy.item()
                num_updates += 1

        # Average metrics
        if num_updates > 0:
            for key in metrics_sum:
                metrics_sum[key] /= num_updates

        metrics_sum["num_updates"] = float(num_updates)
        metrics_sum["buffer_size"] = float(len(self.buffer))

        return metrics_sum

    def train_iteration(
        self,
        opponent: Agent | None = None,
    ) -> dict[str, float]:
        """Execute one training iteration.

        1. Generate games using MCTS
        2. Add examples to buffer
        3. Train on buffer

        Args:
            opponent: Optional opponent for player 1 during generation.

        Returns:
            Dictionary of training metrics.
        """
        iteration_start = time.time()
        metrics: dict[str, float] = {"iteration": float(self._iteration)}

        # Generate games
        gen_start = time.time()
        examples = self.generate_games(
            num_games=self.config.games_per_iteration,
            opponent=opponent,
        )
        gen_time = time.time() - gen_start

        # Add to buffer
        self.buffer.add_batch(examples)

        # Train
        train_start = time.time()
        train_metrics = self.train_on_buffer()
        train_time = time.time() - train_start

        # Combine metrics
        metrics.update(train_metrics)
        metrics["self_play/games_generated"] = float(self.config.games_per_iteration)
        metrics["self_play/examples_generated"] = float(len(examples))
        metrics["self_play/generation_time"] = gen_time
        metrics["train/training_time"] = train_time
        metrics["total_games"] = float(self._total_games)
        metrics["total_steps"] = float(self._total_steps)
        metrics["iteration_time"] = time.time() - iteration_start

        self._iteration += 1

        return metrics

    # ========================================================================
    # Checkpointing
    # ========================================================================

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "iteration": self._iteration,
            "total_games": self._total_games,
            "total_steps": self._total_steps,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "buffer_size": len(self.buffer),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved MCTS-PPO checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        self._iteration = checkpoint["iteration"]
        self._total_games = checkpoint["total_games"]
        self._total_steps = checkpoint["total_steps"]
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Recreate MCTS with loaded network
        self.mcts = BatchMCTS(
            network=self.network,
            num_simulations=self.config.mcts_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            virtual_loss=self.config.virtual_loss,
            batch_size=self.config.mcts_batch_size,
            device=self._device,
        )

        logger.info(
            f"Loaded MCTS-PPO checkpoint from {path} "
            f"(iteration {self._iteration}, {self._total_games} games)"
        )


# ============================================================================
# Internal Data Structures
# ============================================================================


@dataclass
class _GameInProgress:
    """Internal state for a game being generated."""

    state: state.State
    seed: int
    pending_examples: list[MCTSTrainingExample]
    move_count: int

    @property
    def is_done(self) -> bool:
        """Check if game is finished."""
        return engine.is_terminal(self.state)


# ============================================================================
# Factory Functions
# ============================================================================


def create_mcts_ppo_trainer(
    network: BeastyBarNetwork,
    mcts_simulations: int = 100,
    learning_rate: float = 2e-4,
    games_per_iteration: int = 256,
    **kwargs: Any,
) -> MCTSEnhancedPPO:
    """Create MCTS-enhanced PPO trainer with common defaults.

    Args:
        network: Neural network to train.
        mcts_simulations: Number of MCTS simulations per move.
        learning_rate: Optimizer learning rate.
        games_per_iteration: Games to generate per iteration.
        **kwargs: Additional config parameters.

    Returns:
        Configured MCTSEnhancedPPO trainer.
    """
    config = MCTSPPOConfig(
        mcts_simulations=mcts_simulations,
        learning_rate=learning_rate,
        games_per_iteration=games_per_iteration,
        **kwargs,
    )
    return MCTSEnhancedPPO(network=network, config=config)


__all__ = [
    "MCTSEnhancedPPO",
    "MCTSPPOConfig",
    "MCTSTrainingBuffer",
    "MCTSTrainingExample",
    "create_mcts_ppo_trainer",
]
