"""AlphaZero-style MCTS self-play for neural network training.

This module implements self-play game generation using MCTS with neural network
guidance, following the AlphaZero approach:
1. Run MCTS at each position using neural network policy/value predictions
2. Sample actions from MCTS visit counts with temperature control
3. Store full game trajectories with MCTS policies (visit distributions)
4. After game completion, assign terminal values based on outcome

Key AlphaZero Features:
- Policy target = MCTS visit counts (normalized), not raw network output
- Value target = game outcome (+1/-1/0)
- Temperature starts at 1.0, drops to near 0 after configurable move threshold
- Dirichlet noise (alpha, epsilon) at root for exploration

Example:
    from _03_training.mcts_self_play import generate_mcts_games, MCTSConfig

    # Configure MCTS
    config = MCTSConfig(
        num_simulations=200,
        temperature=1.0,
        temperature_drop_move=15,
    )

    # Generate games with MCTS-guided self-play
    trajectories = generate_mcts_games(
        network=network,
        num_games=100,
        config=config,
    )

    # Convert to training data
    data = mcts_trajectories_to_training_data(trajectories)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from _01_simulator import action_space, observations, simulate
from _02_agents.mcts.batch_mcts import BatchMCTS
from _03_training.game_utils import compute_rewards, compute_winner
from _03_training.trajectory import MCTSPendingStep, MCTSStep
from _03_training.utils import inference_mode

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import nn

logger = logging.getLogger(__name__)

# Safety limits
_MAX_GAME_LENGTH = 200  # Prevent infinite games
_MIN_SIMULATIONS = 10  # Minimum MCTS simulations per move


# ============================================================================
# MCTS Configuration
# ============================================================================


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for AlphaZero-style MCTS.

    Attributes:
        num_simulations: Number of MCTS simulations per move (search depth).
        c_puct: Exploration constant for PUCT formula (higher = more exploration).
        temperature: Initial temperature for action sampling from visit counts.
        temperature_drop_move: Move number after which temperature drops to ~0.
                               Set to 0 to always use greedy selection.
        final_temperature: Temperature to use after temperature_drop_move.
        dirichlet_alpha: Dirichlet noise concentration parameter for root exploration.
        dirichlet_epsilon: Mixing weight for Dirichlet noise at root.
        add_root_noise: Whether to add Dirichlet noise at root (for training).
        batch_size: Number of leaves to batch for neural network evaluation.
                    Higher values = more GPU efficiency, but more memory.
        virtual_loss: Temporary penalty for parallel MCTS selection (exploration).
    """

    num_simulations: int = 200
    c_puct: float = 1.5
    temperature: float = 1.0
    temperature_drop_move: int = 15
    final_temperature: float = 0.1
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    add_root_noise: bool = True
    batch_size: int = 16  # Batch leaf evaluations for GPU efficiency
    virtual_loss: float = 3.0  # Virtual loss for parallel selection

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

    @classmethod
    def from_dict(cls, data: dict) -> MCTSConfig:
        """Create config from dictionary."""
        known_fields = {
            "num_simulations",
            "c_puct",
            "temperature",
            "temperature_drop_move",
            "final_temperature",
            "dirichlet_alpha",
            "dirichlet_epsilon",
            "add_root_noise",
            "batch_size",
            "virtual_loss",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "num_simulations": self.num_simulations,
            "c_puct": self.c_puct,
            "temperature": self.temperature,
            "temperature_drop_move": self.temperature_drop_move,
            "final_temperature": self.final_temperature,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            "add_root_noise": self.add_root_noise,
            "batch_size": self.batch_size,
            "virtual_loss": self.virtual_loss,
        }


# Default MCTS config
DEFAULT_MCTS_CONFIG = MCTSConfig()


# ============================================================================
# MCTS Trajectory Data Structures
# ============================================================================

# Internal alias for backward compatibility with local code
_PendingStep = MCTSPendingStep


@dataclass
class MCTSTrajectory:
    """Complete MCTS self-play game trajectory.

    Contains all steps for both players, with MCTS policies and
    terminal values assigned.

    Attributes:
        steps_p0: Steps for player 0.
        steps_p1: Steps for player 1.
        winner: Winner (0, 1) or None for draw.
        final_scores: Tuple of (score_p0, score_p1).
        game_length: Total number of moves.
        seed: Random seed used.
    """

    steps_p0: list[MCTSStep] = field(default_factory=list)
    steps_p1: list[MCTSStep] = field(default_factory=list)
    winner: int | None = None
    final_scores: tuple[int, int] = (0, 0)
    game_length: int = 0
    seed: int = 0

    def total_steps(self) -> int:
        """Return total number of steps across both players."""
        return len(self.steps_p0) + len(self.steps_p1)

    def all_steps(self) -> list[MCTSStep]:
        """Return all steps from both players interleaved by game order."""
        return self.steps_p0 + self.steps_p1


# ============================================================================
# Training Data Structures
# ============================================================================


@dataclass
class MCTSTrainingData:
    """Training data from MCTS self-play games.

    This is the data format used to train the neural network to match
    MCTS policies and values.

    Attributes:
        observations: Observation tensors, shape (N, OBSERVATION_DIM).
        action_masks: Legal action masks, shape (N, ACTION_DIM).
        policy_targets: MCTS policy distributions, shape (N, ACTION_DIM).
        value_targets: Game outcomes, shape (N,).
    """

    observations: NDArray[np.float32]  # (N, OBSERVATION_DIM)
    action_masks: NDArray[np.float32]  # (N, ACTION_DIM)
    policy_targets: NDArray[np.float32]  # (N, ACTION_DIM)
    value_targets: NDArray[np.float32]  # (N,)

    def __len__(self) -> int:
        """Return number of training samples."""
        return len(self.observations)

    def shuffle(self) -> MCTSTrainingData:
        """Return shuffled copy of training data."""
        indices = np.random.permutation(len(self))
        return MCTSTrainingData(
            observations=self.observations[indices],
            action_masks=self.action_masks[indices],
            policy_targets=self.policy_targets[indices],
            value_targets=self.value_targets[indices],
        )


def mcts_trajectories_to_training_data(
    trajectories: list[MCTSTrajectory],
) -> MCTSTrainingData:
    """Convert MCTS trajectories to training data.

    Args:
        trajectories: List of MCTS trajectories from self-play.

    Returns:
        MCTSTrainingData suitable for neural network training.
    """
    all_observations = []
    all_action_masks = []
    all_policy_targets = []
    all_value_targets = []

    for traj in trajectories:
        for step in traj.steps_p0 + traj.steps_p1:
            all_observations.append(step.observation)
            all_action_masks.append(step.action_mask)
            all_policy_targets.append(step.mcts_policy)
            all_value_targets.append(step.value)

    if not all_observations:
        # Return empty data
        return MCTSTrainingData(
            observations=np.zeros((0, 0), dtype=np.float32),
            action_masks=np.zeros((0, action_space.ACTION_DIM), dtype=np.float32),
            policy_targets=np.zeros((0, action_space.ACTION_DIM), dtype=np.float32),
            value_targets=np.zeros((0,), dtype=np.float32),
        )

    return MCTSTrainingData(
        observations=np.stack(all_observations, axis=0),
        action_masks=np.stack(all_action_masks, axis=0),
        policy_targets=np.stack(all_policy_targets, axis=0),
        value_targets=np.array(all_value_targets, dtype=np.float32),
    )


# ============================================================================
# Neural MCTS Search (Using BatchMCTS for efficiency)
# ============================================================================


def _create_batch_mcts(
    network: nn.Module,
    config: MCTSConfig,
    device: torch.device,
) -> BatchMCTS:
    """Create a BatchMCTS instance from config.

    Args:
        network: Neural network for policy/value prediction.
        config: MCTS configuration.
        device: Device for neural network inference.

    Returns:
        Configured BatchMCTS instance.
    """
    return BatchMCTS(
        network=network,
        num_simulations=max(config.num_simulations, _MIN_SIMULATIONS),
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        virtual_loss=config.virtual_loss,
        batch_size=config.batch_size,
        device=device,
    )


def _mcts_search(
    network: nn.Module,
    game_state: simulate.State,
    perspective: int,
    device: torch.device,
    config: MCTSConfig,
    batch_mcts: BatchMCTS | None = None,
) -> dict[int, float]:
    """Run MCTS search and return visit count distribution.

    Uses BatchMCTS for efficient batched leaf evaluation.

    Args:
        network: Neural network for policy/value prediction.
        game_state: Current game state to search from.
        perspective: Player perspective for value computation.
        device: Device for neural network inference.
        config: MCTS configuration.
        batch_mcts: Optional pre-created BatchMCTS instance (for reuse).

    Returns:
        Dictionary mapping action indices to visit count probabilities.
    """
    # Use provided BatchMCTS or create new one
    if batch_mcts is None:
        batch_mcts = _create_batch_mcts(network, config, device)

    # Run batched search (single state)
    results = batch_mcts.search_batch(
        states=[game_state],
        perspective=perspective,
        temperature=1.0,  # Temperature applied later during action selection
        add_root_noise=config.add_root_noise,
    )

    return results[0] if results else {}


def _visit_distribution_to_dense(
    visit_distribution: dict[int, float],
    action_dim: int | None = None,
) -> NDArray[np.float32]:
    """Convert sparse visit distribution to dense policy array.

    Args:
        visit_distribution: Dictionary mapping action indices to probabilities.
        action_dim: Size of action space.

    Returns:
        Dense policy array of shape (action_dim,).
    """
    if action_dim is None:
        action_dim = action_space.ACTION_DIM
    policy_array = np.zeros(action_dim, dtype=np.float32)
    for action_idx, prob in visit_distribution.items():
        if 0 <= action_idx < action_dim:
            policy_array[action_idx] = prob
    return policy_array


def _sample_action_from_visits(
    visit_distribution: dict[int, float],
    temperature: float,
) -> int:
    """Sample action from MCTS visit distribution with temperature.

    NOTE: This function is intentionally separate from `sample_action` in
    `_02_agents.neural.utils` because it operates on visit counts/distributions
    (dict[int, float]) rather than raw policy logits (Tensor). MCTS uses
    AlphaZero-style temperature scaling: pi(a) ~ N(a)^(1/tau), which applies
    an exponent to visit counts, not softmax temperature to logits.

    Args:
        visit_distribution: Dictionary mapping action indices to visit probabilities.
        temperature: Sampling temperature (1.0 = sample proportionally, ~0 = greedy).

    Returns:
        Selected action index.
    """
    if not visit_distribution:
        raise ValueError("Cannot sample from empty visit distribution")

    actions = list(visit_distribution.keys())
    probs = np.array([visit_distribution[a] for a in actions])

    # Apply temperature
    if temperature <= 0.01:
        # Greedy: select action with highest probability
        best_idx = int(np.argmax(probs))
        return actions[best_idx]

    # Temperature scaling on visit counts (not probabilities)
    # This follows AlphaZero: pi(a) ~ N(a)^(1/tau)
    if temperature != 1.0:
        # Convert to counts-like values and apply temperature
        # Since probs are normalized, we can treat them directly
        scaled = probs ** (1.0 / temperature)
        # Re-normalize
        scaled = scaled / (scaled.sum() + 1e-10)
        probs = scaled

    # Handle numerical issues
    probs = np.clip(probs, 0, 1)
    probs = probs / (probs.sum() + 1e-10)

    # Sample
    selected_idx = np.random.choice(len(actions), p=probs)
    return actions[selected_idx]


# ============================================================================
# MCTS Self-Play Game Generation
# ============================================================================


def _get_visit_distribution_or_fallback(
    network: nn.Module,
    game_state: simulate.State,
    player: int,
    device: torch.device,
    config: MCTSConfig,
    batch_mcts: BatchMCTS,
) -> dict[int, float] | None:
    """Get MCTS visit distribution with uniform fallback.

    Args:
        network: Neural network for policy/value.
        game_state: Current game state.
        player: Active player.
        device: Device for inference.
        config: MCTS configuration.
        batch_mcts: BatchMCTS instance.

    Returns:
        Visit distribution dict, or None if no legal actions.
    """
    visit_distribution = _mcts_search(
        network=network,
        game_state=game_state,
        perspective=player,
        device=device,
        config=config,
        batch_mcts=batch_mcts,
    )

    if not visit_distribution:
        mask = action_space.legal_action_mask_tensor(game_state, player)
        legal_actions = np.where(mask > 0)[0]
        if len(legal_actions) == 0:
            return None
        visit_distribution = {int(a): 1.0 / len(legal_actions) for a in legal_actions}

    return visit_distribution


def _create_pending_step(
    game_state: simulate.State,
    player: int,
    mcts_policy: NDArray[np.float32],
    action_idx: int,
) -> _PendingStep:
    """Create a pending step for trajectory storage.

    Args:
        game_state: Current game state.
        player: Active player.
        mcts_policy: Dense MCTS policy array.
        action_idx: Selected action index.

    Returns:
        PendingStep object with observation and action data.
    """
    obs = observations.state_to_tensor(game_state, player)
    mask = action_space.legal_action_mask_tensor(game_state, player)

    return _PendingStep(
        observation=obs,
        action_mask=mask,
        mcts_policy=mcts_policy,
        action_taken=action_idx,
        player=player,
    )


def _convert_pending_to_mcts_steps(
    pending_steps: list[_PendingStep],
    terminal_value: float,
) -> list[MCTSStep]:
    """Convert pending steps to finalized MCTS steps with terminal values.

    Args:
        pending_steps: List of pending steps.
        terminal_value: Value to assign to all steps.

    Returns:
        List of finalized MCTSStep objects.
    """
    return [
        MCTSStep(
            observation=p.observation,
            action_mask=p.action_mask,
            mcts_policy=p.mcts_policy,
            action_taken=p.action_taken,
            value=terminal_value,
            player=p.player,
        )
        for p in pending_steps
    ]


def _run_mcts_game_loop(
    network: nn.Module,
    game_state: simulate.State,
    config: MCTSConfig,
    device: torch.device,
    batch_mcts: BatchMCTS,
) -> tuple[simulate.State, list[_PendingStep], list[_PendingStep], int]:
    """Run the main MCTS game loop.

    Args:
        network: Neural network for policy/value.
        game_state: Initial game state.
        config: MCTS configuration.
        device: Device for inference.
        batch_mcts: BatchMCTS instance.

    Returns:
        Tuple of (final_state, pending_p0, pending_p1, move_count).
    """
    pending_p0: list[_PendingStep] = []
    pending_p1: list[_PendingStep] = []
    move_count = 0

    while not simulate.is_terminal(game_state) and move_count < _MAX_GAME_LENGTH:
        player = game_state.active_player

        visit_distribution = _get_visit_distribution_or_fallback(
            network, game_state, player, device, config, batch_mcts
        )

        if visit_distribution is None:
            break

        mcts_policy = _visit_distribution_to_dense(visit_distribution)
        temperature = config.get_temperature(move_count)
        action_idx = _sample_action_from_visits(visit_distribution, temperature)

        pending = _create_pending_step(game_state, player, mcts_policy, action_idx)

        if player == 0:
            pending_p0.append(pending)
        else:
            pending_p1.append(pending)

        action = action_space.index_to_action(action_idx)
        game_state = simulate.apply(game_state, action)
        move_count += 1

    return game_state, pending_p0, pending_p1, move_count


def play_mcts_game(
    network: nn.Module,
    config: MCTSConfig | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    batch_mcts: BatchMCTS | None = None,
) -> MCTSTrajectory:
    """Play one self-play game using MCTS with neural network guidance.

    Implements AlphaZero-style self-play:
    1. At each state, run MCTS to get improved policy (visit distribution)
    2. Sample action from MCTS policy with temperature (annealed over game)
    3. Store MCTS policy as training target
    4. After game ends, assign terminal value to all positions

    Args:
        network: Neural network for policy/value prediction.
        config: MCTS configuration. Uses defaults if None.
        seed: Random seed for reproducibility.
        device: Device for network inference.
        batch_mcts: Optional pre-created BatchMCTS instance (for reuse).
                    If provided, significantly reduces overhead.

    Returns:
        MCTSTrajectory with complete game including MCTS policies.
    """
    if config is None:
        config = DEFAULT_MCTS_CONFIG

    if seed is None:
        seed = np.random.randint(0, 2**31)

    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    if batch_mcts is None:
        batch_mcts = _create_batch_mcts(network, config, device)

    game_state = simulate.new_game(seed)

    with inference_mode(network):
        final_state, pending_p0, pending_p1, move_count = _run_mcts_game_loop(
            network, game_state, config, device, batch_mcts
        )

    # Compute game outcome
    scores = simulate.score(final_state)
    final_scores = (scores[0], scores[1])
    winner = compute_winner(final_scores)
    value_p0, value_p1 = compute_rewards(winner)

    # Convert pending steps to final steps
    steps_p0 = _convert_pending_to_mcts_steps(pending_p0, value_p0)
    steps_p1 = _convert_pending_to_mcts_steps(pending_p1, value_p1)

    return MCTSTrajectory(
        steps_p0=steps_p0,
        steps_p1=steps_p1,
        winner=winner,
        final_scores=final_scores,
        game_length=move_count,
        seed=seed,
    )


# ============================================================================
# Batch Game Generation
# ============================================================================


def generate_mcts_games(
    network: nn.Module,
    num_games: int,
    config: MCTSConfig | None = None,
    device: torch.device | None = None,
    seeds: list[int] | None = None,
    progress_callback: callable | None = None,
) -> list[MCTSTrajectory]:
    """Generate multiple MCTS self-play games with batched neural evaluation.

    Uses BatchMCTS for efficient GPU utilization. Each MCTS search batches
    leaf evaluations, providing 5-10x speedup over sequential evaluation.

    Args:
        network: Neural network for policy/value prediction.
        num_games: Number of games to generate.
        config: MCTS configuration. Uses defaults if None.
        device: Device for network inference.
        seeds: Optional list of seeds for each game.
        progress_callback: Optional callback(game_idx, total) for progress.

    Returns:
        List of MCTSTrajectory objects.
    """
    if num_games <= 0:
        raise ValueError(f"num_games must be positive, got {num_games}")

    if config is None:
        config = DEFAULT_MCTS_CONFIG

    # Generate seeds if not provided
    if seeds is None:
        base_seed = np.random.randint(0, 2**31)
        seeds = [base_seed + i for i in range(num_games)]
    elif len(seeds) != num_games:
        raise ValueError(f"seeds length ({len(seeds)}) must match num_games ({num_games})")

    # Determine device
    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Create a single BatchMCTS instance and reuse across all games
    batch_mcts = _create_batch_mcts(network, config, device)
    logger.info(
        f"Using BatchMCTS with batch_size={config.batch_size}, "
        f"simulations={config.num_simulations}"
    )

    # Generate games with shared BatchMCTS - use inference_mode
    trajectories = []
    with inference_mode(network):
        for i, seed in enumerate(seeds):
            if progress_callback is not None:
                progress_callback(i, num_games)
            elif i % max(1, num_games // 10) == 0:
                logger.info(f"Generating MCTS game {i+1}/{num_games}")

            trajectory = play_mcts_game(
                network=network,
                config=config,
                seed=seed,
                device=device,
                batch_mcts=batch_mcts,  # Reuse BatchMCTS instance
            )
            trajectories.append(trajectory)

    return trajectories


# ============================================================================
# Parallel Game Generation (Maximum GPU Efficiency)
# ============================================================================


@dataclass
class _ParallelGameState:
    """State for a single game in parallel generation."""

    game_state: simulate.State
    seed: int
    move_count: int = 0
    pending_p0: list[_PendingStep] = field(default_factory=list)
    pending_p1: list[_PendingStep] = field(default_factory=list)
    finished: bool = False


def _init_parallel_games(
    seeds: list[int],
    seed_idx: int,
    games_to_start: int,
) -> tuple[list[_ParallelGameState], int]:
    """Initialize a batch of parallel games.

    Args:
        seeds: List of all game seeds.
        seed_idx: Current index into seeds list.
        games_to_start: Number of games to start.

    Returns:
        Tuple of (list of new game states, updated seed index).
    """
    active_games = []
    for _ in range(games_to_start):
        if seed_idx < len(seeds):
            seed = seeds[seed_idx]
            seed_idx += 1
            game_state = simulate.new_game(seed)
            active_games.append(_ParallelGameState(game_state=game_state, seed=seed))
    return active_games, seed_idx


def _collect_active_states(
    active_games: list[_ParallelGameState],
) -> tuple[list[simulate.State], list[int]]:
    """Collect states from active games that need MCTS decisions.

    Args:
        active_games: List of parallel game states.

    Returns:
        Tuple of (states needing search, corresponding game indices).
    """
    states_to_search = []
    game_indices = []

    for i, pg in enumerate(active_games):
        if not pg.finished and not simulate.is_terminal(pg.game_state):
            states_to_search.append(pg.game_state)
            game_indices.append(i)

    return states_to_search, game_indices


def _group_states_by_perspective(
    states: list[simulate.State],
    game_indices: list[int],
) -> tuple[list[tuple[simulate.State, int]], list[tuple[simulate.State, int]]]:
    """Group states by active player perspective.

    Args:
        states: List of game states.
        game_indices: Corresponding game indices.

    Returns:
        Tuple of (player 0 states with indices, player 1 states with indices).
    """
    perspectives = [s.active_player for s in states]
    p0_states = [
        (s, i) for s, i, p in zip(states, game_indices, perspectives, strict=False) if p == 0
    ]
    p1_states = [
        (s, i) for s, i, p in zip(states, game_indices, perspectives, strict=False) if p == 1
    ]
    return p0_states, p1_states


def _process_mcts_batch_for_perspective(
    states_with_indices: list[tuple[simulate.State, int]],
    perspective: int,
    batch_mcts: BatchMCTS,
    config: MCTSConfig,
    active_games: list[_ParallelGameState],
) -> None:
    """Run batched MCTS for states of a single perspective and apply results.

    Args:
        states_with_indices: List of (state, game_index) tuples.
        perspective: Player perspective (0 or 1).
        batch_mcts: BatchMCTS instance.
        config: MCTS configuration.
        active_games: List of parallel game states to update.
    """
    if not states_with_indices:
        return

    results = batch_mcts.search_batch(
        states=[s for s, _ in states_with_indices],
        perspective=perspective,
        add_root_noise=config.add_root_noise,
    )

    for (state, game_idx), visit_dist in zip(states_with_indices, results, strict=False):
        _apply_mcts_result(active_games[game_idx], visit_dist, config, state)


def _update_finished_status(active_games: list[_ParallelGameState]) -> None:
    """Mark games as finished if terminal or max length reached.

    Args:
        active_games: List of parallel game states to check.
    """
    for pg in active_games:
        if simulate.is_terminal(pg.game_state) or pg.move_count >= _MAX_GAME_LENGTH:
            pg.finished = True


def _run_parallel_game_batch(
    active_games: list[_ParallelGameState],
    batch_mcts: BatchMCTS,
    config: MCTSConfig,
) -> None:
    """Run all active games until completion.

    Args:
        active_games: List of parallel game states.
        batch_mcts: BatchMCTS instance.
        config: MCTS configuration.
    """
    while active_games:
        states_to_search, game_indices = _collect_active_states(active_games)

        if not states_to_search:
            break

        p0_states, p1_states = _group_states_by_perspective(states_to_search, game_indices)

        _process_mcts_batch_for_perspective(
            p0_states, 0, batch_mcts, config, active_games
        )
        _process_mcts_batch_for_perspective(
            p1_states, 1, batch_mcts, config, active_games
        )

        _update_finished_status(active_games)


def generate_mcts_games_parallel(
    network: nn.Module,
    num_games: int,
    config: MCTSConfig | None = None,
    device: torch.device | None = None,
    seeds: list[int] | None = None,
    parallel_games: int = 16,
    progress_callback: callable | None = None,
) -> list[MCTSTrajectory]:
    """Generate MCTS games with parallel execution and cross-game batching.

    This is the most efficient method for GPU utilization. It runs multiple
    games simultaneously and batches MCTS searches across all in-progress
    games, maximizing GPU throughput.

    Performance characteristics:
    - Batches leaf evaluations within each MCTS search (batch_size config)
    - Batches MCTS searches across multiple games (parallel_games parameter)
    - Expected speedup: 10-50x over sequential generation on GPU

    Args:
        network: Neural network for policy/value prediction.
        num_games: Total number of games to generate.
        config: MCTS configuration. Uses defaults if None.
        device: Device for network inference.
        seeds: Optional list of seeds for each game.
        parallel_games: Number of games to run in parallel (default 16).
                        Higher = more GPU efficiency but more memory.
        progress_callback: Optional callback(completed, total) for progress.

    Returns:
        List of MCTSTrajectory objects.
    """
    if num_games <= 0:
        raise ValueError(f"num_games must be positive, got {num_games}")

    if config is None:
        config = DEFAULT_MCTS_CONFIG

    # Generate seeds if not provided
    if seeds is None:
        base_seed = np.random.randint(0, 2**31)
        seeds = [base_seed + i for i in range(num_games)]
    elif len(seeds) != num_games:
        raise ValueError(f"seeds length ({len(seeds)}) must match num_games ({num_games})")

    # Determine device
    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Create BatchMCTS with higher batch size for cross-game batching
    effective_batch_size = max(config.batch_size, parallel_games * 2)
    batch_mcts = BatchMCTS(
        network=network,
        num_simulations=max(config.num_simulations, _MIN_SIMULATIONS),
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        virtual_loss=config.virtual_loss,
        batch_size=effective_batch_size,
        device=device,
    )

    logger.info(
        f"Parallel MCTS: {parallel_games} games, "
        f"batch_size={effective_batch_size}, sims={config.num_simulations}"
    )

    trajectories: list[MCTSTrajectory] = []
    seed_idx = 0

    with inference_mode(network):
        while len(trajectories) < num_games:
            games_to_start = min(parallel_games, num_games - len(trajectories))
            active_games, seed_idx = _init_parallel_games(seeds, seed_idx, games_to_start)

            _run_parallel_game_batch(active_games, batch_mcts, config)

            # Finalize completed games
            for pg in active_games:
                traj = _finalize_parallel_game(pg)
                trajectories.append(traj)

            if progress_callback is not None:
                progress_callback(len(trajectories), num_games)
            else:
                logger.info(f"Generated {len(trajectories)}/{num_games} MCTS games")

    return trajectories


def _apply_mcts_result(
    pg: _ParallelGameState,
    visit_distribution: dict[int, float],
    config: MCTSConfig,
    game_state: simulate.State,
) -> None:
    """Apply MCTS result to a parallel game state."""
    player = game_state.active_player

    # Handle edge case: no valid actions
    if not visit_distribution:
        mask = action_space.legal_action_mask_tensor(game_state, player)
        legal_actions = np.where(mask > 0)[0]
        if len(legal_actions) == 0:
            pg.finished = True
            return
        visit_distribution = {int(a): 1.0 / len(legal_actions) for a in legal_actions}

    # Convert to dense policy
    mcts_policy = _visit_distribution_to_dense(visit_distribution)

    # Get temperature and sample action
    temperature = config.get_temperature(pg.move_count)
    action_idx = _sample_action_from_visits(visit_distribution, temperature)

    # Get observation and mask
    obs = observations.state_to_tensor(game_state, player)
    mask = action_space.legal_action_mask_tensor(game_state, player)

    # Store pending step
    pending = _PendingStep(
        observation=obs,
        action_mask=mask,
        mcts_policy=mcts_policy,
        action_taken=action_idx,
        player=player,
    )

    if player == 0:
        pg.pending_p0.append(pending)
    else:
        pg.pending_p1.append(pending)

    # Apply action
    action = action_space.index_to_action(action_idx)
    pg.game_state = simulate.apply(pg.game_state, action)
    pg.move_count += 1


def _finalize_parallel_game(pg: _ParallelGameState) -> MCTSTrajectory:
    """Convert parallel game state to trajectory with terminal values."""
    scores = simulate.score(pg.game_state)
    final_scores = (scores[0], scores[1])
    winner = compute_winner(final_scores)
    value_p0, value_p1 = compute_rewards(winner)

    return MCTSTrajectory(
        steps_p0=_convert_pending_to_mcts_steps(pg.pending_p0, value_p0),
        steps_p1=_convert_pending_to_mcts_steps(pg.pending_p1, value_p1),
        winner=winner,
        final_scores=final_scores,
        game_length=pg.move_count,
        seed=pg.seed,
    )


# ============================================================================
# Opponent-Diverse Game Generation
# ============================================================================


@dataclass
class _MixedGameState:
    """State for a game with potentially different opponent types."""

    game_state: simulate.State
    seed: int
    training_player: int  # Which player we're training (0 or 1)
    opponent_agent: object | None  # Agent for non-neural opponent, or None for self-play
    move_count: int = 0
    pending_steps: list[_PendingStep] = field(default_factory=list)
    finished: bool = False


def generate_mcts_games_with_opponents(
    network: nn.Module,
    num_games: int,
    opponent_network: nn.Module | None = None,
    opponent_agent: object | None = None,
    config: MCTSConfig | None = None,
    device: torch.device | None = None,
    parallel_games: int = 16,
) -> list[MCTSTrajectory]:
    """Generate MCTS games against various opponent types.

    Supports three modes:
    1. Self-play: opponent_network=None, opponent_agent=None (both players use network)
    2. Neural opponent: opponent_network provided (opponent uses different network)
    3. Agent opponent: opponent_agent provided (opponent uses simple agent)

    For agent opponents, only the training player (randomly assigned) uses MCTS.
    This provides training signal against diverse play styles.

    Args:
        network: Neural network for training player.
        num_games: Number of games to generate.
        opponent_network: Optional different network for opponent.
        opponent_agent: Optional agent (RandomAgent, HeuristicAgent) for opponent.
        config: MCTS configuration.
        device: Device for network inference.
        parallel_games: Number of games to run in parallel.

    Returns:
        List of MCTSTrajectory objects (only training player's perspective).
    """
    if config is None:
        config = DEFAULT_MCTS_CONFIG

    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Determine opponent type
    is_self_play = opponent_network is None and opponent_agent is None
    is_neural_opponent = opponent_network is not None

    if is_self_play:
        # Pure self-play - use existing parallel function
        return generate_mcts_games_parallel(
            network=network,
            num_games=num_games,
            config=config,
            device=device,
            parallel_games=parallel_games,
        )

    if is_neural_opponent:
        # Neural vs neural - both use MCTS
        return _generate_neural_vs_neural(
            network=network,
            opponent_network=opponent_network,
            num_games=num_games,
            config=config,
            device=device,
            parallel_games=parallel_games,
        )

    # Agent opponent - MCTS for training player, simple agent for opponent
    return _generate_mcts_vs_agent(
        network=network,
        opponent_agent=opponent_agent,
        num_games=num_games,
        config=config,
        device=device,
        parallel_games=parallel_games,
    )


def _init_mixed_game_batch(
    base_seed: int,
    batch_start: int,
    batch_size: int,
    opponent_agent: object | None = None,
) -> list[_MixedGameState]:
    """Initialize a batch of mixed games with random training player assignment.

    Args:
        base_seed: Base seed for game initialization.
        batch_start: Starting index in batch.
        batch_size: Number of games to initialize.
        opponent_agent: Optional agent for opponent moves.

    Returns:
        List of initialized MixedGameState objects.
    """
    active_games = []
    for i in range(batch_size):
        seed = base_seed + batch_start + i
        game_state = simulate.new_game(seed)
        training_player = np.random.randint(0, 2)
        active_games.append(_MixedGameState(
            game_state=game_state,
            seed=seed,
            training_player=training_player,
            opponent_agent=opponent_agent,
        ))
    return active_games


def _collect_training_and_opponent_states(
    active_games: list[_MixedGameState],
) -> tuple[list[simulate.State], list[int], list[simulate.State], list[int]]:
    """Collect states for training player and opponent moves.

    Args:
        active_games: List of active game states.

    Returns:
        Tuple of (train_states, train_indices, opp_states, opp_indices).
    """
    train_states = []
    train_indices = []
    opp_states = []
    opp_indices = []

    for i, g in enumerate(active_games):
        if g.finished or simulate.is_terminal(g.game_state):
            g.finished = True
            continue
        player = g.game_state.active_player
        if player == g.training_player:
            train_states.append(g.game_state)
            train_indices.append(i)
        else:
            opp_states.append(g.game_state)
            opp_indices.append(i)

    return train_states, train_indices, opp_states, opp_indices


def _process_training_moves_batch(
    train_states: list[simulate.State],
    train_indices: list[int],
    active_games: list[_MixedGameState],
    batch_mcts: BatchMCTS,
    config: MCTSConfig,
) -> None:
    """Process training player moves with batched MCTS.

    Args:
        train_states: States needing training player moves.
        train_indices: Indices into active_games.
        active_games: List of all active games.
        batch_mcts: BatchMCTS instance for training network.
        config: MCTS configuration.
    """
    if not train_states:
        return

    perspectives = [active_games[i].training_player for i in train_indices]

    for perspective in [0, 1]:
        p_states = [
            (s, i) for s, i, p in zip(train_states, train_indices, perspectives, strict=False)
            if p == perspective
        ]
        if p_states:
            results = batch_mcts.search_batch(
                states=[s for s, _ in p_states],
                perspective=perspective,
                add_root_noise=config.add_root_noise,
            )
            for (_, game_idx), visit_dist in zip(p_states, results, strict=False):
                _apply_mcts_result_training_only(active_games[game_idx], visit_dist, config)


def _process_opponent_mcts_moves_batch(
    opp_states: list[simulate.State],
    opp_indices: list[int],
    active_games: list[_MixedGameState],
    batch_mcts_opp: BatchMCTS,
    config: MCTSConfig,
) -> None:
    """Process opponent moves with batched MCTS (no training data).

    Args:
        opp_states: States needing opponent moves.
        opp_indices: Indices into active_games.
        active_games: List of all active games.
        batch_mcts_opp: BatchMCTS instance for opponent network.
        config: MCTS configuration.
    """
    if not opp_states:
        return

    perspectives = [1 - active_games[i].training_player for i in opp_indices]

    for perspective in [0, 1]:
        p_states = [
            (s, i) for s, i, p in zip(opp_states, opp_indices, perspectives, strict=False)
            if p == perspective
        ]
        if p_states:
            results = batch_mcts_opp.search_batch(
                states=[s for s, _ in p_states],
                perspective=perspective,
                add_root_noise=False,
            )
            for (_, game_idx), visit_dist in zip(p_states, results, strict=False):
                _apply_opponent_move(active_games[game_idx], visit_dist, config)


def _update_mixed_games_finished_status(active_games: list[_MixedGameState]) -> None:
    """Update finished status for all mixed games.

    Args:
        active_games: List of active games to check.
    """
    for g in active_games:
        if simulate.is_terminal(g.game_state) or g.move_count >= _MAX_GAME_LENGTH:
            g.finished = True


def _finalize_mixed_games_batch(
    active_games: list[_MixedGameState],
) -> list[MCTSTrajectory]:
    """Finalize a batch of mixed games to trajectories.

    Args:
        active_games: List of completed games.

    Returns:
        List of MCTSTrajectory objects.
    """
    return [_finalize_training_game(g) for g in active_games]


def _generate_neural_vs_neural(
    network: nn.Module,
    opponent_network: nn.Module,
    num_games: int,
    config: MCTSConfig,
    device: torch.device,
    parallel_games: int,
) -> list[MCTSTrajectory]:
    """Generate games between two neural networks using MCTS."""
    from _02_agents.mcts.batch_mcts import BatchMCTS

    effective_batch_size = max(config.batch_size, parallel_games * 2)

    batch_mcts_train = BatchMCTS(
        network=network,
        num_simulations=max(config.num_simulations, _MIN_SIMULATIONS),
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        virtual_loss=config.virtual_loss,
        batch_size=effective_batch_size,
        device=device,
    )

    batch_mcts_opp = BatchMCTS(
        network=opponent_network,
        num_simulations=max(config.num_simulations, _MIN_SIMULATIONS),
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        virtual_loss=config.virtual_loss,
        batch_size=effective_batch_size,
        device=device,
    )

    trajectories: list[MCTSTrajectory] = []
    base_seed = np.random.randint(0, 2**31)

    with inference_mode(network):
        opponent_network.eval()
        for batch_start in range(0, num_games, parallel_games):
            batch_size = min(parallel_games, num_games - batch_start)
            active_games = _init_mixed_game_batch(base_seed, batch_start, batch_size)

            while any(not g.finished for g in active_games):
                train_states, train_indices, opp_states, opp_indices = (
                    _collect_training_and_opponent_states(active_games)
                )

                _process_training_moves_batch(
                    train_states, train_indices, active_games, batch_mcts_train, config
                )
                _process_opponent_mcts_moves_batch(
                    opp_states, opp_indices, active_games, batch_mcts_opp, config
                )

                _update_mixed_games_finished_status(active_games)

            trajectories.extend(_finalize_mixed_games_batch(active_games))
            logger.info(f"Generated {len(trajectories)}/{num_games} neural-vs-neural games")

    return trajectories


def _collect_training_and_agent_states(
    active_games: list[_MixedGameState],
) -> tuple[list[simulate.State], list[int], list[int]]:
    """Collect states for training player and agent moves.

    Args:
        active_games: List of active game states.

    Returns:
        Tuple of (train_states, train_indices, agent_game_indices).
    """
    train_states = []
    train_indices = []
    agent_games = []

    for i, g in enumerate(active_games):
        if g.finished or simulate.is_terminal(g.game_state):
            g.finished = True
            continue
        player = g.game_state.active_player
        if player == g.training_player:
            train_states.append(g.game_state)
            train_indices.append(i)
        else:
            agent_games.append(i)

    return train_states, train_indices, agent_games


def _process_agent_moves_batch(
    agent_game_indices: list[int],
    active_games: list[_MixedGameState],
) -> None:
    """Process all agent moves in the batch.

    Args:
        agent_game_indices: Indices of games needing agent moves.
        active_games: List of all active games.
    """
    for game_idx in agent_game_indices:
        _apply_agent_move(active_games[game_idx])


def _generate_mcts_vs_agent(
    network: nn.Module,
    opponent_agent: object,
    num_games: int,
    config: MCTSConfig,
    device: torch.device,
    parallel_games: int,
) -> list[MCTSTrajectory]:
    """Generate games between MCTS network and simple agent."""
    from _02_agents.mcts.batch_mcts import BatchMCTS

    effective_batch_size = max(config.batch_size, parallel_games * 2)
    batch_mcts = BatchMCTS(
        network=network,
        num_simulations=max(config.num_simulations, _MIN_SIMULATIONS),
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        virtual_loss=config.virtual_loss,
        batch_size=effective_batch_size,
        device=device,
    )

    trajectories: list[MCTSTrajectory] = []
    base_seed = np.random.randint(0, 2**31)

    with inference_mode(network):
        for batch_start in range(0, num_games, parallel_games):
            batch_size = min(parallel_games, num_games - batch_start)
            active_games = _init_mixed_game_batch(
                base_seed, batch_start, batch_size, opponent_agent
            )

            while any(not g.finished for g in active_games):
                train_states, train_indices, agent_games = (
                    _collect_training_and_agent_states(active_games)
                )

                _process_training_moves_batch(
                    train_states, train_indices, active_games, batch_mcts, config
                )
                _process_agent_moves_batch(agent_games, active_games)

                _update_mixed_games_finished_status(active_games)

            trajectories.extend(_finalize_mixed_games_batch(active_games))
            logger.info(f"Generated {len(trajectories)}/{num_games} MCTS-vs-agent games")

    return trajectories


def _apply_mcts_result_training_only(
    g: _MixedGameState,
    visit_distribution: dict[int, float],
    config: MCTSConfig,
) -> None:
    """Apply MCTS result and store training data for training player only."""
    player = g.game_state.active_player

    if not visit_distribution:
        mask = action_space.legal_action_mask_tensor(g.game_state, player)
        legal_actions = np.where(mask > 0)[0]
        if len(legal_actions) == 0:
            g.finished = True
            return
        visit_distribution = {int(a): 1.0 / len(legal_actions) for a in legal_actions}

    mcts_policy = _visit_distribution_to_dense(visit_distribution)
    temperature = config.get_temperature(g.move_count)
    action_idx = _sample_action_from_visits(visit_distribution, temperature)

    # Store training data
    obs = observations.state_to_tensor(g.game_state, player)
    mask = action_space.legal_action_mask_tensor(g.game_state, player)

    g.pending_steps.append(_PendingStep(
        observation=obs,
        action_mask=mask,
        mcts_policy=mcts_policy,
        action_taken=action_idx,
        player=player,
    ))

    action = action_space.index_to_action(action_idx)
    g.game_state = simulate.apply(g.game_state, action)
    g.move_count += 1


def _apply_opponent_move(
    g: _MixedGameState,
    visit_distribution: dict[int, float],
    config: MCTSConfig,
) -> None:
    """Apply opponent MCTS move (no training data stored)."""
    player = g.game_state.active_player

    if not visit_distribution:
        mask = action_space.legal_action_mask_tensor(g.game_state, player)
        legal_actions = np.where(mask > 0)[0]
        if len(legal_actions) == 0:
            g.finished = True
            return
        visit_distribution = {int(a): 1.0 / len(legal_actions) for a in legal_actions}

    temperature = config.get_temperature(g.move_count)
    action_idx = _sample_action_from_visits(visit_distribution, temperature)

    action = action_space.index_to_action(action_idx)
    g.game_state = simulate.apply(g.game_state, action)
    g.move_count += 1


def _apply_agent_move(g: _MixedGameState) -> None:
    """Apply simple agent move (no MCTS, no training data)."""
    player = g.game_state.active_player
    legal = list(simulate.legal_actions(g.game_state, player))

    if not legal:
        g.finished = True
        return

    # Get action from agent
    action = g.opponent_agent.select_action(g.game_state, legal)

    g.game_state = simulate.apply(g.game_state, action)
    g.move_count += 1


def _finalize_training_game(g: _MixedGameState) -> MCTSTrajectory:
    """Convert mixed game to trajectory (training player only)."""
    scores = simulate.score(g.game_state)
    final_scores = (scores[0], scores[1])

    train_score = scores[g.training_player]
    opp_score = scores[1 - g.training_player]

    if train_score > opp_score:
        value = 1.0
        winner = g.training_player
    elif train_score < opp_score:
        value = -1.0
        winner = 1 - g.training_player
    else:
        value = 0.0
        winner = None

    # All pending steps are from training player
    steps = [
        MCTSStep(
            observation=p.observation,
            action_mask=p.action_mask,
            mcts_policy=p.mcts_policy,
            action_taken=p.action_taken,
            value=value,
            player=p.player,
        )
        for p in g.pending_steps
    ]

    # Put in appropriate player list
    if g.training_player == 0:
        steps_p0, steps_p1 = steps, []
    else:
        steps_p0, steps_p1 = [], steps

    return MCTSTrajectory(
        steps_p0=steps_p0,
        steps_p1=steps_p1,
        winner=winner,
        final_scores=final_scores,
        game_length=g.move_count,
        seed=g.seed,
    )


# ============================================================================
# Statistics
# ============================================================================


@dataclass(frozen=True, slots=True)
class MCTSSelfPlayStats:
    """Statistics from a batch of MCTS self-play games.

    Attributes:
        games_played: Total number of games.
        p0_wins: Number of wins by player 0.
        p1_wins: Number of wins by player 1.
        draws: Number of draws.
        avg_game_length: Average number of moves per game.
        total_steps: Total training samples generated.
        avg_policy_entropy: Average entropy of MCTS policies.
    """

    games_played: int
    p0_wins: int
    p1_wins: int
    draws: int
    avg_game_length: float
    total_steps: int
    avg_policy_entropy: float

    @property
    def p0_win_rate(self) -> float:
        """Win rate of player 0."""
        return self.p0_wins / self.games_played if self.games_played > 0 else 0.0

    @property
    def p1_win_rate(self) -> float:
        """Win rate of player 1."""
        return self.p1_wins / self.games_played if self.games_played > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        """Draw rate."""
        return self.draws / self.games_played if self.games_played > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert stats to dictionary for logging."""
        return {
            "games_played": float(self.games_played),
            "p0_wins": float(self.p0_wins),
            "p1_wins": float(self.p1_wins),
            "draws": float(self.draws),
            "p0_win_rate": self.p0_win_rate,
            "p1_win_rate": self.p1_win_rate,
            "draw_rate": self.draw_rate,
            "avg_game_length": self.avg_game_length,
            "total_steps": float(self.total_steps),
            "avg_policy_entropy": self.avg_policy_entropy,
        }


def _collect_mcts_step_statistics(
    trajectories: list[MCTSTrajectory],
) -> tuple[int, float]:
    """Collect step count and entropy sum from MCTS trajectories.

    Args:
        trajectories: List of MCTS trajectories.

    Returns:
        Tuple of (total_steps, entropy_sum).
    """
    total_steps = 0
    entropy_sum = 0.0

    for traj in trajectories:
        for step in traj.steps_p0 + traj.steps_p1:
            total_steps += 1
            policy = step.mcts_policy
            policy = policy[policy > 1e-10]
            if len(policy) > 0:
                entropy = -np.sum(policy * np.log(policy + 1e-10))
                entropy_sum += entropy

    return total_steps, entropy_sum


def compute_mcts_stats(trajectories: list[MCTSTrajectory]) -> MCTSSelfPlayStats:
    """Compute statistics from MCTS self-play games.

    Args:
        trajectories: List of MCTS trajectories.

    Returns:
        MCTSSelfPlayStats with aggregated statistics.
    """
    if not trajectories:
        return MCTSSelfPlayStats(
            games_played=0,
            p0_wins=0,
            p1_wins=0,
            draws=0,
            avg_game_length=0.0,
            total_steps=0,
            avg_policy_entropy=0.0,
        )

    games_played = len(trajectories)
    p0_wins = sum(1 for t in trajectories if t.winner == 0)
    p1_wins = sum(1 for t in trajectories if t.winner == 1)
    draws = sum(1 for t in trajectories if t.winner is None)

    total_length = sum(t.game_length for t in trajectories)
    avg_game_length = total_length / games_played

    total_steps, entropy_sum = _collect_mcts_step_statistics(trajectories)
    avg_policy_entropy = entropy_sum / total_steps if total_steps > 0 else 0.0

    return MCTSSelfPlayStats(
        games_played=games_played,
        p0_wins=p0_wins,
        p1_wins=p1_wins,
        draws=draws,
        avg_game_length=avg_game_length,
        total_steps=total_steps,
        avg_policy_entropy=avg_policy_entropy,
    )


# ============================================================================
# AlphaZero-style Policy Loss
# ============================================================================


def policy_loss_cross_entropy(
    policy_logits: torch.Tensor,
    policy_targets: torch.Tensor,
    action_masks: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy policy loss against MCTS targets.

    This is the AlphaZero policy loss:
        L_p = -sum(pi_mcts * log(pi_network))

    Where pi_mcts is the MCTS visit distribution and pi_network is the
    network's softmax policy.

    Args:
        policy_logits: Network policy logits, shape (batch, action_dim).
        policy_targets: MCTS policy targets, shape (batch, action_dim).
        action_masks: Legal action masks, shape (batch, action_dim).

    Returns:
        Scalar cross-entropy loss.
    """
    # Mask illegal actions
    masked_logits = policy_logits.clone()
    masked_logits[action_masks == 0] = float("-inf")

    # Log softmax for numerical stability
    log_probs = F.log_softmax(masked_logits, dim=-1)

    # Handle -inf in log_probs (from masked actions)
    log_probs = torch.where(
        torch.isinf(log_probs),
        torch.zeros_like(log_probs),
        log_probs,
    )

    # Cross-entropy: -sum(target * log_prob)
    # Only sum over actions where target > 0 (legal actions visited by MCTS)
    loss = -(policy_targets * log_probs).sum(dim=-1).mean()

    return loss


def value_loss_mse(
    predicted_values: torch.Tensor,
    value_targets: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE value loss against game outcomes.

    Args:
        predicted_values: Network value predictions, shape (batch,) or (batch, 1).
        value_targets: Game outcome targets, shape (batch,).

    Returns:
        Scalar MSE loss.
    """
    predicted_values = predicted_values.squeeze()
    return F.mse_loss(predicted_values, value_targets)


def alphazero_loss(
    policy_logits: torch.Tensor,
    value_preds: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    action_masks: torch.Tensor,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined AlphaZero loss.

    Total loss = policy_weight * L_p + value_weight * L_v

    Args:
        policy_logits: Network policy logits, shape (batch, action_dim).
        value_preds: Network value predictions, shape (batch,) or (batch, 1).
        policy_targets: MCTS policy targets, shape (batch, action_dim).
        value_targets: Game outcome targets, shape (batch,).
        action_masks: Legal action masks, shape (batch, action_dim).
        policy_weight: Weight for policy loss.
        value_weight: Weight for value loss.

    Returns:
        Tuple of (total_loss, policy_loss, value_loss).
    """
    p_loss = policy_loss_cross_entropy(policy_logits, policy_targets, action_masks)
    v_loss = value_loss_mse(value_preds, value_targets)

    total_loss = policy_weight * p_loss + value_weight * v_loss

    return total_loss, p_loss, v_loss


__all__ = [
    "DEFAULT_MCTS_CONFIG",
    "MCTSConfig",
    "MCTSSelfPlayStats",
    "MCTSStep",
    "MCTSTrainingData",
    "MCTSTrajectory",
    "alphazero_loss",
    "compute_mcts_stats",
    "generate_mcts_games",
    "generate_mcts_games_parallel",
    "generate_mcts_games_with_opponents",
    "mcts_trajectories_to_training_data",
    "play_mcts_game",
    "policy_loss_cross_entropy",
    "value_loss_mse",
]
