"""Self-play game runner for neural network training.

This module generates training games by having the neural network play
against itself or against diverse opponents from an opponent pool. Both
players can use the same network (standard self-play), or Player 1 can
use a different opponent (opponent pool diversity mode).

**Opponent Diversity**: To prevent self-play collapse, use `play_game()` with
an `opponent` parameter. Player 0 always uses the training network and its
transitions are collected for training. Player 1 can be:
- Same network (default self-play)
- Past checkpoint (prevents forgetting)
- Random agent (baseline calibration)
- Heuristic agent (quality baseline)

**GPU Optimization**: For high GPU utilization, use `generate_games_batched()`
which employs a vectorized environment to batch inference across many games.
This transforms inference from:
    512 games x 20 steps x batch_size=1 = 10,240 tiny GPU calls
To:
    ~20 steps x batch_size=512 = 20 large GPU calls

Example:
    from _03_training.self_play import play_game, generate_games

    # Run single self-play game
    trajectory = play_game(network, seed=42, temperature=1.0)

    # Play against a different opponent (opponent diversity)
    from _02_agents.random_agent import RandomAgent
    trajectory = play_game(network, seed=42, opponent=RandomAgent())

    # Generate batch of games (uses vectorized env for GPU efficiency)
    trajectories = generate_games(network, num_games=256)

    # Convert to transitions for replay buffer (only Player 0 transitions)
    transitions = trajectories_to_transitions(trajectories, player=0)
    buffer.add_batch(transitions)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from _01_simulator import simulate
from _01_simulator.action_space import (
    action_index,
    index_to_action,
    legal_action_mask_tensor,
)
from _01_simulator.observations import state_to_tensor
from _02_agents.neural.utils import sample_action

from .game_utils import compute_rewards, compute_winner
from .replay_buffer import Transition
from .trajectory import PPOPendingStep, TrajectoryStep
from .utils import inference_mode

if TYPE_CHECKING:
    from torch import nn

    from _02_agents.base import Agent

logger = logging.getLogger(__name__)

# Global variable for worker process model
_worker_network: nn.Module | None = None
_worker_device: torch.device | None = None


# ============================================================================
# Trajectory Data Structures
# ============================================================================


# Internal alias for backward compatibility with local code
_PendingStep = PPOPendingStep


@dataclass
class GameTrajectory:
    """Complete trajectory data from a single self-play game.

    Contains trajectories for both players, game outcome information,
    and metadata for reproducibility and analysis.

    Attributes:
        steps_p0: List of trajectory steps for player 0.
        steps_p1: List of trajectory steps for player 1.
        winner: Winner index (0, 1) or None for draw.
        final_scores: Tuple of (player0_score, player1_score).
        game_length: Total number of turns (steps by both players).
        seed: Random seed used for game initialization.
    """

    steps_p0: list[TrajectoryStep] = field(default_factory=list)
    steps_p1: list[TrajectoryStep] = field(default_factory=list)
    winner: int | None = None
    final_scores: tuple[int, int] = (0, 0)
    game_length: int = 0
    seed: int = 0

    def total_steps(self) -> int:
        """Return total number of steps across both players."""
        return len(self.steps_p0) + len(self.steps_p1)

    def get_steps(self, player: int) -> list[TrajectoryStep]:
        """Get trajectory steps for a specific player."""
        if player == 0:
            return self.steps_p0
        elif player == 1:
            return self.steps_p1
        else:
            raise ValueError(f"Invalid player index: {player}")


# ============================================================================
# Self-Play Game Runner Helpers
# ============================================================================


def _determine_device(network: nn.Module, device: torch.device | None) -> torch.device:
    """Determine the device for network inference.

    Args:
        network: Neural network module.
        device: Optional explicit device.

    Returns:
        Device to use for inference.
    """
    if device is not None:
        return device
    try:
        return next(network.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _handle_agent_opponent_turn(
    game_state: simulate.State,
    player: int,
    opponent: Agent,
    pending_steps: list[_PendingStep],
) -> simulate.State:
    """Handle a turn for an agent opponent (random, heuristic, etc.).

    Args:
        game_state: Current game state.
        player: Active player index.
        opponent: Agent instance with select_action method.
        pending_steps: List to append the pending step to.

    Returns:
        New game state after applying the action.
    """
    legal = simulate.legal_actions(game_state, player)
    action = opponent.select_action(game_state, legal)
    action_idx = action_index(action)

    obs = state_to_tensor(game_state, player)
    mask = legal_action_mask_tensor(game_state, player)

    pending = _PendingStep(
        observation=obs,
        action_mask=mask,
        action=action_idx,
        action_prob=1.0 / len(legal),
        value=0.0,
    )
    pending_steps.append(pending)

    return simulate.apply(game_state, action)


def _handle_neural_player_turn(
    game_state: simulate.State,
    player: int,
    network: nn.Module,
    device: torch.device,
    temperature: float,
) -> tuple[simulate.State, _PendingStep]:
    """Handle a turn for a neural network player.

    Args:
        game_state: Current game state.
        player: Active player index.
        network: Neural network for policy/value prediction.
        device: Device for inference.
        temperature: Sampling temperature.

    Returns:
        Tuple of (new game state, pending step).
    """
    obs = state_to_tensor(game_state, player)
    mask = legal_action_mask_tensor(game_state, player)

    obs_tensor = torch.from_numpy(obs).to(device)
    mask_tensor = torch.from_numpy(mask).to(device)

    policy_logits, value = network(obs_tensor, mask_tensor)

    action_idx, action_prob = sample_action(
        policy_logits, mask_tensor, temperature, return_prob=True
    )

    pending = _PendingStep(
        observation=obs,
        action_mask=mask,
        action=action_idx,
        action_prob=action_prob,
        value=float(value.squeeze().item()),
    )

    action = index_to_action(action_idx)
    new_state = simulate.apply(game_state, action)

    return new_state, pending


# ============================================================================
# Self-Play Game Runner
# ============================================================================


def play_game(
    network: nn.Module,
    seed: int | None = None,
    temperature: float = 1.0,
    device: torch.device | None = None,
    shaped_rewards: bool = False,
    opponent: Agent | nn.Module | None = None,
    opponent_network: nn.Module | None = None,
) -> GameTrajectory:
    """Run a single self-play game using the given network.

    By default, both players use the same network for action selection (standard
    self-play). Optionally, Player 1 can use a different opponent from the pool
    to provide training diversity and prevent self-play collapse.

    Args:
        network: Neural network for policy/value estimation (Player 0).
        seed: Random seed for game initialization. If None, uses random seed.
        temperature: Temperature for action sampling. Higher = more random.
        device: Device for network inference. If None, uses network's device.
        shaped_rewards: If True, use score-margin shaped rewards.
        opponent: Optional opponent agent for Player 1. If provided, must be
            an Agent subclass with select_action() method. Takes precedence
            over opponent_network.
        opponent_network: Optional neural network for Player 1. If provided,
            Player 1 uses this network instead of the training network.
            Ignored if opponent is provided.

    Returns:
        GameTrajectory containing full game data for both players.
        When using opponent diversity, only Player 0's trajectory should
        be used for training (it's the learning agent).

    Example:
        >>> # Standard self-play
        >>> trajectory = play_game(network, seed=42, temperature=1.0)
        >>> print(f"Winner: {trajectory.winner}, Length: {trajectory.game_length}")

        >>> # With opponent diversity
        >>> from _02_agents.random_agent import RandomAgent
        >>> trajectory = play_game(network, opponent=RandomAgent())
    """
    # Handle seed
    if seed is None:
        seed = np.random.randint(0, 2**31)

    # Determine device
    device = _determine_device(network, device)

    # Determine if we have a non-neural opponent (Agent subclass)
    has_agent_opponent = opponent is not None and hasattr(opponent, "select_action")

    # Determine the network for Player 1
    # Priority: agent opponent > opponent_network > same network
    p1_network = network  # Default: same network
    if opponent_network is not None and not has_agent_opponent:
        p1_network = opponent_network
        p1_network.eval()

    # Initialize game
    game_state = simulate.new_game(seed)

    # Trajectory storage
    steps_p0: list[_PendingStep] = []
    steps_p1: list[_PendingStep] = []

    # Play until terminal - use inference_mode context manager
    with inference_mode(network):
        while not simulate.is_terminal(game_state):
            player = game_state.active_player

            # Handle Player 1 with agent opponent (random, heuristic, etc.)
            if player == 1 and has_agent_opponent:
                game_state = _handle_agent_opponent_turn(
                    game_state, player, opponent, steps_p1
                )
                continue

            # Neural network player (P0 always, P1 when no agent opponent)
            current_network = network if player == 0 else p1_network
            game_state, pending = _handle_neural_player_turn(
                game_state, player, current_network, device, temperature
            )

            if player == 0:
                steps_p0.append(pending)
            else:
                steps_p1.append(pending)

    # Get final scores
    scores = simulate.score(game_state)
    final_scores = (scores[0], scores[1])

    # Determine winner and rewards
    winner = compute_winner(final_scores)
    reward_p0, reward_p1 = compute_rewards(winner, final_scores, shaped=shaped_rewards)

    # Convert pending steps to final trajectory steps with rewards
    trajectory_p0 = _finalize_trajectory(steps_p0, reward_p0)
    trajectory_p1 = _finalize_trajectory(steps_p1, reward_p1)

    return GameTrajectory(
        steps_p0=trajectory_p0,
        steps_p1=trajectory_p1,
        winner=winner,
        final_scores=final_scores,
        game_length=len(steps_p0) + len(steps_p1),
        seed=seed,
    )


def _finalize_trajectory(
    pending_steps: list[_PendingStep],
    final_reward: float,
) -> list[TrajectoryStep]:
    """Convert pending steps to final trajectory steps with rewards.

    Only the last step receives the game outcome reward.
    All other steps get zero reward (rewards are propagated via GAE later).

    Args:
        pending_steps: List of pending steps from game play.
        final_reward: Reward for the final step (game outcome).

    Returns:
        List of finalized TrajectoryStep objects.
    """
    if not pending_steps:
        return []

    result = []
    for i, pending in enumerate(pending_steps):
        is_last = i == len(pending_steps) - 1
        step = TrajectoryStep(
            observation=pending.observation,
            action_mask=pending.action_mask,
            action=pending.action,
            action_prob=pending.action_prob,
            value=pending.value,
            reward=final_reward if is_last else 0.0,
            done=is_last,
        )
        result.append(step)

    return result


# ============================================================================
# Batch Game Generation
# ============================================================================


def _init_worker(network_state_dict: dict[str, Any], network_class: type, network_config: Any) -> None:
    """Initialize worker process with network."""
    global _worker_network, _worker_device

    # Create network on CPU for workers
    _worker_device = torch.device("cpu")
    _worker_network = network_class(network_config)
    _worker_network.load_state_dict(network_state_dict)
    _worker_network.to(_worker_device)
    _worker_network.eval()


def _play_game_worker(args: tuple[int, float, bool]) -> GameTrajectory:
    """Worker function for parallel game generation."""
    seed, temperature, shaped_rewards = args

    global _worker_network, _worker_device
    if _worker_network is None:
        raise RuntimeError("Worker network not initialized")

    return play_game(
        network=_worker_network,
        seed=seed,
        temperature=temperature,
        device=_worker_device,
        shaped_rewards=shaped_rewards,
    )


def _validate_and_generate_seeds(
    num_games: int,
    seeds: list[int] | None,
) -> list[int]:
    """Validate num_games and generate seeds if not provided.

    Args:
        num_games: Number of games to generate.
        seeds: Optional list of seeds.

    Returns:
        List of seeds for each game.

    Raises:
        ValueError: If num_games <= 0 or seeds length doesn't match.
    """
    if num_games <= 0:
        raise ValueError(f"num_games must be positive, got {num_games}")

    if seeds is None:
        base_seed = np.random.randint(0, 2**31)
        return [base_seed + i for i in range(num_games)]

    if len(seeds) != num_games:
        raise ValueError(f"seeds length ({len(seeds)}) must match num_games ({num_games})")
    return seeds


def _generate_games_sequential(
    network: nn.Module,
    seeds: list[int],
    temperature: float,
    device: torch.device | None,
    shaped_rewards: bool,
    opponent: Agent | None = None,
    opponent_network: nn.Module | None = None,
) -> list[GameTrajectory]:
    """Generate games sequentially (legacy fallback).

    Args:
        network: Neural network for policy/value estimation.
        seeds: List of seeds for each game.
        temperature: Temperature for action sampling.
        device: Device for network inference.
        shaped_rewards: If True, use score-margin shaped rewards.
        opponent: Optional Agent opponent for Player 1.
        opponent_network: Optional neural network for Player 1.

    Returns:
        List of GameTrajectory objects.
    """
    trajectories = []
    for seed in seeds:
        trajectory = play_game(
            network=network,
            seed=seed,
            temperature=temperature,
            device=device,
            shaped_rewards=shaped_rewards,
            opponent=opponent,
            opponent_network=opponent_network,
        )
        trajectories.append(trajectory)
    return trajectories


def generate_games(
    network: nn.Module,
    num_games: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
    num_workers: int = 1,
    seeds: list[int] | None = None,
    shaped_rewards: bool = False,
    use_vectorized: bool = True,
    opponent: Agent | None = None,
    opponent_network: nn.Module | None = None,
) -> list[GameTrajectory]:
    """Generate multiple self-play games.

    By default, uses vectorized environment for batched GPU inference.
    This dramatically improves GPU utilization by batching observations
    from all active games into single inference calls.

    For opponent diversity training, provide an opponent or opponent_network
    parameter. The vectorized environment now supports mixed opponent types,
    preserving most GPU efficiency even when playing against non-neural opponents.

    Args:
        network: Neural network for policy/value estimation (Player 0).
        num_games: Number of games to generate.
        temperature: Temperature for action sampling.
        device: Device for network inference.
        num_workers: Number of parallel workers (only used if use_vectorized=False).
        seeds: List of seeds for each game. If None, generates random seeds.
        shaped_rewards: If True, use score-margin shaped rewards.
        use_vectorized: If True (default), use vectorized env for GPU efficiency.
                       Set to False to use legacy sequential generation.
        opponent: Optional Agent opponent for Player 1 (random, heuristic, etc.).
        opponent_network: Optional neural network for Player 1.

    Returns:
        List of GameTrajectory objects, one per game.
    """
    seeds = _validate_and_generate_seeds(num_games, seeds)
    has_opponent = opponent is not None or opponent_network is not None

    # Use vectorized environment for GPU efficiency (default)
    if use_vectorized:
        if has_opponent:
            return generate_games_batched_with_opponent(
                network=network,
                num_games=num_games,
                temperature=temperature,
                device=device,
                seeds=seeds,
                shaped_rewards=shaped_rewards,
                opponent=opponent,
                opponent_network=opponent_network,
            )
        return generate_games_batched(
            network=network,
            num_games=num_games,
            temperature=temperature,
            device=device,
            seeds=seeds,
            shaped_rewards=shaped_rewards,
        )

    # Sequential generation (legacy fallback)
    if num_workers <= 1 or has_opponent:
        return _generate_games_sequential(
            network=network,
            seeds=seeds,
            temperature=temperature,
            device=device,
            shaped_rewards=shaped_rewards,
            opponent=opponent,
            opponent_network=opponent_network,
        )

    # Parallel generation (legacy, uses CPU multiprocessing)
    # Note: Does not support opponent diversity
    try:
        return _generate_games_parallel(
            network=network,
            seeds=seeds,
            temperature=temperature,
            num_workers=num_workers,
            shaped_rewards=shaped_rewards,
        )
    except Exception as e:
        logger.warning(f"Parallel generation failed: {e}. Falling back to sequential.")
        return _generate_games_sequential(
            network=network,
            seeds=seeds,
            temperature=temperature,
            device=device,
            shaped_rewards=shaped_rewards,
        )


def generate_games_batched(
    network: nn.Module,
    num_games: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
    seeds: list[int] | None = None,
    shaped_rewards: bool = False,
) -> list[GameTrajectory]:
    """Generate self-play games using vectorized environment for batched inference.

    This is the GPU-optimized version that batches inference across all active games.
    Instead of running games sequentially with single-sample inference:
        512 games x 20 steps x batch_size=1 = 10,240 tiny GPU calls

    This runs all games in parallel and batches inference:
        ~20 steps x batch_size=512 = 20 large GPU calls

    This dramatically improves GPU utilization from ~1-18% to ~80-95%.

    Args:
        network: Neural network for policy/value estimation.
        num_games: Number of games to generate.
        temperature: Temperature for action sampling.
        device: Device for network inference.
        seeds: Optional list of seeds for each game.
        shaped_rewards: If True, use score-margin shaped rewards.

    Returns:
        List of GameTrajectory objects, one per game.
    """
    # Try Cython-accelerated environment first (5-10x faster)
    try:
        from .vectorized_env_cython import (
            generate_games_vectorized_cython,
            is_cython_available,
        )

        if is_cython_available():
            env_trajectories, stats = generate_games_vectorized_cython(
                network=network,
                num_games=num_games,
                temperature=temperature,
                device=device,
                seeds=seeds,
                shaped_rewards=shaped_rewards,
            )
            logger.debug(
                f"Cython vectorized generation: {stats['inference_calls']} inference calls, "
                f"avg batch size {stats['avg_batch_size']:.1f}"
            )
            return _convert_env_trajectories(env_trajectories, shaped_rewards)
    except ImportError:
        pass

    # Fall back to pure Python vectorized environment
    from .vectorized_env import generate_games_vectorized

    env_trajectories, stats = generate_games_vectorized(
        network=network,
        num_games=num_games,
        temperature=temperature,
        device=device,
        seeds=seeds,
        shaped_rewards=shaped_rewards,
    )

    logger.debug(
        f"Vectorized generation: {stats['inference_calls']} inference calls, "
        f"avg batch size {stats['avg_batch_size']:.1f}"
    )

    # Convert EnvTrajectory to GameTrajectory format
    return _convert_env_trajectories(env_trajectories, shaped_rewards)


def generate_games_batched_with_opponent(
    network: nn.Module,
    num_games: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
    seeds: list[int] | None = None,
    shaped_rewards: bool = False,
    opponent: Agent | None = None,
    opponent_network: nn.Module | None = None,
) -> list[GameTrajectory]:
    """Generate games with opponent diversity using vectorized environment.

    This version supports playing against non-neural opponents (random, heuristic)
    while still preserving most of the GPU efficiency benefits. Player 0 (training
    player) uses batched network inference, while player 1 can use a simple agent.

    Key insight: Even with sequential opponent moves, we still get massive speedup:
    - Network inference (player 0) is batched across all games
    - Only ~50% of moves are opponent moves
    - Opponent moves (heuristic/random) are computationally cheap

    Args:
        network: Neural network for player 0 (training player).
        num_games: Number of games to generate.
        temperature: Temperature for action sampling.
        device: Device for network inference.
        seeds: Optional list of seeds for each game.
        shaped_rewards: If True, use score-margin shaped rewards.
        opponent: Optional Agent for player 1 (RandomAgent, HeuristicAgent).
        opponent_network: Optional neural network for player 1 (checkpoint).
            Ignored if opponent is provided.

    Returns:
        List of GameTrajectory objects, one per game.
    """
    # Try Cython-accelerated environment first (5-10x faster)
    try:
        from .vectorized_env_cython import (
            generate_games_vectorized_cython_with_opponent,
            is_cython_available,
        )

        if is_cython_available():
            env_trajectories, stats = generate_games_vectorized_cython_with_opponent(
                network=network,
                opponent=opponent,
                opponent_network=opponent_network,
                num_games=num_games,
                temperature=temperature,
                device=device,
                seeds=seeds,
                shaped_rewards=shaped_rewards,
            )
            p0_calls = stats.get("p0_inference_calls", 0)
            p1_calls = stats.get("p1_inference_calls", 0)
            p1_agent = stats.get("p1_agent_steps", 0)

            if p1_agent > 0:
                logger.debug(
                    f"Cython vectorized generation with agent opponent: "
                    f"{p0_calls:.0f} P0 inference calls, {p1_agent:.0f} agent steps, "
                    f"avg batch size {stats['avg_batch_size']:.1f}"
                )
            else:
                logger.debug(
                    f"Cython vectorized generation with network opponent: "
                    f"{p0_calls:.0f} P0 calls, {p1_calls:.0f} P1 calls, "
                    f"avg batch size {stats['avg_batch_size']:.1f}"
                )
            return _convert_env_trajectories(env_trajectories, shaped_rewards)
    except ImportError:
        pass

    # Fall back to pure Python vectorized environment
    from .vectorized_env import generate_games_vectorized_with_opponent

    # Generate games using vectorized environment with opponent support
    env_trajectories, stats = generate_games_vectorized_with_opponent(
        network=network,
        opponent=opponent,
        opponent_network=opponent_network,
        num_games=num_games,
        temperature=temperature,
        device=device,
        seeds=seeds,
        shaped_rewards=shaped_rewards,
    )

    # Log stats
    p0_calls = stats.get("p0_inference_calls", 0)
    p1_calls = stats.get("p1_inference_calls", 0)
    p1_agent = stats.get("p1_agent_steps", 0)

    if p1_agent > 0:
        logger.debug(
            f"Vectorized generation with agent opponent: "
            f"{p0_calls:.0f} P0 inference calls, {p1_agent:.0f} agent steps, "
            f"avg batch size {stats['avg_batch_size']:.1f}"
        )
    else:
        logger.debug(
            f"Vectorized generation with network opponent: "
            f"{p0_calls:.0f} P0 calls, {p1_calls:.0f} P1 calls, "
            f"avg batch size {stats['avg_batch_size']:.1f}"
        )

    # Convert EnvTrajectory to GameTrajectory format
    return _convert_env_trajectories(env_trajectories, shaped_rewards)


def _convert_pending_steps_to_trajectory(
    pending_steps: list,
    final_reward: float,
) -> list[TrajectoryStep]:
    """Convert pending steps to trajectory steps with terminal reward.

    Args:
        pending_steps: List of pending step objects with observation/action data.
        final_reward: Reward to assign to the final step.

    Returns:
        List of TrajectoryStep objects with rewards assigned.
    """
    result = []
    for i, step in enumerate(pending_steps):
        is_last = i == len(pending_steps) - 1
        trajectory_step = TrajectoryStep(
            observation=step.observation,
            action_mask=step.action_mask,
            action=step.action,
            action_prob=step.action_prob,
            value=step.value,
            reward=final_reward if is_last else 0.0,
            done=is_last,
        )
        result.append(trajectory_step)
    return result


def _convert_single_env_trajectory(
    env_traj,
    shaped_rewards: bool,
) -> GameTrajectory | None:
    """Convert a single EnvTrajectory to GameTrajectory.

    Args:
        env_traj: EnvTrajectory from vectorized environment (Python or Cython).
        shaped_rewards: Whether to use shaped rewards.

    Returns:
        GameTrajectory or None if conversion not possible.
    """
    # Support both Python EnvTrajectory (has final_state) and Cython (has final_scores)
    if hasattr(env_traj, "final_state"):
        final_state = env_traj.final_state
        if final_state is None:
            return None
        scores = simulate.score(final_state)
        final_scores = (scores[0], scores[1])
    elif hasattr(env_traj, "final_scores") and env_traj.final_scores is not None:
        final_scores = env_traj.final_scores
    else:
        return None

    # Compute game outcome
    winner = compute_winner(final_scores)
    reward_p0, reward_p1 = compute_rewards(winner, final_scores, shaped=shaped_rewards)

    # Convert steps for both players
    steps_p0 = _convert_pending_steps_to_trajectory(env_traj.steps_p0, reward_p0)
    steps_p1 = _convert_pending_steps_to_trajectory(env_traj.steps_p1, reward_p1)

    return GameTrajectory(
        steps_p0=steps_p0,
        steps_p1=steps_p1,
        winner=winner,
        final_scores=final_scores,
        game_length=len(steps_p0) + len(steps_p1),
        seed=env_traj.seed,
    )


def _convert_env_trajectories(
    env_trajectories: list,
    shaped_rewards: bool = False,
) -> list[GameTrajectory]:
    """Convert vectorized env trajectories to GameTrajectory format.

    Args:
        env_trajectories: List of EnvTrajectory objects from vectorized env
            (either Python or Cython version).
        shaped_rewards: Whether shaped rewards were used.

    Returns:
        List of GameTrajectory objects compatible with existing code.
    """
    result = []
    for env_traj in env_trajectories:
        # Check for required attributes (duck typing) instead of isinstance
        # to support both Python EnvTrajectory and Cython EnvTrajectory
        if not hasattr(env_traj, "steps_p0") or not hasattr(env_traj, "steps_p1"):
            continue

        game_traj = _convert_single_env_trajectory(env_traj, shaped_rewards)
        if game_traj is not None:
            result.append(game_traj)

    return result


def _generate_games_parallel(
    network: nn.Module,
    seeds: list[int],
    temperature: float,
    num_workers: int,
    shaped_rewards: bool,
) -> list[GameTrajectory]:
    """Generate games using multiprocessing pool."""
    from _02_agents.neural.network import BeastyBarNetwork
    from _02_agents.neural.utils import NetworkConfig

    # Get network config and state dict
    state_dict = network.state_dict()

    # Try to get config from network
    if hasattr(network, "config"):
        config = network.config
    else:
        # Fallback: infer config from state dict
        hidden_dim = state_dict["fc1.weight"].shape[0]
        config = NetworkConfig(hidden_dim=hidden_dim)

    # Prepare worker arguments
    worker_args = [(seed, temperature, shaped_rewards) for seed in seeds]

    # Use fork method - faster startup, safe since workers use CPU only
    # (spawn is too slow due to reimporting PyTorch in each worker)
    ctx = mp.get_context("fork")

    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(state_dict, BeastyBarNetwork, config),
    ) as pool:
        trajectories = pool.map(_play_game_worker, worker_args)

    return trajectories


def generate_seed_sequence(base_seed: int, count: int) -> list[int]:
    """Generate a deterministic sequence of seeds from a base seed.

    Useful for reproducible game generation across training runs.

    Args:
        base_seed: Starting seed for the sequence.
        count: Number of seeds to generate.

    Returns:
        List of deterministic seeds derived from base_seed.

    Example:
        >>> seeds = generate_seed_sequence(42, 100)
        >>> # seeds will be identical across runs
    """
    rng = np.random.RandomState(base_seed)
    return [int(rng.randint(0, 2**31)) for _ in range(count)]


# ============================================================================
# Trajectory to Transition Conversion
# ============================================================================


def trajectories_to_transitions(
    trajectories: list[GameTrajectory],
    player: int | None = None,
) -> list[Transition]:
    """Convert game trajectories into individual transitions for replay buffer.

    Flattens trajectory steps from specified player(s) and games into a single
    list of Transition objects suitable for adding to a ReplayBuffer.

    When using opponent diversity training, set player=0 to only collect
    transitions from the learning agent (Player 0), excluding opponent moves.

    Args:
        trajectories: List of game trajectories.
        player: Player to collect transitions from (0, 1, or None for both).
                When using opponent diversity, set to 0 to only collect
                transitions from the learning agent.

    Returns:
        List of Transition objects for the specified player(s).

    Example:
        >>> # Standard self-play: collect from both players
        >>> trajectories = generate_games(network, num_games=10)
        >>> transitions = trajectories_to_transitions(trajectories)
        >>> buffer.add_batch(transitions)

        >>> # Opponent diversity: collect only Player 0 (learning agent)
        >>> trajectories = generate_games(network, opponent=random_agent)
        >>> transitions = trajectories_to_transitions(trajectories, player=0)
        >>> buffer.add_batch(transitions)
    """
    transitions = []

    for trajectory in trajectories:
        # Process player 0 steps
        if player is None or player == 0:
            for step in trajectory.steps_p0:
                transition = Transition(
                    observation=step.observation,
                    action_mask=step.action_mask,
                    action=step.action,
                    action_prob=step.action_prob,
                    value=step.value,
                    reward=step.reward,
                    done=step.done,
                )
                transitions.append(transition)

        # Process player 1 steps
        if player is None or player == 1:
            for step in trajectory.steps_p1:
                transition = Transition(
                    observation=step.observation,
                    action_mask=step.action_mask,
                    action=step.action,
                    action_prob=step.action_prob,
                    value=step.value,
                    reward=step.reward,
                    done=step.done,
                )
                transitions.append(transition)

    return transitions


def trajectory_to_player_transitions(
    trajectory: GameTrajectory,
    player: int,
) -> list[Transition]:
    """Convert a single player's trajectory to transitions.

    Args:
        trajectory: Game trajectory.
        player: Player index (0 or 1).

    Returns:
        List of Transition objects for the specified player.
    """
    steps = trajectory.get_steps(player)
    return [
        Transition(
            observation=step.observation,
            action_mask=step.action_mask,
            action=step.action,
            action_prob=step.action_prob,
            value=step.value,
            reward=step.reward,
            done=step.done,
        )
        for step in steps
    ]


# ============================================================================
# Statistics Tracking
# ============================================================================


@dataclass(frozen=True, slots=True)
class SelfPlayStats:
    """Statistics from a batch of self-play games.

    Attributes:
        games_played: Total number of games.
        p0_wins: Number of wins by player 0.
        p1_wins: Number of wins by player 1.
        draws: Number of draws.
        avg_game_length: Average number of turns per game.
        avg_action_entropy: Average entropy of action distributions.
        total_steps: Total trajectory steps across all games.
        avg_value: Average value estimate across all steps.
    """

    games_played: int
    p0_wins: int
    p1_wins: int
    draws: int
    avg_game_length: float
    avg_action_entropy: float
    total_steps: int
    avg_value: float

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

    def to_dict(self) -> dict[str, float | int]:
        """Convert stats to dictionary for logging."""
        return {
            "games_played": self.games_played,
            "p0_wins": self.p0_wins,
            "p1_wins": self.p1_wins,
            "draws": self.draws,
            "p0_win_rate": self.p0_win_rate,
            "p1_win_rate": self.p1_win_rate,
            "draw_rate": self.draw_rate,
            "avg_game_length": self.avg_game_length,
            "avg_action_entropy": self.avg_action_entropy,
            "total_steps": self.total_steps,
            "avg_value": self.avg_value,
        }


def _collect_step_statistics(
    trajectories: list[GameTrajectory],
) -> tuple[list[float], list[float], int]:
    """Collect action probabilities and values from all trajectory steps.

    Args:
        trajectories: List of game trajectories.

    Returns:
        Tuple of (action_probs, values, total_steps).
    """
    all_probs = []
    all_values = []
    total_steps = 0

    for trajectory in trajectories:
        for step in trajectory.steps_p0:
            all_probs.append(step.action_prob)
            all_values.append(step.value)
            total_steps += 1
        for step in trajectory.steps_p1:
            all_probs.append(step.action_prob)
            all_values.append(step.value)
            total_steps += 1

    return all_probs, all_values, total_steps


def _compute_action_entropy(probs: list[float]) -> float:
    """Compute average entropy proxy from action probabilities.

    Uses -log(p) as proxy for uncertainty.

    Args:
        probs: List of action probabilities.

    Returns:
        Average entropy value.
    """
    if not probs:
        return 0.0

    probs_array = np.array(probs)
    probs_array = np.clip(probs_array, 1e-10, 1.0)
    return float(-np.mean(np.log(probs_array)))


def compute_stats(trajectories: list[GameTrajectory]) -> SelfPlayStats:
    """Compute statistics from a batch of game trajectories.

    Args:
        trajectories: List of completed game trajectories.

    Returns:
        SelfPlayStats with aggregated statistics.

    Example:
        >>> trajectories = generate_games(network, num_games=100)
        >>> stats = compute_stats(trajectories)
        >>> print(f"P0 win rate: {stats.p0_win_rate:.1%}")
    """
    if not trajectories:
        return SelfPlayStats(
            games_played=0,
            p0_wins=0,
            p1_wins=0,
            draws=0,
            avg_game_length=0.0,
            avg_action_entropy=0.0,
            total_steps=0,
            avg_value=0.0,
        )

    games_played = len(trajectories)
    p0_wins = sum(1 for t in trajectories if t.winner == 0)
    p1_wins = sum(1 for t in trajectories if t.winner == 1)
    draws = sum(1 for t in trajectories if t.winner is None)

    total_length = sum(t.game_length for t in trajectories)
    avg_game_length = total_length / games_played

    all_probs, all_values, total_steps = _collect_step_statistics(trajectories)
    avg_action_entropy = _compute_action_entropy(all_probs)
    avg_value = float(np.mean(all_values)) if all_values else 0.0

    return SelfPlayStats(
        games_played=games_played,
        p0_wins=p0_wins,
        p1_wins=p1_wins,
        draws=draws,
        avg_game_length=avg_game_length,
        avg_action_entropy=avg_action_entropy,
        total_steps=total_steps,
        avg_value=avg_value,
    )


def compute_action_distribution(trajectories: list[GameTrajectory]) -> dict[int, int]:
    """Compute action frequency distribution from trajectories.

    Useful for analyzing policy behavior and detecting action biases.

    Args:
        trajectories: List of game trajectories.

    Returns:
        Dictionary mapping action indices to their frequency counts.
    """
    action_counts: dict[int, int] = {}

    for trajectory in trajectories:
        for step in trajectory.steps_p0:
            action_counts[step.action] = action_counts.get(step.action, 0) + 1
        for step in trajectory.steps_p1:
            action_counts[step.action] = action_counts.get(step.action, 0) + 1

    return action_counts


# ============================================================================
# Seeding Utilities
# ============================================================================


def set_self_play_seed(seed: int) -> None:
    """Set random seeds for reproducible self-play.

    Sets seeds for numpy random number generator used in game generation.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)


def create_game_seeds(
    num_games: int,
    base_seed: int | None = None,
) -> list[int]:
    """Create deterministic game seeds for reproducible training.

    If base_seed is provided, generates deterministic seed sequence.
    Otherwise, generates random seeds.

    Args:
        num_games: Number of game seeds to create.
        base_seed: Optional base seed for deterministic sequence.

    Returns:
        List of integer seeds, one per game.
    """
    if base_seed is not None:
        return generate_seed_sequence(base_seed, num_games)
    return [np.random.randint(0, 2**31) for _ in range(num_games)]


__all__ = [
    "GameTrajectory",
    "SelfPlayStats",
    "TrajectoryStep",
    "compute_action_distribution",
    "compute_stats",
    "create_game_seeds",
    "generate_games",
    "generate_games_batched",
    "generate_games_batched_with_opponent",
    "generate_seed_sequence",
    "play_game",
    "set_self_play_seed",
    "trajectories_to_transitions",
    "trajectory_to_player_transitions",
]
