"""Self-play game runner for neural network training.

This module generates training games by having the neural network play
against itself. Both players use the same network, collecting full game
trajectories that can be converted to transitions for replay buffer storage.

Example:
    from _03_training.self_play import play_game, generate_games

    # Run single self-play game
    trajectory = play_game(network, seed=42, temperature=1.0)

    # Generate batch of games
    trajectories = generate_games(network, num_games=256)

    # Convert to transitions for replay buffer
    transitions = trajectories_to_transitions(trajectories)
    buffer.add_batch(transitions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from _01_simulator import simulate
from _01_simulator.action_space import index_to_action, legal_action_mask_tensor
from _01_simulator.observations import state_to_tensor

from .replay_buffer import Transition

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import nn

logger = logging.getLogger(__name__)


# ============================================================================
# Trajectory Data Structures
# ============================================================================


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """Single step in a player's game trajectory.

    Attributes:
        observation: State observation tensor of shape (OBSERVATION_DIM,).
        action_mask: Legal action mask of shape (ACTION_DIM,).
        action: Selected action index.
        action_prob: Probability of selected action under policy.
        value: Value estimate from critic at this state.
        reward: Reward received (set after game ends).
        done: Whether this is the last step of the trajectory.
    """

    observation: NDArray[np.float32]  # (OBSERVATION_DIM,)
    action_mask: NDArray[np.float32]  # (ACTION_DIM,)
    action: int
    action_prob: float
    value: float
    reward: float = 0.0
    done: bool = False


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
# Self-Play Game Runner
# ============================================================================


def play_game(
    network: nn.Module,
    seed: int | None = None,
    temperature: float = 1.0,
    device: torch.device | None = None,
    shaped_rewards: bool = False,
) -> GameTrajectory:
    """Run a single self-play game using the given network.

    Both players use the same network for action selection. The game
    runs until terminal state, then rewards are assigned based on outcome.

    Args:
        network: Neural network for policy/value estimation.
        seed: Random seed for game initialization. If None, uses random seed.
        temperature: Temperature for action sampling. Higher = more random.
        device: Device for network inference. If None, uses network's device.
        shaped_rewards: If True, use score-margin shaped rewards.

    Returns:
        GameTrajectory containing full game data for both players.

    Example:
        >>> trajectory = play_game(network, seed=42, temperature=1.0)
        >>> print(f"Winner: {trajectory.winner}, Length: {trajectory.game_length}")
    """
    # Handle seed
    if seed is None:
        seed = np.random.randint(0, 2**31)

    # Determine device
    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Initialize game
    game_state = simulate.new_game(seed)

    # Trajectory storage
    steps_p0: list[_PendingStep] = []
    steps_p1: list[_PendingStep] = []

    # Set network to eval mode
    network.eval()

    # Play until terminal
    with torch.no_grad():
        while not simulate.is_terminal(game_state):
            player = game_state.active_player

            # Get observation and mask
            obs = state_to_tensor(game_state, player)
            mask = legal_action_mask_tensor(game_state, player)

            # Convert to tensors
            obs_tensor = torch.from_numpy(obs).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)

            # Forward pass
            policy_logits, value = network(obs_tensor, mask_tensor)

            # Sample action
            action_idx, action_prob = _sample_action_with_prob(
                policy_logits, mask_tensor, temperature
            )

            # Store pending step (reward to be filled later)
            pending = _PendingStep(
                observation=obs,
                action_mask=mask,
                action=action_idx,
                action_prob=action_prob,
                value=float(value.squeeze().item()),
            )

            if player == 0:
                steps_p0.append(pending)
            else:
                steps_p1.append(pending)

            # Apply action
            action = index_to_action(action_idx)
            game_state = simulate.apply(game_state, action)

    # Get final scores
    scores = simulate.score(game_state)
    final_scores = (scores[0], scores[1])

    # Determine winner
    if scores[0] > scores[1]:
        winner = 0
    elif scores[1] > scores[0]:
        winner = 1
    else:
        winner = None

    # Assign rewards
    if shaped_rewards:
        # Shaped rewards based on score margin
        margin = abs(scores[0] - scores[1])
        base_reward = min(1.0 + margin * 0.1, 2.0)  # Cap at 2.0

        reward_p0 = base_reward if winner == 0 else (-base_reward if winner == 1 else 0.0)
        reward_p1 = base_reward if winner == 1 else (-base_reward if winner == 0 else 0.0)
    else:
        # Standard rewards: +1 win, -1 loss, 0 draw
        reward_p0 = 1.0 if winner == 0 else (-1.0 if winner == 1 else 0.0)
        reward_p1 = 1.0 if winner == 1 else (-1.0 if winner == 0 else 0.0)

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


@dataclass
class _PendingStep:
    """Temporary storage for trajectory step before reward assignment."""

    observation: NDArray[np.float32]
    action_mask: NDArray[np.float32]
    action: int
    action_prob: float
    value: float


def _sample_action_with_prob(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> tuple[int, float]:
    """Sample action from masked logits and return action index and probability.

    Args:
        logits: Raw policy logits of shape (action_dim,) or (1, action_dim).
        mask: Binary mask of shape (action_dim,) or (1, action_dim).
        temperature: Temperature for softmax scaling.

    Returns:
        Tuple of (action_index, action_probability).
    """
    # Flatten if needed
    logits = logits.squeeze()
    mask = mask.squeeze()

    # Apply mask
    masked_logits = torch.where(mask > 0, logits, torch.tensor(float("-inf"), device=logits.device))

    # Apply temperature and softmax
    scaled_logits = masked_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Sample action
    action_idx = torch.multinomial(probs, num_samples=1).item()

    # Get probability of selected action
    action_prob = probs[action_idx].item()

    return int(action_idx), float(action_prob)


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


def generate_games(
    network: nn.Module,
    num_games: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
    num_workers: int = 1,
    seeds: list[int] | None = None,
    shaped_rewards: bool = False,
) -> list[GameTrajectory]:
    """Generate multiple self-play games.

    Args:
        network: Neural network for policy/value estimation.
        num_games: Number of games to generate.
        temperature: Temperature for action sampling.
        device: Device for network inference.
        num_workers: Number of parallel workers (future use, currently ignored).
        seeds: List of seeds for each game. If None, generates random seeds.
        shaped_rewards: If True, use score-margin shaped rewards.

    Returns:
        List of GameTrajectory objects, one per game.

    Note:
        The num_workers parameter is reserved for future multiprocessing
        implementation. Currently games are generated sequentially.
    """
    if num_games <= 0:
        raise ValueError(f"num_games must be positive, got {num_games}")

    # Generate seeds if not provided
    if seeds is None:
        base_seed = np.random.randint(0, 2**31)
        seeds = [base_seed + i for i in range(num_games)]
    elif len(seeds) != num_games:
        raise ValueError(f"seeds length ({len(seeds)}) must match num_games ({num_games})")

    trajectories = []

    # TODO: Implement parallel generation when num_workers > 1
    # For now, generate sequentially
    if num_workers > 1:
        logger.warning(
            f"num_workers={num_workers} requested but parallel generation not yet implemented. "
            "Running sequentially."
        )

    for seed in seeds:
        trajectory = play_game(
            network=network,
            seed=seed,
            temperature=temperature,
            device=device,
            shaped_rewards=shaped_rewards,
        )
        trajectories.append(trajectory)

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
) -> list[Transition]:
    """Convert game trajectories into individual transitions for replay buffer.

    Flattens all trajectory steps from all players and games into a single
    list of Transition objects suitable for adding to a ReplayBuffer.

    Args:
        trajectories: List of game trajectories.

    Returns:
        List of Transition objects, one per step across all games/players.

    Example:
        >>> trajectories = generate_games(network, num_games=10)
        >>> transitions = trajectories_to_transitions(trajectories)
        >>> buffer.add_batch(transitions)
    """
    transitions = []

    for trajectory in trajectories:
        # Process player 0 steps
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

    # Game length
    total_length = sum(t.game_length for t in trajectories)
    avg_game_length = total_length / games_played

    # Collect all action probabilities for entropy calculation
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

    # Compute average entropy from action probabilities
    # Entropy approximation: -sum(p * log(p))
    # For single action: -p * log(p) - (1-p) * log(1-p) is complex
    # Simpler: use -log(p) as proxy for uncertainty
    if all_probs:
        # Clip to avoid log(0)
        probs_array = np.array(all_probs)
        probs_array = np.clip(probs_array, 1e-10, 1.0)
        avg_action_entropy = -np.mean(np.log(probs_array))
    else:
        avg_action_entropy = 0.0

    # Average value
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
    "generate_seed_sequence",
    "play_game",
    "set_self_play_seed",
    "trajectories_to_transitions",
    "trajectory_to_player_transitions",
]
