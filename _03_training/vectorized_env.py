"""Vectorized game environment for batched self-play inference.

This module implements a vectorized environment that runs N games simultaneously,
enabling batched neural network inference for dramatically improved GPU utilization.

Instead of:
    512 games x 20 steps x batch_size=1 = 10,240 tiny GPU calls

We get:
    ~20 steps x batch_size=512 = 20 large GPU calls

This is the key to efficient GPU utilization during self-play.

Note: The game simulation is pure Python (GIL-bound), so CPU parallelization
via threading doesn't help. For further speedup, the simulator would need to
be rewritten in Cython/Numba/Rust, or use multiprocessing.

Example:
    env = VectorizedGameEnv(num_envs=512, device=device)
    env.reset(seeds)

    while not env.all_done():
        obs, masks = env.get_observations_and_masks()
        policy, values = network(obs, masks)
        actions, probs = sample_batch(policy, masks, temperature)
        env.step(actions, probs, values)

    trajectories = env.get_trajectories()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from _01_simulator import simulate
from _01_simulator.action_space import (
    ACTION_DIM,
    action_index,
    index_to_action,
    legal_action_mask_tensor,
)
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor
from _03_training.game_utils import compute_rewards, compute_winner
from _03_training.trajectory import PendingStep
from _03_training.utils import inference_mode

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import nn

    from _01_simulator.state import State
    from _02_agents.base import Agent


@dataclass
class EnvTrajectory:
    """Complete trajectory data from a single game in vectorized env.

    Stores steps for both players separately for proper GAE computation.
    """

    steps_p0: list[PendingStep] = field(default_factory=list)
    steps_p1: list[PendingStep] = field(default_factory=list)
    seed: int = 0
    final_state: State | None = None

    def add_step(self, step: PendingStep) -> None:
        """Add step to appropriate player's trajectory."""
        if step.player == 0:
            self.steps_p0.append(step)
        else:
            self.steps_p1.append(step)


class VectorizedGameEnv:
    """Vectorized environment running N games simultaneously for batched inference.

    This enables efficient GPU utilization by batching observations from all
    active games into a single tensor for inference, then distributing actions
    back to each game.

    Attributes:
        num_envs: Number of parallel game environments.
        device: PyTorch device for tensor operations.
        games: List of game states, one per environment.
        active: Boolean array indicating which games are still running.
        trajectories: Trajectory data for each game.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | None = None,
    ) -> None:
        """Initialize vectorized environment.

        Args:
            num_envs: Number of parallel game environments.
            device: PyTorch device for inference tensors.
        """
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")

        # Game states and status
        self.games: list[State | None] = [None] * num_envs
        self.active = np.zeros(num_envs, dtype=bool)

        # Trajectory storage
        self.trajectories: list[EnvTrajectory] = [EnvTrajectory() for _ in range(num_envs)]

        # Pre-allocated arrays for efficiency
        self._obs_buffer = np.zeros((num_envs, OBSERVATION_DIM), dtype=np.float32)
        self._mask_buffer = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)
        self._player_buffer = np.zeros(num_envs, dtype=np.int32)

    def reset(self, seeds: list[int] | None = None) -> None:
        """Reset all environments with new games.

        Args:
            seeds: Optional list of seeds for each environment.
                   If None, generates random seeds.
        """
        if seeds is None:
            base_seed = np.random.randint(0, 2**31)
            seeds = [base_seed + i for i in range(self.num_envs)]
        elif len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} seeds, got {len(seeds)}")

        for i, seed in enumerate(seeds):
            self.games[i] = simulate.new_game(seed)
            self.active[i] = True
            self.trajectories[i] = EnvTrajectory(seed=seed)

    def num_active(self) -> int:
        """Return number of currently active (non-terminal) games."""
        return int(self.active.sum())

    def all_done(self) -> bool:
        """Check if all games have reached terminal state."""
        return not self.active.any()

    def get_active_indices(self) -> NDArray[np.int64]:
        """Return indices of active (non-terminal) games."""
        return np.where(self.active)[0]

    def get_observations_and_masks(self) -> tuple[torch.Tensor, torch.Tensor, NDArray[np.int64]]:
        """Get batched observations and action masks for all active games.

        Returns:
            Tuple of:
                - observations: (num_active, OBSERVATION_DIM) tensor on device
                - masks: (num_active, ACTION_DIM) tensor on device
                - active_indices: (num_active,) numpy array of env indices
        """
        active_indices = self.get_active_indices()
        num_active = len(active_indices)

        if num_active == 0:
            return (
                torch.empty((0, OBSERVATION_DIM), device=self.device),
                torch.empty((0, ACTION_DIM), device=self.device),
                active_indices,
            )

        # Build observations and masks for active games
        obs_batch = np.zeros((num_active, OBSERVATION_DIM), dtype=np.float32)
        mask_batch = np.zeros((num_active, ACTION_DIM), dtype=np.float32)
        players = np.zeros(num_active, dtype=np.int32)

        for batch_idx, env_idx in enumerate(active_indices):
            game = self.games[env_idx]
            if game is None:
                continue

            player = game.active_player
            players[batch_idx] = player

            # Get observation and mask from current player's perspective
            obs_batch[batch_idx] = state_to_tensor(game, player)
            mask_batch[batch_idx] = legal_action_mask_tensor(game, player)

        # Store player info for step()
        self._active_players = players
        self._active_indices = active_indices

        # Convert to tensors on device
        obs_tensor = torch.from_numpy(obs_batch).to(self.device)
        mask_tensor = torch.from_numpy(mask_batch).to(self.device)

        return obs_tensor, mask_tensor, active_indices

    def step(
        self,
        actions: torch.Tensor,
        action_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> int:
        """Apply actions to all active games and collect trajectory data.

        Args:
            actions: (num_active,) tensor of action indices.
            action_probs: (num_active,) tensor of action probabilities.
            values: (num_active,) tensor of value estimates.

        Returns:
            Number of games that finished this step.
        """
        # Convert to numpy for game logic
        actions_np = actions.cpu().numpy()
        probs_np = action_probs.cpu().numpy()
        values_np = values.cpu().numpy()

        active_indices = self._active_indices
        players = self._active_players
        games_finished = 0

        for batch_idx, env_idx in enumerate(active_indices):
            game = self.games[env_idx]
            if game is None:
                continue

            player = players[batch_idx]
            action_idx = int(actions_np[batch_idx])
            action_prob = float(probs_np[batch_idx])
            value = float(values_np[batch_idx])

            # Get observation and mask that were used for this decision
            obs = state_to_tensor(game, player)
            mask = legal_action_mask_tensor(game, player)

            # Store trajectory step
            pending = PendingStep(
                observation=obs,
                action_mask=mask,
                action=action_idx,
                action_prob=action_prob,
                value=value,
                player=player,
            )
            self.trajectories[env_idx].add_step(pending)

            # Apply action to game
            action = index_to_action(action_idx)
            new_state = simulate.apply(game, action)
            self.games[env_idx] = new_state

            # Check for terminal state
            if simulate.is_terminal(new_state):
                self.active[env_idx] = False
                self.trajectories[env_idx].final_state = new_state
                games_finished += 1

        return games_finished

    def get_trajectories(self, shaped_rewards: bool = False) -> list[EnvTrajectory]:
        """Get all trajectories with rewards assigned.

        Args:
            shaped_rewards: If True, use score-margin shaped rewards.

        Returns:
            List of EnvTrajectory objects with rewards assigned.
        """
        for env_idx in range(self.num_envs):
            traj = self.trajectories[env_idx]
            final_state = traj.final_state

            if final_state is None:
                continue

            # Get final scores
            scores = simulate.score(final_state)
            final_scores = (scores[0], scores[1])

            # Determine winner and rewards
            winner = compute_winner(final_scores)
            reward_p0, reward_p1 = compute_rewards(winner, final_scores, shaped=shaped_rewards)

            # Assign rewards to last steps
            if traj.steps_p0:
                traj.steps_p0[-1] = PendingStep(
                    observation=traj.steps_p0[-1].observation,
                    action_mask=traj.steps_p0[-1].action_mask,
                    action=traj.steps_p0[-1].action,
                    action_prob=traj.steps_p0[-1].action_prob,
                    value=traj.steps_p0[-1].value,
                    player=0,
                )
                # Store reward in a way we can retrieve later
                traj.steps_p0[-1]._reward = reward_p0  # type: ignore[attr-defined]

            if traj.steps_p1:
                traj.steps_p1[-1] = PendingStep(
                    observation=traj.steps_p1[-1].observation,
                    action_mask=traj.steps_p1[-1].action_mask,
                    action=traj.steps_p1[-1].action,
                    action_prob=traj.steps_p1[-1].action_prob,
                    value=traj.steps_p1[-1].value,
                    player=1,
                )
                traj.steps_p1[-1]._reward = reward_p1  # type: ignore[attr-defined]

        return self.trajectories


def sample_actions_batch(
    logits: torch.Tensor,
    masks: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample actions from batched policy logits with temperature.

    Args:
        logits: (batch, ACTION_DIM) raw policy logits.
        masks: (batch, ACTION_DIM) legal action masks (1=legal, 0=illegal).
        temperature: Temperature for sampling. Higher = more random.

    Returns:
        Tuple of:
            - actions: (batch,) sampled action indices.
            - probs: (batch,) probability of each sampled action.
    """
    # Apply mask (set illegal actions to -inf)
    masked_logits = torch.where(
        masks > 0,
        logits,
        torch.tensor(float("-inf"), device=logits.device),
    )

    # Apply temperature and softmax
    scaled_logits = masked_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Handle NaN/Inf from unstable values - fall back to uniform over valid
    invalid_mask = torch.isnan(probs).any(dim=-1) | torch.isinf(probs).any(dim=-1)
    if invalid_mask.any():
        uniform_probs = masks / masks.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs = torch.where(invalid_mask.unsqueeze(-1), uniform_probs, probs)

    # Sample actions
    actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Get probability of each sampled action
    action_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    return actions, action_probs


def generate_games_vectorized(
    network: nn.Module,
    num_games: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
    seeds: list[int] | None = None,
    shaped_rewards: bool = False,
) -> tuple[list[EnvTrajectory], dict[str, float]]:
    """Generate self-play games using vectorized environment for batched inference.

    This is the main entry point for efficient self-play game generation.
    Instead of running games sequentially with single-sample inference,
    this runs all games in parallel and batches inference calls.

    Args:
        network: Neural network for policy/value estimation.
        num_games: Number of games to generate.
        temperature: Temperature for action sampling.
        device: Device for network inference.
        seeds: Optional list of seeds for each game.
        shaped_rewards: If True, use score-margin shaped rewards.

    Returns:
        Tuple of:
            - trajectories: List of EnvTrajectory objects.
            - stats: Dictionary of generation statistics.
    """
    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Create vectorized environment
    env = VectorizedGameEnv(num_envs=num_games, device=device)

    # Reset with seeds
    if seeds is None:
        base_seed = np.random.randint(0, 2**31)
        seeds = [base_seed + i for i in range(num_games)]
    env.reset(seeds)

    # Stats tracking
    total_steps = 0
    inference_calls = 0

    # Run until all games complete - use inference_mode context manager
    with inference_mode(network):
        while not env.all_done():
            # Get batched observations and masks
            obs, masks, _ = env.get_observations_and_masks()

            if obs.size(0) == 0:
                break

            # Batched inference (the key optimization!)
            policy_logits, values = network(obs, masks)
            values = values.squeeze(-1)
            inference_calls += 1

            # Sample actions for all active games
            actions, action_probs = sample_actions_batch(policy_logits, masks, temperature)

            # Step all games
            env.step(actions, action_probs, values)
            total_steps += len(actions)

    # Get trajectories with rewards
    trajectories = env.get_trajectories(shaped_rewards=shaped_rewards)

    stats = {
        "total_steps": total_steps,
        "inference_calls": inference_calls,
        "avg_batch_size": total_steps / max(inference_calls, 1),
        "games_generated": num_games,
    }

    return trajectories, stats


class VectorizedGameEnvWithOpponents:
    """Vectorized environment with opponent diversity support.

    Runs N games simultaneously where:
    - Player 0 (training player) uses batched neural network inference
    - Player 1 (opponent) can be a simple agent (random/heuristic) or another network

    This preserves most of the GPU efficiency benefits even when playing against
    non-neural opponents, because:
    - Network inference for player 0 is still batched across all games
    - Only ~50% of moves are opponent moves
    - Agent opponents (random/heuristic) are computationally cheap

    When opponent_network is provided, both players use batched inference.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | None = None,
    ) -> None:
        """Initialize vectorized environment with opponent support.

        Args:
            num_envs: Number of parallel game environments.
            device: PyTorch device for inference tensors.
        """
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")

        # Game states and status
        self.games: list[State | None] = [None] * num_envs
        self.active = np.zeros(num_envs, dtype=bool)

        # Trajectory storage
        self.trajectories: list[EnvTrajectory] = [EnvTrajectory() for _ in range(num_envs)]

        # Pre-allocated arrays for efficiency
        self._obs_buffer = np.zeros((num_envs, OBSERVATION_DIM), dtype=np.float32)
        self._mask_buffer = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)
        self._player_buffer = np.zeros(num_envs, dtype=np.int32)

    def reset(self, seeds: list[int] | None = None) -> None:
        """Reset all environments with new games.

        Args:
            seeds: Optional list of seeds for each environment.
                   If None, generates random seeds.
        """
        if seeds is None:
            base_seed = np.random.randint(0, 2**31)
            seeds = [base_seed + i for i in range(self.num_envs)]
        elif len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} seeds, got {len(seeds)}")

        for i, seed in enumerate(seeds):
            self.games[i] = simulate.new_game(seed)
            self.active[i] = True
            self.trajectories[i] = EnvTrajectory(seed=seed)

    def num_active(self) -> int:
        """Return number of currently active (non-terminal) games."""
        return int(self.active.sum())

    def all_done(self) -> bool:
        """Check if all games have reached terminal state."""
        return not self.active.any()

    def get_active_indices(self) -> NDArray[np.int64]:
        """Return indices of active (non-terminal) games."""
        return np.where(self.active)[0]

    def get_player0_games(self) -> list[int]:
        """Get environment indices of active games where it's player 0's turn.

        Returns:
            List of environment indices with player 0 active.
        """
        active_indices = self.get_active_indices()
        env_indices = []

        for env_idx in active_indices:
            game = self.games[env_idx]
            if game is not None and game.active_player == 0:
                env_indices.append(int(env_idx))

        return env_indices

    def get_player1_games(self) -> list[int]:
        """Get environment indices of active games where it's player 1's turn.

        Returns:
            List of environment indices with player 1 active.
        """
        active_indices = self.get_active_indices()
        env_indices = []

        for env_idx in active_indices:
            game = self.games[env_idx]
            if game is not None and game.active_player == 1:
                env_indices.append(int(env_idx))

        return env_indices

    def get_observations_for_player(
        self, player: int, env_indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get batched observations and masks for specific player in specified games.

        Args:
            player: Player index (0 or 1).
            env_indices: List of environment indices to get observations for.

        Returns:
            Tuple of:
                - observations: (num_games, OBSERVATION_DIM) tensor on device
                - masks: (num_games, ACTION_DIM) tensor on device
        """
        num_games = len(env_indices)

        if num_games == 0:
            return (
                torch.empty((0, OBSERVATION_DIM), device=self.device),
                torch.empty((0, ACTION_DIM), device=self.device),
            )

        obs_batch = np.zeros((num_games, OBSERVATION_DIM), dtype=np.float32)
        mask_batch = np.zeros((num_games, ACTION_DIM), dtype=np.float32)

        for batch_idx, env_idx in enumerate(env_indices):
            game = self.games[env_idx]
            if game is None:
                continue

            obs_batch[batch_idx] = state_to_tensor(game, player)
            mask_batch[batch_idx] = legal_action_mask_tensor(game, player)

        obs_tensor = torch.from_numpy(obs_batch).to(self.device)
        mask_tensor = torch.from_numpy(mask_batch).to(self.device)

        return obs_tensor, mask_tensor

    def step_player0_network(
        self,
        env_indices: list[int],
        actions: torch.Tensor,
        action_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> int:
        """Apply network-selected actions for player 0 games.

        Args:
            env_indices: Environment indices for these actions.
            actions: (num_games,) tensor of action indices.
            action_probs: (num_games,) tensor of action probabilities.
            values: (num_games,) tensor of value estimates.

        Returns:
            Number of games that finished this step.
        """
        if len(env_indices) == 0:
            return 0

        actions_np = actions.cpu().numpy()
        probs_np = action_probs.cpu().numpy()
        values_np = values.cpu().numpy()

        games_finished = 0

        for batch_idx, env_idx in enumerate(env_indices):
            game = self.games[env_idx]
            if game is None:
                continue

            action_idx = int(actions_np[batch_idx])
            action_prob = float(probs_np[batch_idx])
            value = float(values_np[batch_idx])

            # Get observation and mask for trajectory
            obs = state_to_tensor(game, 0)
            mask = legal_action_mask_tensor(game, 0)

            # Store trajectory step
            pending = PendingStep(
                observation=obs,
                action_mask=mask,
                action=action_idx,
                action_prob=action_prob,
                value=value,
                player=0,
            )
            self.trajectories[env_idx].add_step(pending)

            # Apply action
            action = index_to_action(action_idx)
            new_state = simulate.apply(game, action)
            self.games[env_idx] = new_state

            # Check for terminal state
            if simulate.is_terminal(new_state):
                self.active[env_idx] = False
                self.trajectories[env_idx].final_state = new_state
                games_finished += 1

        return games_finished

    def step_player1_agent(
        self,
        env_indices: list[int],
        agent: Agent,
    ) -> int:
        """Apply agent-selected actions for player 1 games.

        This is sequential per-game but amortized across the batch.

        Args:
            env_indices: Environment indices where player 1 is active.
            agent: Agent to select actions (RandomAgent, HeuristicAgent, etc.).

        Returns:
            Number of games that finished this step.
        """
        if len(env_indices) == 0:
            return 0

        games_finished = 0

        for env_idx in env_indices:
            game = self.games[env_idx]
            if game is None:
                continue

            # Get legal actions and let agent select
            legal = simulate.legal_actions(game, 1)
            action = agent.select_action(game, legal)
            action_idx = action_index(action)

            # Get observation and mask for trajectory
            obs = state_to_tensor(game, 1)
            mask = legal_action_mask_tensor(game, 1)

            # Store trajectory step with placeholder values
            pending = PendingStep(
                observation=obs,
                action_mask=mask,
                action=action_idx,
                action_prob=1.0 / len(legal),  # Uniform estimate
                value=0.0,  # No value estimate from agent
                player=1,
            )
            self.trajectories[env_idx].add_step(pending)

            # Apply action
            new_state = simulate.apply(game, action)
            self.games[env_idx] = new_state

            # Check for terminal state
            if simulate.is_terminal(new_state):
                self.active[env_idx] = False
                self.trajectories[env_idx].final_state = new_state
                games_finished += 1

        return games_finished

    def step_player1_network(
        self,
        env_indices: list[int],
        actions: torch.Tensor,
        action_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> int:
        """Apply network-selected actions for player 1 games (checkpoint opponent).

        Args:
            env_indices: Environment indices for these actions.
            actions: (num_games,) tensor of action indices.
            action_probs: (num_games,) tensor of action probabilities.
            values: (num_games,) tensor of value estimates.

        Returns:
            Number of games that finished this step.
        """
        if len(env_indices) == 0:
            return 0

        actions_np = actions.cpu().numpy()
        probs_np = action_probs.cpu().numpy()
        values_np = values.cpu().numpy()

        games_finished = 0

        for batch_idx, env_idx in enumerate(env_indices):
            game = self.games[env_idx]
            if game is None:
                continue

            action_idx = int(actions_np[batch_idx])
            action_prob = float(probs_np[batch_idx])
            value = float(values_np[batch_idx])

            # Get observation and mask for trajectory
            obs = state_to_tensor(game, 1)
            mask = legal_action_mask_tensor(game, 1)

            # Store trajectory step
            pending = PendingStep(
                observation=obs,
                action_mask=mask,
                action=action_idx,
                action_prob=action_prob,
                value=value,
                player=1,
            )
            self.trajectories[env_idx].add_step(pending)

            # Apply action
            action = index_to_action(action_idx)
            new_state = simulate.apply(game, action)
            self.games[env_idx] = new_state

            # Check for terminal state
            if simulate.is_terminal(new_state):
                self.active[env_idx] = False
                self.trajectories[env_idx].final_state = new_state
                games_finished += 1

        return games_finished

    def get_trajectories(self, shaped_rewards: bool = False) -> list[EnvTrajectory]:
        """Get all trajectories with rewards assigned.

        Args:
            shaped_rewards: If True, use score-margin shaped rewards.

        Returns:
            List of EnvTrajectory objects with rewards assigned.
        """
        for env_idx in range(self.num_envs):
            traj = self.trajectories[env_idx]
            final_state = traj.final_state

            if final_state is None:
                continue

            # Get final scores
            scores = simulate.score(final_state)
            final_scores = (scores[0], scores[1])

            # Determine winner and rewards
            winner = compute_winner(final_scores)
            reward_p0, reward_p1 = compute_rewards(winner, final_scores, shaped=shaped_rewards)

            # Assign rewards to last steps
            if traj.steps_p0:
                traj.steps_p0[-1] = PendingStep(
                    observation=traj.steps_p0[-1].observation,
                    action_mask=traj.steps_p0[-1].action_mask,
                    action=traj.steps_p0[-1].action,
                    action_prob=traj.steps_p0[-1].action_prob,
                    value=traj.steps_p0[-1].value,
                    player=0,
                )
                traj.steps_p0[-1]._reward = reward_p0  # type: ignore[attr-defined]

            if traj.steps_p1:
                traj.steps_p1[-1] = PendingStep(
                    observation=traj.steps_p1[-1].observation,
                    action_mask=traj.steps_p1[-1].action_mask,
                    action=traj.steps_p1[-1].action,
                    action_prob=traj.steps_p1[-1].action_prob,
                    value=traj.steps_p1[-1].value,
                    player=1,
                )
                traj.steps_p1[-1]._reward = reward_p1  # type: ignore[attr-defined]

        return self.trajectories


def generate_games_vectorized_with_opponent(
    network: nn.Module,
    opponent: Agent | None = None,
    opponent_network: nn.Module | None = None,
    num_games: int = 256,
    temperature: float = 1.0,
    device: torch.device | None = None,
    seeds: list[int] | None = None,
    shaped_rewards: bool = False,
) -> tuple[list[EnvTrajectory], dict[str, float]]:
    """Generate games with opponent diversity using vectorized environment.

    Player 0 (training player) always uses the main network with batched inference.
    Player 1 (opponent) can be:
    - None: Same as player 0 (standard self-play, uses generate_games_vectorized)
    - Agent: Random/Heuristic agent (sequential but cheap)
    - opponent_network: Another neural network (batched inference for both)

    This preserves most GPU efficiency even with non-neural opponents because:
    - Network inference for player 0 is still batched across all games
    - Only ~50% of moves are opponent moves
    - Agent opponents (random/heuristic) are computationally cheap

    Args:
        network: Neural network for player 0 (training player).
        opponent: Optional Agent for player 1 (RandomAgent, HeuristicAgent).
        opponent_network: Optional neural network for player 1 (checkpoint).
            Ignored if opponent is provided.
        num_games: Number of games to generate.
        temperature: Temperature for action sampling.
        device: Device for network inference.
        seeds: Optional list of seeds for each game.
        shaped_rewards: If True, use score-margin shaped rewards.

    Returns:
        Tuple of:
            - trajectories: List of EnvTrajectory objects.
            - stats: Dictionary of generation statistics.
    """
    # Import here to avoid circular imports at module level
    from _02_agents.base import Agent as AgentBase

    # If no opponent, use standard self-play
    if opponent is None and opponent_network is None:
        return generate_games_vectorized(
            network=network,
            num_games=num_games,
            temperature=temperature,
            device=device,
            seeds=seeds,
            shaped_rewards=shaped_rewards,
        )

    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Create vectorized environment
    env = VectorizedGameEnvWithOpponents(num_envs=num_games, device=device)

    # Reset with seeds
    if seeds is None:
        base_seed = np.random.randint(0, 2**31)
        seeds = [base_seed + i for i in range(num_games)]
    env.reset(seeds)

    # Determine opponent type
    has_agent_opponent = opponent is not None and isinstance(opponent, AgentBase)
    has_network_opponent = opponent_network is not None and not has_agent_opponent

    # Stats tracking
    total_steps = 0
    p0_inference_calls = 0
    p1_inference_calls = 0
    p1_agent_steps = 0

    # Run until all games complete - use inference_mode for both networks
    with inference_mode(network):
        if has_network_opponent and opponent_network is not None:
            opponent_network.eval()

        with torch.no_grad():
            while not env.all_done():
                # Get games where it's player 0's turn
                p0_env_indices = env.get_player0_games()

                # Get games where it's player 1's turn
                p1_env_indices = env.get_player1_games()

                # Process player 0 moves (batched network inference)
                if p0_env_indices:
                    obs, masks = env.get_observations_for_player(0, p0_env_indices)

                    if obs.size(0) > 0:
                        # Batched inference for player 0
                        policy_logits, values = network(obs, masks)
                        values = values.squeeze(-1)
                        p0_inference_calls += 1

                        # Sample actions
                        actions, action_probs = sample_actions_batch(
                            policy_logits, masks, temperature
                        )

                        # Step player 0 games
                        env.step_player0_network(p0_env_indices, actions, action_probs, values)
                        total_steps += len(p0_env_indices)

                # Process player 1 moves
                if p1_env_indices:
                    if has_agent_opponent and opponent is not None:
                        # Sequential agent moves (but cheap)
                        env.step_player1_agent(p1_env_indices, opponent)
                        total_steps += len(p1_env_indices)
                        p1_agent_steps += len(p1_env_indices)

                    elif has_network_opponent and opponent_network is not None:
                        # Batched inference for opponent network
                        obs, masks = env.get_observations_for_player(1, p1_env_indices)

                        if obs.size(0) > 0:
                            policy_logits, values = opponent_network(obs, masks)
                            values = values.squeeze(-1)
                            p1_inference_calls += 1

                            actions, action_probs = sample_actions_batch(
                                policy_logits, masks, temperature
                            )
                            env.step_player1_network(
                                p1_env_indices, actions, action_probs, values
                            )
                            total_steps += len(p1_env_indices)

    # Get trajectories with rewards
    trajectories = env.get_trajectories(shaped_rewards=shaped_rewards)

    total_inference_calls = p0_inference_calls + p1_inference_calls
    stats = {
        "total_steps": float(total_steps),
        "p0_inference_calls": float(p0_inference_calls),
        "p1_inference_calls": float(p1_inference_calls),
        "p1_agent_steps": float(p1_agent_steps),
        "total_inference_calls": float(total_inference_calls),
        "avg_batch_size": total_steps / max(total_inference_calls, 1),
        "games_generated": float(num_games),
    }

    return trajectories, stats


# =============================================================================
# Cython Auto-Detection: Use Cython implementations when available
# =============================================================================
# This section transparently swaps in Cython-accelerated implementations
# when the Cython extension is built. Code importing from this module will
# automatically get the faster versions without any changes.

_USING_CYTHON = False

try:
    from _03_training.vectorized_env_cython import (
        generate_games_vectorized_cython,
        is_cython_available,
    )

    if is_cython_available():
        # Cython is available - use accelerated version
        _original_generate_games_vectorized = generate_games_vectorized

        def generate_games_vectorized(
            network,
            num_games: int,
            temperature: float = 1.0,
            device=None,
            seeds: list[int] | None = None,
            shaped_rewards: bool = False,
        ):
            """Auto-dispatching wrapper that uses Cython when available."""
            return generate_games_vectorized_cython(
                network=network,
                num_games=num_games,
                temperature=temperature,
                device=device,
                seeds=seeds,
                shaped_rewards=shaped_rewards,
            )

        _USING_CYTHON = True

except ImportError:
    pass


def is_using_cython() -> bool:
    """Check if Cython acceleration is being used."""
    return _USING_CYTHON


__all__ = [
    "EnvTrajectory",
    "PendingStep",
    "VectorizedGameEnv",
    "VectorizedGameEnvWithOpponents",
    "generate_games_vectorized",
    "generate_games_vectorized_with_opponent",
    "is_using_cython",
    "sample_actions_batch",
]
