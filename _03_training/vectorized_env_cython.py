"""Cython-accelerated vectorized game environment.

This module provides a drop-in replacement for VectorizedGameEnv that uses
the Cython extension for GIL-free, multi-threaded game simulation.

When Cython is available, this provides 10-20x speedup on CPU operations
through OpenMP parallelization across all available cores.

Example:
    from _03_training.vectorized_env_cython import (
        VectorizedGameEnvCython,
        generate_games_vectorized_cython,
        is_cython_available,
    )

    if is_cython_available():
        env = VectorizedGameEnvCython(num_envs=512)
    else:
        # Fall back to pure Python
        from _03_training.vectorized_env import VectorizedGameEnv
        env = VectorizedGameEnv(num_envs=512)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

# Try to import Cython acceleration
try:
    from _01_simulator._cython import (
        GameStateArray,
        encode_observations_parallel,
        get_legal_masks_parallel,
        is_cython_available,
        step_batch_parallel,
    )

    _CYTHON_AVAILABLE = is_cython_available()
except ImportError:
    _CYTHON_AVAILABLE = False

    def is_cython_available() -> bool:
        return False


# Import pure Python implementation for fallback
from _01_simulator import simulate
from _01_simulator.action_space import ACTION_DIM, index_to_action, legal_action_mask_tensor
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor
from _03_training.game_utils import compute_rewards, compute_winner
from _03_training.trajectory import PendingStep

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import nn

    from _01_simulator.state import State


@dataclass
class EnvTrajectory:
    """Complete trajectory data from a single game."""

    steps_p0: list[PendingStep] = field(default_factory=list)
    steps_p1: list[PendingStep] = field(default_factory=list)
    seed: int = 0
    final_scores: tuple[int, int] | None = None

    def add_step(self, step: PendingStep) -> None:
        """Add step to appropriate player's trajectory."""
        if step.player == 0:
            self.steps_p0.append(step)
        else:
            self.steps_p1.append(step)


class VectorizedGameEnvCython:
    """Cython-accelerated vectorized environment with OpenMP parallelization.

    This provides true multi-threaded game simulation by:
    1. Storing all game states in contiguous C memory
    2. Using GIL-releasing Cython functions
    3. Parallelizing across CPU cores with OpenMP

    Falls back to pure Python implementation if Cython is not available.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | None = None,
        num_threads: int | None = None,
    ) -> None:
        """Initialize vectorized environment.

        Args:
            num_envs: Number of parallel game environments.
            device: PyTorch device for inference tensors.
            num_threads: Number of threads for parallel operations.
                         If None, uses all available cores.
        """
        self.num_envs = num_envs
        self.device = device or torch.device("cpu")
        self.num_threads = num_threads or os.cpu_count() or 4
        self._use_cython = _CYTHON_AVAILABLE

        if self._use_cython:
            # Cython mode: use C struct array
            self._c_states = GameStateArray(num_envs)
        else:
            # Fallback: Python game states
            self._py_games: list[State | None] = [None] * num_envs

        # Common state tracking
        self.active = np.zeros(num_envs, dtype=bool)
        self.trajectories: list[EnvTrajectory] = [EnvTrajectory() for _ in range(num_envs)]

        # Pre-allocated buffers for batch operations
        self._obs_buffer = np.zeros((num_envs, OBSERVATION_DIM), dtype=np.float32)
        self._mask_buffer = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)
        self._active_indices: NDArray[np.int64] = np.array([], dtype=np.int64)
        self._active_players: NDArray[np.int32] = np.array([], dtype=np.int32)

    @property
    def using_cython(self) -> bool:
        """Check if Cython acceleration is active."""
        return self._use_cython

    def reset(self, seeds: list[int] | None = None) -> None:
        """Reset all environments with new games."""
        if seeds is None:
            base_seed = np.random.randint(0, 2**31)
            seeds = [base_seed + i for i in range(self.num_envs)]
        elif len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} seeds, got {len(seeds)}")

        if self._use_cython:
            self._c_states.resize(0)
            for i, seed in enumerate(seeds):
                self._c_states.init_game(i, seed)
        else:
            for i, seed in enumerate(seeds):
                self._py_games[i] = simulate.new_game(seed)

        self.active[:] = True
        self.trajectories = [EnvTrajectory(seed=seed) for seed in seeds]

    def num_active(self) -> int:
        """Return number of currently active games."""
        return int(self.active.sum())

    def all_done(self) -> bool:
        """Check if all games have finished."""
        return not self.active.any()

    def get_active_indices(self) -> NDArray[np.int64]:
        """Return indices of active games."""
        return np.where(self.active)[0]

    def get_observations_and_masks(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, NDArray[np.int64]]:
        """Get batched observations and action masks for all active games.

        Returns:
            Tuple of (observations, masks, active_indices)
        """
        active_indices = self.get_active_indices()
        num_active = len(active_indices)

        if num_active == 0:
            return (
                torch.empty((0, OBSERVATION_DIM), device=self.device),
                torch.empty((0, ACTION_DIM), device=self.device),
                active_indices,
            )

        # Prepare output buffers (contiguous slice)
        obs_batch = np.zeros((num_active, OBSERVATION_DIM), dtype=np.float32)
        mask_batch = np.zeros((num_active, ACTION_DIM), dtype=np.float32)
        players = np.zeros(num_active, dtype=np.int32)

        if self._use_cython:
            # Use parallel Cython encoding
            encode_observations_parallel(
                self._c_states,
                active_indices,
                obs_batch,
                self.num_threads,
            )
            get_legal_masks_parallel(
                self._c_states,
                active_indices,
                mask_batch,
                self.num_threads,
            )
            # Get active players
            for batch_idx, env_idx in enumerate(active_indices):
                players[batch_idx] = self._c_states.get_active_player(env_idx)
        else:
            # Pure Python fallback
            for batch_idx, env_idx in enumerate(active_indices):
                game = self._py_games[env_idx]
                if game is None:
                    continue
                player = game.active_player
                players[batch_idx] = player
                obs_batch[batch_idx] = state_to_tensor(game, player)
                mask_batch[batch_idx] = legal_action_mask_tensor(game, player)

        # Store for step()
        self._active_players = players
        self._active_indices = active_indices

        # Convert to tensors
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

        Returns:
            Number of games that finished this step.
        """
        actions_np = actions.cpu().numpy().astype(np.int64)
        probs_np = action_probs.cpu().numpy()
        values_np = values.cpu().numpy()

        active_indices = self._active_indices
        players = self._active_players

        # Store trajectory data before stepping
        for batch_idx, env_idx in enumerate(active_indices):
            player = int(players[batch_idx])
            action_idx = int(actions_np[batch_idx])
            action_prob = float(probs_np[batch_idx])
            value = float(values_np[batch_idx])

            # Get current observation and mask for this step
            if self._use_cython:
                from _01_simulator._cython import encode_single_observation, get_single_legal_mask

                obs = encode_single_observation(self._c_states, env_idx, player)
                mask = get_single_legal_mask(self._c_states, env_idx, player)
            else:
                game = self._py_games[env_idx]
                if game is None:
                    continue
                obs = state_to_tensor(game, player)
                mask = legal_action_mask_tensor(game, player)

            pending = PendingStep(
                observation=obs,
                action_mask=mask,
                action=action_idx,
                action_prob=action_prob,
                value=value,
                player=player,
            )
            self.trajectories[env_idx].add_step(pending)

        # Apply actions
        if self._use_cython:
            # Parallel Cython step
            games_finished = step_batch_parallel(
                self._c_states,
                active_indices,
                actions_np,
                self.num_threads,
            )
            # Check terminal states and update active flags
            for env_idx in active_indices:
                if self._c_states.is_terminal(env_idx):
                    self.active[env_idx] = False
                    scores = self._c_states.get_scores(env_idx)
                    self.trajectories[env_idx].final_scores = scores
        else:
            # Pure Python fallback
            games_finished = 0
            for batch_idx, env_idx in enumerate(active_indices):
                game = self._py_games[env_idx]
                if game is None:
                    continue

                action_idx = int(actions_np[batch_idx])
                action = index_to_action(action_idx)
                new_state = simulate.apply(game, action)
                self._py_games[env_idx] = new_state

                if simulate.is_terminal(new_state):
                    self.active[env_idx] = False
                    scores = simulate.score(new_state)
                    self.trajectories[env_idx].final_scores = (scores[0], scores[1])
                    games_finished += 1

        return games_finished

    def get_trajectories(self, shaped_rewards: bool = False) -> list[EnvTrajectory]:
        """Get all trajectories with rewards assigned."""
        for env_idx in range(self.num_envs):
            traj = self.trajectories[env_idx]
            final_scores = traj.final_scores

            if final_scores is None:
                continue

            # Determine winner and rewards
            winner = compute_winner(final_scores)
            reward_p0, reward_p1 = compute_rewards(winner, final_scores, shaped=shaped_rewards)

            # Store rewards
            if traj.steps_p0:
                traj.steps_p0[-1]._reward = reward_p0  # type: ignore[attr-defined]
            if traj.steps_p1:
                traj.steps_p1[-1]._reward = reward_p1  # type: ignore[attr-defined]

        return self.trajectories


def sample_actions_batch(
    logits: torch.Tensor,
    masks: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample actions from batched policy logits with temperature."""
    masked_logits = torch.where(
        masks > 0,
        logits,
        torch.tensor(float("-inf"), device=logits.device),
    )

    scaled_logits = masked_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Handle NaN/Inf
    invalid_mask = torch.isnan(probs).any(dim=-1) | torch.isinf(probs).any(dim=-1)
    if invalid_mask.any():
        uniform_probs = masks / masks.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs = torch.where(invalid_mask.unsqueeze(-1), uniform_probs, probs)

    actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
    action_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    return actions, action_probs


def generate_games_vectorized_cython(
    network: nn.Module,
    num_games: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
    seeds: list[int] | None = None,
    shaped_rewards: bool = False,
    num_threads: int | None = None,
) -> tuple[list[EnvTrajectory], dict[str, float]]:
    """Generate self-play games using Cython-accelerated environment.

    This is the main entry point for efficient self-play with Cython.
    """
    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    env = VectorizedGameEnvCython(
        num_envs=num_games,
        device=device,
        num_threads=num_threads,
    )

    if seeds is None:
        base_seed = np.random.randint(0, 2**31)
        seeds = [base_seed + i for i in range(num_games)]
    env.reset(seeds)

    network.eval()

    total_steps = 0
    inference_calls = 0

    with torch.no_grad():
        while not env.all_done():
            obs, masks, _ = env.get_observations_and_masks()

            if obs.size(0) == 0:
                break

            policy_logits, values = network(obs, masks)
            values = values.squeeze(-1)
            inference_calls += 1

            actions, action_probs = sample_actions_batch(policy_logits, masks, temperature)
            env.step(actions, action_probs, values)
            total_steps += len(actions)

    trajectories = env.get_trajectories(shaped_rewards=shaped_rewards)

    stats = {
        "total_steps": total_steps,
        "inference_calls": inference_calls,
        "avg_batch_size": total_steps / max(inference_calls, 1),
        "games_generated": num_games,
        "using_cython": env.using_cython,
    }

    return trajectories, stats


def _is_neural_network(obj: object) -> bool:
    """Check if an object is a neural network (nn.Module or has forward method).

    This is used to distinguish between Agent opponents (which need Python State objects)
    and neural network opponents (which can use Cython acceleration).

    Args:
        obj: Object to check.

    Returns:
        True if the object appears to be a neural network.
    """
    # Check if it's a torch.nn.Module
    if isinstance(obj, torch.nn.Module):
        return True
    # Duck typing: has forward method but not select_action (which agents have)
    return bool(hasattr(obj, "forward") and callable(getattr(obj, "forward", None)) and not hasattr(obj, "select_action"))


def generate_games_vectorized_cython_with_opponent(
    network: nn.Module,
    opponent: Agent | nn.Module | None = None,
    opponent_network: nn.Module | None = None,
    num_games: int = 256,
    temperature: float = 1.0,
    device: torch.device | None = None,
    seeds: list[int] | None = None,
    shaped_rewards: bool = False,
    num_threads: int | None = None,
) -> tuple[list[EnvTrajectory], dict[str, float]]:
    """Generate games with opponent diversity using Cython-accelerated environment.

    Player 0 (training player) always uses the main network with batched inference.
    Player 1 (opponent) can be:
    - None: Same as player 0 (standard self-play)
    - Agent: Random/Heuristic agent - falls back to pure Python (Cython lacks State conversion)
    - nn.Module: Another neural network (batched inference for both, Cython accelerated)
    - opponent_network: Another neural network (batched inference for both, Cython accelerated)

    The function automatically detects whether the opponent is a neural network or an agent:
    - Neural networks (nn.Module or objects with forward() but not select_action()) use Cython
    - Agents (objects with select_action() method) fall back to Python

    Note: When opponent is an Agent, this falls back to pure Python because the Cython
    extension doesn't expose a c_state_to_python conversion function needed for agents.

    Args:
        network: Neural network for player 0 (training player).
        opponent: Optional opponent for player 1. Can be either:
            - An Agent (RandomAgent, HeuristicAgent) - falls back to Python
            - A neural network (nn.Module) - uses Cython acceleration
        opponent_network: Optional neural network for player 1 (checkpoint).
            Ignored if opponent is provided.
        num_games: Number of games to generate.
        temperature: Temperature for action sampling.
        device: Device for network inference.
        seeds: Optional list of seeds for each game.
        shaped_rewards: If True, use score-margin shaped rewards.
        num_threads: Number of threads for Cython parallel operations.

    Returns:
        Tuple of:
            - trajectories: List of EnvTrajectory objects.
            - stats: Dictionary of generation statistics.
    """
    # If no opponent, use standard self-play
    if opponent is None and opponent_network is None:
        return generate_games_vectorized_cython(
            network=network,
            num_games=num_games,
            temperature=temperature,
            device=device,
            seeds=seeds,
            shaped_rewards=shaped_rewards,
            num_threads=num_threads,
        )

    # Check if opponent is a neural network (can use Cython path)
    # This handles the case where a network is passed as `opponent` instead of `opponent_network`
    if opponent is not None and _is_neural_network(opponent):
        # Treat the network passed as `opponent` as `opponent_network`
        opponent_network = opponent  # type: ignore[assignment]
        opponent = None

    # For agent opponents (have select_action method), fall back to pure Python
    # (Cython doesn't expose c_state_to_python for agent.select_action calls)
    has_agent_opponent = opponent is not None and hasattr(opponent, "select_action")
    if has_agent_opponent:
        from _03_training.vectorized_env import generate_games_vectorized_with_opponent

        return generate_games_vectorized_with_opponent(
            network=network,
            opponent=opponent,
            opponent_network=None,
            num_games=num_games,
            temperature=temperature,
            device=device,
            seeds=seeds,
            shaped_rewards=shaped_rewards,
        )

    # Network opponent case - can use Cython acceleration
    if device is None:
        try:
            device = next(network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    env = VectorizedGameEnvCython(
        num_envs=num_games,
        device=device,
        num_threads=num_threads,
    )

    if seeds is None:
        base_seed = np.random.randint(0, 2**31)
        seeds = [base_seed + i for i in range(num_games)]
    env.reset(seeds)

    # Stats tracking
    total_steps = 0
    p0_inference_calls = 0
    p1_inference_calls = 0

    network.eval()
    if opponent_network is not None:
        opponent_network.eval()

    with torch.no_grad():
        while not env.all_done():
            # Get observations and masks for all active games
            obs, masks, active_indices = env.get_observations_and_masks()

            if obs.size(0) == 0:
                break

            # Get active players for current batch
            players = env._active_players

            # Separate games by active player
            p0_batch_indices = []
            p1_batch_indices = []
            p0_env_indices = []
            p1_env_indices = []

            for batch_idx, env_idx in enumerate(active_indices):
                if players[batch_idx] == 0:
                    p0_batch_indices.append(batch_idx)
                    p0_env_indices.append(env_idx)
                else:
                    p1_batch_indices.append(batch_idx)
                    p1_env_indices.append(env_idx)

            # Process player 0 moves (always batched network inference)
            if p0_batch_indices:
                p0_obs = obs[p0_batch_indices]
                p0_masks = masks[p0_batch_indices]

                policy_logits, values = network(p0_obs, p0_masks)
                values = values.squeeze(-1)
                p0_inference_calls += 1

                actions, action_probs = sample_actions_batch(policy_logits, p0_masks, temperature)

                # Apply actions via Cython
                actions_np = actions.cpu().numpy().astype(np.int64)
                probs_np = action_probs.cpu().numpy()
                values_np = values.cpu().numpy()
                p0_env_indices_np = np.array(p0_env_indices, dtype=np.int64)

                # Store trajectory data and step
                for i, env_idx in enumerate(p0_env_indices):
                    player = 0
                    action_idx = int(actions_np[i])
                    action_prob = float(probs_np[i])
                    value = float(values_np[i])

                    if env._use_cython:
                        from _01_simulator._cython import encode_single_observation, get_single_legal_mask
                        obs_arr = encode_single_observation(env._c_states, env_idx, player)
                        mask_arr = get_single_legal_mask(env._c_states, env_idx, player)
                    else:
                        game = env._py_games[env_idx]
                        obs_arr = state_to_tensor(game, player)
                        mask_arr = legal_action_mask_tensor(game, player)

                    pending = PendingStep(
                        observation=obs_arr,
                        action_mask=mask_arr,
                        action=action_idx,
                        action_prob=action_prob,
                        value=value,
                        player=player,
                    )
                    env.trajectories[env_idx].add_step(pending)

                # Step using Cython
                if env._use_cython:
                    step_batch_parallel(
                        env._c_states,
                        p0_env_indices_np,
                        actions_np,
                        env.num_threads,
                    )
                    for env_idx in p0_env_indices:
                        if env._c_states.is_terminal(env_idx):
                            env.active[env_idx] = False
                            scores = env._c_states.get_scores(env_idx)
                            env.trajectories[env_idx].final_scores = scores
                else:
                    for i, env_idx in enumerate(p0_env_indices):
                        game = env._py_games[env_idx]
                        action = index_to_action(int(actions_np[i]))
                        new_state = simulate.apply(game, action)
                        env._py_games[env_idx] = new_state
                        if simulate.is_terminal(new_state):
                            env.active[env_idx] = False
                            scores = simulate.score(new_state)
                            env.trajectories[env_idx].final_scores = (scores[0], scores[1])

                total_steps += len(p0_env_indices)

            # Process player 1 moves (network opponent)
            if p1_batch_indices and opponent_network is not None:
                # Batched inference for opponent network
                p1_obs = obs[p1_batch_indices]
                p1_masks = masks[p1_batch_indices]

                policy_logits, values = opponent_network(p1_obs, p1_masks)
                values = values.squeeze(-1)
                p1_inference_calls += 1

                actions, action_probs = sample_actions_batch(policy_logits, p1_masks, temperature)

                actions_np = actions.cpu().numpy().astype(np.int64)
                probs_np = action_probs.cpu().numpy()
                values_np = values.cpu().numpy()
                p1_env_indices_np = np.array(p1_env_indices, dtype=np.int64)

                # Store trajectory data
                for i, env_idx in enumerate(p1_env_indices):
                    player = 1
                    action_idx = int(actions_np[i])
                    action_prob = float(probs_np[i])
                    value = float(values_np[i])

                    if env._use_cython:
                        from _01_simulator._cython import encode_single_observation, get_single_legal_mask
                        obs_arr = encode_single_observation(env._c_states, env_idx, player)
                        mask_arr = get_single_legal_mask(env._c_states, env_idx, player)
                    else:
                        game = env._py_games[env_idx]
                        obs_arr = state_to_tensor(game, player)
                        mask_arr = legal_action_mask_tensor(game, player)

                    pending = PendingStep(
                        observation=obs_arr,
                        action_mask=mask_arr,
                        action=action_idx,
                        action_prob=action_prob,
                        value=value,
                        player=player,
                    )
                    env.trajectories[env_idx].add_step(pending)

                # Step using Cython
                if env._use_cython:
                    step_batch_parallel(
                        env._c_states,
                        p1_env_indices_np,
                        actions_np,
                        env.num_threads,
                    )
                    for env_idx in p1_env_indices:
                        if env._c_states.is_terminal(env_idx):
                            env.active[env_idx] = False
                            scores = env._c_states.get_scores(env_idx)
                            env.trajectories[env_idx].final_scores = scores
                else:
                    for i, env_idx in enumerate(p1_env_indices):
                        game = env._py_games[env_idx]
                        action = index_to_action(int(actions_np[i]))
                        new_state = simulate.apply(game, action)
                        env._py_games[env_idx] = new_state
                        if simulate.is_terminal(new_state):
                            env.active[env_idx] = False
                            scores = simulate.score(new_state)
                            env.trajectories[env_idx].final_scores = (scores[0], scores[1])

                total_steps += len(p1_env_indices)

    trajectories = env.get_trajectories(shaped_rewards=shaped_rewards)

    total_inference_calls = p0_inference_calls + p1_inference_calls
    stats = {
        "total_steps": float(total_steps),
        "p0_inference_calls": float(p0_inference_calls),
        "p1_inference_calls": float(p1_inference_calls),
        "p1_agent_steps": 0.0,  # Always 0 for Cython path (agents fall back to pure Python)
        "total_inference_calls": float(total_inference_calls),
        "avg_batch_size": total_steps / max(total_inference_calls, 1),
        "games_generated": float(num_games),
        "using_cython": env.using_cython,
    }

    return trajectories, stats


if TYPE_CHECKING:
    from _02_agents.base import Agent


__all__ = [
    "EnvTrajectory",
    "PendingStep",
    "VectorizedGameEnvCython",
    "generate_games_vectorized_cython",
    "generate_games_vectorized_cython_with_opponent",
    "is_cython_available",
    "sample_actions_batch",
]
