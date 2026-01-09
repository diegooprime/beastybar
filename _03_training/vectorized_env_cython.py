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
        is_terminal_batch,
        python_state_to_c,
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


__all__ = [
    "EnvTrajectory",
    "PendingStep",
    "VectorizedGameEnvCython",
    "generate_games_vectorized_cython",
    "is_cython_available",
    "sample_actions_batch",
]
