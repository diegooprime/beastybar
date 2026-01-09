"""
Pytest configuration and fixtures for neural network tests.

Provides shared test fixtures for game states, observations, networks,
and training data to be used across all neural test modules.
"""

import random

import numpy as np
import pytest

from _01_simulator import engine, state
from _01_simulator.action_space import ACTION_DIM, legal_action_mask_tensor
from _01_simulator.actions import Action
from _01_simulator.observations import OBSERVATION_DIM, build_observation, state_to_tensor
from _02_agents.neural.network import create_network
from _02_agents.neural.utils import NetworkConfig


@pytest.fixture
def sample_game_state():
    """Provide a mid-game State for testing.

    Returns a game state with:
    - 2 players (per rules.PLAYER_COUNT)
    - Animals in the queue zone
    - Some cards moved to beasty_bar and thats_it zones
    - Cards played from hands (reduced hand sizes)
    - Current player ready to act
    - Not a terminal state
    """
    # Create initial state with fixed seed for reproducibility
    game_state = state.initial_state(seed=12345)

    # Play 5-8 random moves to create a realistic mid-game state
    rng = random.Random(12345)
    moves_played = 0
    target_moves = rng.randint(5, 8)

    while moves_played < target_moves and not engine.is_terminal(game_state):
        player = game_state.active_player
        legal = list(engine.legal_actions(game_state, player))
        if not legal:
            break
        action = rng.choice(legal)
        game_state = engine.step(game_state, action)
        moves_played += 1

    # Verify we have a proper mid-game state
    assert not engine.is_terminal(game_state), "Mid-game fixture should not be terminal"
    assert game_state.turn > 0, "Mid-game fixture should have advanced turns"

    return game_state


@pytest.fixture
def sample_observation(sample_game_state):
    """Provide an Observation from a game state.

    Returns observation for the current player in sample_game_state,
    containing visible game information from their perspective.
    """
    return build_observation(
        game_state=sample_game_state, perspective=sample_game_state.active_player
    )


@pytest.fixture
def sample_actions():
    """Provide a list of sample Action objects.

    Returns diverse actions representing playing different hand cards
    with or without additional parameters.
    """
    actions = [
        Action(hand_index=0),
        Action(hand_index=1),
        Action(hand_index=2, params=(1,)),  # Action with params (e.g., kangaroo hop)
        Action(hand_index=3, params=(0, 2)),  # Action with multiple params (e.g., chameleon)
    ]
    return actions


@pytest.fixture
def random_network():
    """Provide an initialized neural network for testing.

    Returns a randomly initialized actor-critic network
    ready for testing forward passes and training.
    """
    config = NetworkConfig(
        hidden_dim=32,  # Small for fast testing
        num_heads=2,
        num_layers=1,
        dropout=0.0,  # No dropout for deterministic tests
    )
    return create_network(config)


@pytest.fixture
def sample_trajectory():
    """Provide realistic trajectory data for testing.

    Returns a dictionary containing a complete game trajectory:
    - observations: List of observation tensors (numpy arrays)
    - actions: List of action indices (integers)
    - rewards: List of rewards (floats)
    - dones: List of terminal flags (booleans)
    - action_masks: List of legal action masks (numpy arrays)
    - values: List of value estimates (floats)
    - action_probs: List of action probabilities (floats)

    The trajectory represents a realistic game sequence with proper
    observation dimensions and valid action indices.
    """
    # Play a short game to generate realistic trajectory data
    game_state = state.initial_state(seed=54321)
    rng = random.Random(54321)

    observations = []
    actions = []
    rewards = []
    dones = []
    action_masks = []
    values = []
    action_probs = []

    # Collect trajectory from 6-10 steps
    steps = 0
    max_steps = rng.randint(6, 10)

    while steps < max_steps and not engine.is_terminal(game_state):
        player = game_state.active_player

        # Get observation tensor
        obs_tensor = state_to_tensor(game_state, player)
        observations.append(obs_tensor)

        # Get action mask
        mask = legal_action_mask_tensor(game_state, player)
        action_masks.append(mask)

        # Get legal actions and select one
        legal = list(engine.legal_actions(game_state, player))
        if not legal:
            break

        from _01_simulator.action_space import action_index

        selected_action = rng.choice(legal)
        action_idx = action_index(selected_action)
        actions.append(action_idx)

        # Simulate value and action probability
        values.append(rng.uniform(-0.5, 0.5))
        legal_count = sum(mask)
        action_probs.append(1.0 / max(legal_count, 1))

        # Step game
        game_state = engine.step(game_state, selected_action)
        steps += 1

        # Intermediate rewards are 0, final reward set at end
        is_done = engine.is_terminal(game_state)
        dones.append(is_done)

        if is_done:
            scores = engine.score(game_state)
            # Reward from perspective of player who just acted
            if scores[player] > scores[1 - player]:
                rewards.append(1.0)
            elif scores[player] < scores[1 - player]:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    # If game didn't end naturally, mark last step as done
    if dones and not dones[-1]:
        dones[-1] = True

    trajectory = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "action_masks": action_masks,
        "values": values,
        "action_probs": action_probs,
    }

    return trajectory


@pytest.fixture
def sample_batch(sample_trajectory):
    """Provide a batch of trajectories for training.

    Returns a batched version of trajectory data with proper tensor shapes:
    - observations: Tensor of shape (batch_size, obs_dim)
    - actions: Tensor of shape (batch_size,)
    - rewards: Tensor of shape (batch_size,)
    - dones: Tensor of shape (batch_size,)
    - action_masks: Tensor of shape (batch_size, num_actions)
    - values: Tensor of shape (batch_size,)
    - action_probs: Tensor of shape (batch_size,)

    Creates batch by replicating and varying the sample trajectory.
    """
    batch_size = 4

    # Get data from sample trajectory
    traj_obs = sample_trajectory["observations"]
    traj_actions = sample_trajectory["actions"]
    traj_rewards = sample_trajectory["rewards"]
    traj_dones = sample_trajectory["dones"]
    traj_masks = sample_trajectory["action_masks"]
    traj_values = sample_trajectory["values"]
    traj_probs = sample_trajectory["action_probs"]

    # Use minimum of trajectory length and batch_size
    actual_batch_size = min(batch_size, len(traj_obs))

    if actual_batch_size == 0:
        # Fallback: create synthetic batch data
        actual_batch_size = batch_size
        observations = np.random.randn(actual_batch_size, OBSERVATION_DIM).astype(np.float32)
        action_masks = np.ones((actual_batch_size, ACTION_DIM), dtype=np.float32)
        # Make some actions illegal for realism
        for i in range(actual_batch_size):
            illegal_count = np.random.randint(ACTION_DIM // 2, ACTION_DIM - 4)
            illegal_indices = np.random.choice(ACTION_DIM, illegal_count, replace=False)
            action_masks[i, illegal_indices] = 0.0
        actions = np.array([np.random.choice(np.where(m > 0)[0]) for m in action_masks], dtype=np.int64)
        rewards = np.random.choice([-1.0, 0.0, 1.0], size=actual_batch_size).astype(np.float32)
        dones = np.array([i == actual_batch_size - 1 for i in range(actual_batch_size)], dtype=bool)
        values = np.random.uniform(-1, 1, size=actual_batch_size).astype(np.float32)
        action_probs = np.random.uniform(0.01, 1.0, size=actual_batch_size).astype(np.float32)
    else:
        # Stack trajectory data into batch arrays
        observations = np.stack(traj_obs[:actual_batch_size], axis=0).astype(np.float32)
        action_masks = np.stack(traj_masks[:actual_batch_size], axis=0).astype(np.float32)
        actions = np.array(traj_actions[:actual_batch_size], dtype=np.int64)
        rewards = np.array(traj_rewards[:actual_batch_size], dtype=np.float32)
        dones = np.array(traj_dones[:actual_batch_size], dtype=bool)
        values = np.array(traj_values[:actual_batch_size], dtype=np.float32)
        action_probs = np.array(traj_probs[:actual_batch_size], dtype=np.float32)

    batch = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "action_masks": action_masks,
        "values": values,
        "action_probs": action_probs,
        "batch_size": actual_batch_size,
    }

    return batch
