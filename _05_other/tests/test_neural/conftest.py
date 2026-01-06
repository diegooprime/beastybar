"""
Pytest configuration and fixtures for neural network tests.

Provides shared test fixtures for game states, observations, networks,
and training data to be used across all neural test modules.
"""

import pytest
from _01_simulator.state import State
from _01_simulator.observations import Observation
from _01_simulator.actions import Action


@pytest.fixture
def sample_game_state():
    """Provide a mid-game State for testing.

    Returns a game state with:
    - Multiple players (3-5)
    - Some animals in zones (sky, land, water)
    - Some cards in hands
    - Current player ready to act
    - Not a terminal state
    """
    # Create a simple mid-game state
    state = State.create_initial_state(num_players=3)

    # TODO: Add some animals to zones and cards to hands
    # This will be properly implemented when Phase 10.2 begins
    # For now, return initial state as placeholder

    return state


@pytest.fixture
def sample_observation(sample_game_state):
    """Provide an Observation from a game state.

    Returns observation for the current player in sample_game_state,
    containing visible game information from their perspective.
    """
    return Observation.from_state(
        state=sample_game_state, player_index=sample_game_state.current_player
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
        Action(hand_index=2, params=(1,)),  # Action with params
        Action(hand_index=3, params=(0, 2)),  # Action with multiple params
    ]
    return actions


@pytest.fixture
def random_network():
    """Provide an initialized neural network (when available).

    Returns a randomly initialized actor-critic network
    ready for testing forward passes and training.

    Currently skipped - will be implemented in Phase 10.4.
    """
    pytest.skip("Network fixture not available - waiting for Phase 10.4")


@pytest.fixture
def sample_trajectory():
    """Provide mock trajectory data for testing.

    Returns a dictionary containing a complete game trajectory:
    - observations: List of observation tensors
    - actions: List of action indices
    - rewards: List of rewards
    - dones: List of terminal flags
    - action_masks: List of legal action masks

    Currently returns mock data structure.
    """
    trajectory = {
        "observations": [],  # Will be populated with tensors
        "actions": [],  # Will be populated with action indices
        "rewards": [],  # Will be populated with float rewards
        "dones": [],  # Will be populated with bool flags
        "action_masks": [],  # Will be populated with mask tensors
    }

    # TODO: Populate with realistic mock data
    # This will be properly implemented when Phase 10.6-10.7 begins

    return trajectory


@pytest.fixture
def sample_batch(sample_trajectory):
    """Provide a batch of trajectories for training.

    Returns a batched version of trajectory data:
    - observations: Tensor of shape (batch_size, seq_len, obs_dim)
    - actions: Tensor of shape (batch_size, seq_len)
    - rewards: Tensor of shape (batch_size, seq_len)
    - dones: Tensor of shape (batch_size, seq_len)
    - action_masks: Tensor of shape (batch_size, seq_len, num_actions)

    Currently returns mock structure.
    """
    batch_size = 4

    batch = {
        "observations": None,  # Will be tensor
        "actions": None,  # Will be tensor
        "rewards": None,  # Will be tensor
        "dones": None,  # Will be tensor
        "action_masks": None,  # Will be tensor
    }

    # TODO: Create proper batched tensors
    # This will be properly implemented when Phase 10.6 begins

    return batch
