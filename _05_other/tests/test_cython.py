"""Tests for Cython-accelerated game simulation.

These tests verify that the Cython implementation produces identical
results to the pure Python implementation.
"""

import numpy as np
import pytest

from _01_simulator.action_space import ACTION_DIM, legal_action_mask_tensor
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor
from _01_simulator.state import initial_state

# Try to import Cython module
try:
    from _01_simulator._cython._cython_core import (
        encode_single_observation,
        get_single_legal_mask,
        get_single_scores,
        step_single,
    )

    from _01_simulator._cython import (
        GameStateArray,
        is_cython_available,
        python_state_to_c,
    )

    CYTHON_AVAILABLE = is_cython_available()
except ImportError:
    CYTHON_AVAILABLE = False


pytestmark = pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not available")


class TestGameStateConversion:
    """Test conversion between Python State and C GameState."""

    def test_state_conversion_preserves_metadata(self):
        """Test that state metadata is preserved during conversion."""
        py_state = initial_state(seed=42)
        c_states = GameStateArray(1)
        python_state_to_c(py_state, c_states, 0)

        assert c_states.get_active_player(0) == py_state.active_player
        assert not c_states.is_terminal(0)

    def test_state_conversion_multiple_seeds(self):
        """Test conversion with multiple different seeds."""
        seeds = [42, 123, 456, 789, 1000]
        c_states = GameStateArray(len(seeds))

        for i, seed in enumerate(seeds):
            py_state = initial_state(seed=seed)
            python_state_to_c(py_state, c_states, i)

        for i in range(len(seeds)):
            assert c_states.get_active_player(i) == 0
            assert not c_states.is_terminal(i)


class TestObservationEncoding:
    """Test observation encoding matches Python implementation."""

    def test_observation_dimensions(self):
        """Test that observation has correct dimensions."""
        c_states = GameStateArray(1)
        c_states.init_game(0, 42)

        obs = encode_single_observation(c_states, 0)
        assert obs.shape == (OBSERVATION_DIM,)
        assert obs.dtype == np.float32

    def test_observation_values_in_range(self):
        """Test that observation values are in [0, 1] range."""
        c_states = GameStateArray(1)
        c_states.init_game(0, 42)

        obs = encode_single_observation(c_states, 0)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_observation_matches_python(self):
        """Test that Cython observation matches Python implementation."""
        seed = 42
        py_state = initial_state(seed=seed)

        # Get Python observation
        py_obs = state_to_tensor(py_state, py_state.active_player)

        # Get Cython observation
        c_states = GameStateArray(1)
        python_state_to_c(py_state, c_states, 0)
        c_obs = encode_single_observation(c_states, 0)

        # Compare
        np.testing.assert_allclose(c_obs, py_obs, rtol=1e-5, atol=1e-6)

    def test_observation_both_perspectives(self):
        """Test observation from both player perspectives."""
        seed = 42
        py_state = initial_state(seed=seed)

        c_states = GameStateArray(1)
        python_state_to_c(py_state, c_states, 0)

        for perspective in [0, 1]:
            py_obs = state_to_tensor(py_state, perspective)
            c_obs = encode_single_observation(c_states, 0, perspective)
            np.testing.assert_allclose(c_obs, py_obs, rtol=1e-5, atol=1e-6)


class TestLegalActionMask:
    """Test legal action mask generation."""

    def test_mask_dimensions(self):
        """Test that mask has correct dimensions."""
        c_states = GameStateArray(1)
        c_states.init_game(0, 42)

        mask = get_single_legal_mask(c_states, 0)
        assert mask.shape == (ACTION_DIM,)
        assert mask.dtype == np.float32

    def test_mask_values_binary(self):
        """Test that mask values are 0 or 1."""
        c_states = GameStateArray(1)
        c_states.init_game(0, 42)

        mask = get_single_legal_mask(c_states, 0)
        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_mask_has_legal_actions(self):
        """Test that mask has at least one legal action."""
        c_states = GameStateArray(1)
        c_states.init_game(0, 42)

        mask = get_single_legal_mask(c_states, 0)
        assert np.sum(mask) > 0

    def test_mask_matches_python(self):
        """Test that Cython mask matches Python implementation."""
        seed = 42
        py_state = initial_state(seed=seed)

        # Get Python mask
        py_mask = legal_action_mask_tensor(py_state, py_state.active_player)

        # Get Cython mask
        c_states = GameStateArray(1)
        python_state_to_c(py_state, c_states, 0)
        c_mask = get_single_legal_mask(c_states, 0)

        # Compare
        np.testing.assert_array_equal(c_mask, py_mask)


class TestGameStep:
    """Test game stepping matches Python implementation."""

    def test_step_changes_state(self):
        """Test that stepping changes game state."""
        c_states = GameStateArray(1)
        c_states.init_game(0, 42)

        initial_player = c_states.get_active_player(0)

        # Get a legal action
        mask = get_single_legal_mask(c_states, 0)
        action_idx = int(np.argmax(mask))

        step_single(c_states, 0, action_idx)

        # Active player should have changed
        new_player = c_states.get_active_player(0)
        assert new_player != initial_player

    def test_full_game_terminates(self):
        """Test that a full game eventually terminates."""
        c_states = GameStateArray(1)
        c_states.init_game(0, 42)

        max_steps = 100
        steps = 0

        while not c_states.is_terminal(0) and steps < max_steps:
            mask = get_single_legal_mask(c_states, 0)
            action_idx = int(np.argmax(mask))
            step_single(c_states, 0, action_idx)
            steps += 1

        assert c_states.is_terminal(0), f"Game did not terminate after {max_steps} steps"

    def test_scores_valid(self):
        """Test that final scores are valid."""
        c_states = GameStateArray(1)
        c_states.init_game(0, 42)

        # Play until terminal
        max_steps = 100
        steps = 0
        while not c_states.is_terminal(0) and steps < max_steps:
            mask = get_single_legal_mask(c_states, 0)
            action_idx = int(np.argmax(mask))
            step_single(c_states, 0, action_idx)
            steps += 1

        scores = get_single_scores(c_states, 0)
        assert len(scores) == 2
        assert all(0 <= s <= 48 for s in scores)  # Max possible score is 48 (all 12 cards * 4 points)


class TestGameConsistency:
    """Test that Cython games are consistent with Python games."""

    def test_deterministic_with_same_seed(self):
        """Test that same seed produces same game."""
        seed = 42

        # Play game with Cython
        c_states1 = GameStateArray(1)
        c_states1.init_game(0, seed)
        actions1 = []

        while not c_states1.is_terminal(0):
            mask = get_single_legal_mask(c_states1, 0)
            action_idx = int(np.argmax(mask))
            actions1.append(action_idx)
            step_single(c_states1, 0, action_idx)

        # Play same game again
        c_states2 = GameStateArray(1)
        c_states2.init_game(0, seed)
        actions2 = []

        while not c_states2.is_terminal(0):
            mask = get_single_legal_mask(c_states2, 0)
            action_idx = int(np.argmax(mask))
            actions2.append(action_idx)
            step_single(c_states2, 0, action_idx)

        # Should take same actions
        assert actions1 == actions2

        # Should have same scores
        scores1 = get_single_scores(c_states1, 0)
        scores2 = get_single_scores(c_states2, 0)
        assert scores1 == scores2

    def test_different_seeds_different_games(self):
        """Test that different seeds produce different games."""
        c_states = GameStateArray(2)
        c_states.init_game(0, 42)
        c_states.init_game(1, 123)

        obs1 = encode_single_observation(c_states, 0)
        obs2 = encode_single_observation(c_states, 1)

        # Observations should be different (different initial hands)
        assert not np.allclose(obs1, obs2)


class TestBatchOperations:
    """Test batch operations with multiple games."""

    def test_batch_init(self):
        """Test initializing multiple games."""
        num_games = 10
        c_states = GameStateArray(num_games)

        for i in range(num_games):
            c_states.init_game(i, i * 100)

        assert c_states.length == num_games

        for i in range(num_games):
            assert not c_states.is_terminal(i)
            assert c_states.get_active_player(i) == 0

    def test_parallel_observations(self):
        """Test parallel observation encoding."""
        from _01_simulator._cython import encode_observations_parallel

        num_games = 8
        c_states = GameStateArray(num_games)

        for i in range(num_games):
            c_states.init_game(i, i * 100)

        indices = np.arange(num_games, dtype=np.int64)
        output = np.zeros((num_games, OBSERVATION_DIM), dtype=np.float32)

        encode_observations_parallel(c_states, indices, output, num_threads=4)

        # Check all observations are valid
        for i in range(num_games):
            assert np.all(output[i] >= 0.0)
            assert np.all(output[i] <= 1.0)
            assert np.any(output[i] > 0.0)  # Not all zeros

    def test_parallel_masks(self):
        """Test parallel legal action mask generation."""
        from _01_simulator._cython import get_legal_masks_parallel

        num_games = 8
        c_states = GameStateArray(num_games)

        for i in range(num_games):
            c_states.init_game(i, i * 100)

        indices = np.arange(num_games, dtype=np.int64)
        output = np.zeros((num_games, ACTION_DIM), dtype=np.float32)

        get_legal_masks_parallel(c_states, indices, output, num_threads=4)

        # Check all masks are valid
        for i in range(num_games):
            assert np.all((output[i] == 0.0) | (output[i] == 1.0))
            assert np.sum(output[i]) > 0  # At least one legal action

    def test_parallel_step(self):
        """Test parallel game stepping."""
        from _01_simulator._cython import get_legal_masks_parallel, step_batch_parallel

        num_games = 8
        c_states = GameStateArray(num_games)

        for i in range(num_games):
            c_states.init_game(i, i * 100)

        indices = np.arange(num_games, dtype=np.int64)
        masks = np.zeros((num_games, ACTION_DIM), dtype=np.float32)

        get_legal_masks_parallel(c_states, indices, masks, num_threads=4)

        # Get first legal action for each game
        actions = np.array([int(np.argmax(masks[i])) for i in range(num_games)], dtype=np.int64)

        # Step all games
        step_batch_parallel(c_states, indices, actions, num_threads=4)

        # All players should have switched
        for i in range(num_games):
            assert c_states.get_active_player(i) == 1
