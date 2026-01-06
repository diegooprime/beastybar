"""
Test suite for observation tensor encoding and decoding.

Tests the conversion of game observations to neural network input tensors,
ensuring proper shape, normalization, and deterministic encoding.
"""

import numpy as np
import pytest

from _01_simulator.observations import (
    OBSERVATION_DIM,
    batch_states_to_tensor,
    build_observation,
    observation_to_tensor,
    state_to_tensor,
    tensor_to_observation,
)
from _01_simulator import state


def test_observation_tensor_dimensions():
    """Verify OBSERVATION_DIM=988 and tensor has correct shape."""
    # Create initial game state
    game_state = state.initial_state(seed=0)

    # Convert to tensor
    tensor = state_to_tensor(game_state, perspective=0)

    # Verify dimension constant
    assert OBSERVATION_DIM == 988, f"Expected OBSERVATION_DIM=988, got {OBSERVATION_DIM}"

    # Verify tensor shape
    assert tensor.shape == (988,), f"Expected shape (988,), got {tensor.shape}"
    assert tensor.dtype == np.float32, f"Expected dtype float32, got {tensor.dtype}"


def test_observation_encoding_deterministic():
    """Same state → same tensor (deterministic encoding)."""
    # Create initial game state with fixed seed
    game_state = state.initial_state(seed=42)

    # Encode same state multiple times
    tensor1 = state_to_tensor(game_state, perspective=0)
    tensor2 = state_to_tensor(game_state, perspective=0)
    tensor3 = state_to_tensor(game_state, perspective=0)

    # All encodings should be identical
    np.testing.assert_array_equal(tensor1, tensor2)
    np.testing.assert_array_equal(tensor2, tensor3)

    # Verify no stochasticity
    assert np.array_equal(tensor1, tensor2)


def test_observation_empty_zones():
    """Handle empty queue/bar/thats_it zones correctly."""
    # Create initial state where zones start empty
    game_state = state.initial_state(seed=0)

    # At game start, queue/bar/thats_it should be empty
    assert len(game_state.zones.queue) == 0
    assert len(game_state.zones.beasty_bar) == 0
    assert len(game_state.zones.thats_it) == 0

    # Encode state
    tensor = state_to_tensor(game_state, perspective=0)

    # Should not crash and produce valid tensor
    assert tensor.shape == (OBSERVATION_DIM,)
    assert np.all(np.isfinite(tensor)), "Tensor contains NaN or inf values"

    # Decode back and verify empty zones are handled
    decoded = tensor_to_observation(tensor, perspective=0)
    assert "queue" in decoded
    assert "beasty_bar" in decoded
    assert "thats_it" in decoded


def test_observation_full_zones():
    """Handle full zones correctly (stress test with max capacity)."""
    # Create state with some cards played (simulate mid-game)
    from _01_simulator import engine, simulate

    game_state = state.initial_state(seed=123)

    # Play game for several turns to create fuller zones
    for _ in range(10):
        if simulate.is_terminal(game_state):
            break
        legal = list(engine.legal_actions(game_state, game_state.active_player))
        if legal:
            action = legal[0]  # Take first legal action
            game_state = simulate.apply(game_state, action)

    # Encode state
    tensor = state_to_tensor(game_state, perspective=0)

    # Verify valid encoding
    assert tensor.shape == (OBSERVATION_DIM,)
    assert np.all(np.isfinite(tensor)), "Tensor contains NaN or inf values"
    assert np.all((tensor >= 0.0) & (tensor <= 1.0)), "Values outside [0, 1] range"


def test_batch_encoding():
    """Test batch_states_to_tensor with multiple states."""
    # Create multiple states
    states = [
        state.initial_state(seed=i)
        for i in range(5)
    ]
    perspectives = [0, 1, 0, 1, 0]

    # Batch encode
    batch_tensor = batch_states_to_tensor(states, perspectives)

    # Verify shape
    assert batch_tensor.shape == (5, OBSERVATION_DIM), \
        f"Expected shape (5, 988), got {batch_tensor.shape}"
    assert batch_tensor.dtype == np.float32

    # Verify each row matches individual encoding
    for i, (game_state, persp) in enumerate(zip(states, perspectives)):
        individual_tensor = state_to_tensor(game_state, persp)
        np.testing.assert_array_equal(batch_tensor[i], individual_tensor)


def test_encoding_decoding_roundtrip():
    """tensor_to_observation recovers key information."""
    # Create state with known properties
    game_state = state.initial_state(seed=42)
    perspective = 0

    # Build observation
    obs = build_observation(game_state, perspective)

    # Encode to tensor
    tensor = observation_to_tensor(obs, perspective)

    # Decode back
    decoded = tensor_to_observation(tensor, perspective)

    # Verify key fields are present
    assert "perspective" in decoded
    assert decoded["perspective"] == perspective
    assert "queue" in decoded
    assert "beasty_bar" in decoded
    assert "thats_it" in decoded
    assert "own_hand" in decoded
    assert "opponent_hand_count" in decoded

    # Verify scalar fields exist
    assert "is_active_player" in decoded
    assert "turn_normalized" in decoded
    assert "queue_length_normalized" in decoded


def test_observation_normalization():
    """All values in [0, 1] range."""
    # Create multiple game states at different stages
    from _01_simulator import engine, simulate

    seeds = [42, 123, 456, 789, 999]

    for seed in seeds:
        game_state = state.initial_state(seed=seed)

        # Play a few turns to get varied states
        for _ in range(5):
            if simulate.is_terminal(game_state):
                break
            legal = list(engine.legal_actions(game_state, game_state.active_player))
            if legal:
                game_state = simulate.apply(game_state, legal[0])

        # Encode from both perspectives
        for perspective in [0, 1]:
            tensor = state_to_tensor(game_state, perspective)

            # All values must be in [0, 1]
            assert np.all(tensor >= 0.0), \
                f"Tensor has negative values: min={tensor.min()}"
            assert np.all(tensor <= 1.0), \
                f"Tensor has values > 1.0: max={tensor.max()}"
            assert np.all(np.isfinite(tensor)), \
                "Tensor contains NaN or inf values"
