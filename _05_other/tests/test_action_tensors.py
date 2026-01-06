"""Tests for action tensor conversion functions."""

import numpy as np
import pytest

from _01_simulator import action_space, actions, state


def test_action_dim_constant():
    """Verify ACTION_DIM matches catalog length."""
    assert len(action_space.canonical_actions()) == action_space.ACTION_DIM
    assert action_space.ACTION_DIM > 0


def test_action_mask_to_tensor_shape():
    """Verify tensor shape and dtype."""
    game_state = state.initial_state(seed=123)
    space = action_space.legal_action_space(game_state, 0)
    tensor = action_space.action_mask_to_tensor(space)

    assert tensor.shape == (action_space.ACTION_DIM,)
    assert tensor.dtype == np.float32


def test_action_mask_to_tensor_values():
    """Verify tensor values are 0.0 or 1.0."""
    game_state = state.initial_state(seed=123)
    space = action_space.legal_action_space(game_state, 0)
    tensor = action_space.action_mask_to_tensor(space)

    assert np.all((tensor == 0.0) | (tensor == 1.0))


def test_legal_action_mask_tensor_consistency():
    """Verify convenience function matches two-step conversion."""
    game_state = state.initial_state(seed=123)
    perspective = 0

    # Two-step conversion
    space = action_space.legal_action_space(game_state, perspective)
    tensor1 = action_space.action_mask_to_tensor(space)

    # Convenience function
    tensor2 = action_space.legal_action_mask_tensor(game_state, perspective)

    assert np.array_equal(tensor1, tensor2)


def test_batch_action_masks_shape():
    """Verify batch tensor shape."""
    states = [state.initial_state(seed=i) for i in range(5)]
    perspectives = [0, 1, 0, 1, 0]

    batch = action_space.batch_action_masks(states, perspectives)

    assert batch.shape == (5, action_space.ACTION_DIM)
    assert batch.dtype == np.float32


def test_batch_action_masks_per_row_correctness():
    """Verify each row of batch matches individual mask."""
    states = [state.initial_state(seed=i) for i in range(3)]
    perspectives = [0, 1, 0]

    batch = action_space.batch_action_masks(states, perspectives)

    for i, (game_state, perspective) in enumerate(zip(states, perspectives, strict=True)):
        expected = action_space.legal_action_mask_tensor(game_state, perspective)
        assert np.array_equal(batch[i], expected)


def test_batch_action_masks_length_mismatch():
    """Verify error on mismatched lengths."""
    states = [state.initial_state(seed=i) for i in range(3)]
    perspectives = [0, 1]  # Wrong length

    with pytest.raises(ValueError, match="must have same length"):
        action_space.batch_action_masks(states, perspectives)


def test_index_to_action_roundtrip():
    """Verify index_to_action reverses action_index."""
    # Test all actions in catalog
    for idx in range(action_space.ACTION_DIM):
        action = action_space.index_to_action(idx)
        assert action_space.action_index(action) == idx


def test_index_to_action_out_of_range():
    """Verify error on invalid index."""
    with pytest.raises(IndexError, match="out of range"):
        action_space.index_to_action(-1)

    with pytest.raises(IndexError, match="out of range"):
        action_space.index_to_action(action_space.ACTION_DIM)


def test_sample_masked_action_returns_legal():
    """Verify sampled action is legal."""
    game_state = state.initial_state(seed=123)
    mask = action_space.legal_action_mask_tensor(game_state, 0)

    # Create random logits
    logits = np.random.randn(action_space.ACTION_DIM).astype(np.float32)

    # Sample 10 times and verify all are legal
    for _ in range(10):
        idx = action_space.sample_masked_action(logits, mask)
        assert 0 <= idx < action_space.ACTION_DIM
        assert mask[idx] == 1.0


def test_sample_masked_action_temperature():
    """Verify temperature affects sampling (statistical test)."""
    mask = np.ones(action_space.ACTION_DIM, dtype=np.float32)
    logits = np.zeros(action_space.ACTION_DIM, dtype=np.float32)
    logits[0] = 10.0  # Heavily favor first action

    # With high temperature, should get some diversity
    samples_high_temp = [action_space.sample_masked_action(logits, mask, temperature=10.0) for _ in range(100)]
    unique_high = len(set(samples_high_temp))

    # With low temperature, should mostly get first action
    samples_low_temp = [action_space.sample_masked_action(logits, mask, temperature=0.1) for _ in range(100)]
    count_first = sum(1 for s in samples_low_temp if s == 0)

    # High temp should have more diversity
    assert unique_high > 5

    # Low temp should strongly favor first action
    assert count_first > 90


def test_sample_masked_action_no_legal_actions():
    """Verify error when no legal actions."""
    mask = np.zeros(action_space.ACTION_DIM, dtype=np.float32)
    logits = np.random.randn(action_space.ACTION_DIM).astype(np.float32)

    with pytest.raises(ValueError, match="No legal actions"):
        action_space.sample_masked_action(logits, mask)


def test_sample_masked_action_invalid_temperature():
    """Verify error on non-positive temperature."""
    mask = np.ones(action_space.ACTION_DIM, dtype=np.float32)
    logits = np.random.randn(action_space.ACTION_DIM).astype(np.float32)

    with pytest.raises(ValueError, match="Temperature must be positive"):
        action_space.sample_masked_action(logits, mask, temperature=0.0)

    with pytest.raises(ValueError, match="Temperature must be positive"):
        action_space.sample_masked_action(logits, mask, temperature=-1.0)


def test_greedy_masked_action_selects_highest():
    """Verify greedy selection picks highest logit."""
    mask = np.ones(action_space.ACTION_DIM, dtype=np.float32)
    logits = np.random.randn(action_space.ACTION_DIM).astype(np.float32)

    # Set a specific action to have highest logit
    highest_idx = 42
    logits[highest_idx] = 100.0

    result = action_space.greedy_masked_action(logits, mask)
    assert result == highest_idx


def test_greedy_masked_action_respects_mask():
    """Verify greedy selection only considers legal actions."""
    mask = np.zeros(action_space.ACTION_DIM, dtype=np.float32)
    logits = np.random.randn(action_space.ACTION_DIM).astype(np.float32)

    # Make first action have highest logit but illegal
    logits[0] = 100.0
    mask[0] = 0.0

    # Make second action legal with lower logit
    logits[1] = 50.0
    mask[1] = 1.0

    # Make third action legal with even lower logit
    logits[2] = 10.0
    mask[2] = 1.0

    result = action_space.greedy_masked_action(logits, mask)
    assert result == 1  # Should pick second action (highest among legal)


def test_greedy_masked_action_no_legal_actions():
    """Verify error when no legal actions."""
    mask = np.zeros(action_space.ACTION_DIM, dtype=np.float32)
    logits = np.random.randn(action_space.ACTION_DIM).astype(np.float32)

    with pytest.raises(ValueError, match="No legal actions"):
        action_space.greedy_masked_action(logits, mask)


def test_mask_alignment_with_catalog():
    """Verify mask indices align with action catalog."""
    game_state = state.initial_state(seed=123)
    space = action_space.legal_action_space(game_state, 0)
    tensor = action_space.action_mask_to_tensor(space)

    # Check that all legal indices in the space have mask=1.0
    for idx in space.legal_indices:
        assert tensor[idx] == 1.0

    # Check that count of 1.0s matches number of legal actions
    assert np.sum(tensor) == len(space.legal_indices)


def test_tensor_conversion_with_empty_hand():
    """Test mask when player has no cards (edge case)."""
    game_state = state.initial_state(seed=123)

    # Remove all cards from player 0's hand (simulate end game)
    players = list(game_state.players)
    players[0] = state.PlayerState(deck=players[0].deck, hand=())
    game_state = state.State(
        seed=game_state.seed,
        turn=game_state.turn,
        active_player=game_state.active_player,
        players=tuple(players),
        zones=game_state.zones,
    )

    space = action_space.legal_action_space(game_state, 0)
    tensor = action_space.action_mask_to_tensor(space)

    # Should have no legal actions
    assert np.sum(tensor) == 0


def test_mask_encodes_parameterized_actions():
    """Verify mask includes parameterized actions when appropriate."""
    # Create a state where Kangaroo is playable (needs queue with cards)
    game_state = state.initial_state(seed=123)

    # Get the mask and check it's working
    mask = action_space.legal_action_mask_tensor(game_state, 0)

    # At game start, some actions should be legal
    assert np.sum(mask) > 0

    # Verify we can decode legal actions
    legal_indices = np.where(mask > 0)[0]
    for idx in legal_indices:
        action = action_space.index_to_action(idx)
        # Should be valid Action objects
        assert isinstance(action, actions.Action)
        assert 0 <= action.hand_index < 4
