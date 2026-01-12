"""
Test suite for action masking and action space encoding.

Tests the conversion of legal actions to binary masks, action sampling,
and the mapping between action indices and game actions.
"""

import numpy as np
import pytest

from _01_simulator import state
from _01_simulator.action_space import (
    ACTION_DIM,
    action_index,
    batch_action_masks,
    greedy_masked_action,
    index_to_action,
    legal_action_mask_tensor,
    sample_masked_action,
)
from _01_simulator.engine import legal_actions


def test_action_mask_dimensions():
    """Verify ACTION_DIM=124 and mask has correct shape."""
    # Create initial game state
    game_state = state.initial_state(seed=0)

    # Get action mask
    mask = legal_action_mask_tensor(game_state, perspective=0)

    # Verify dimension constant
    assert ACTION_DIM == 124, f"Expected ACTION_DIM=124, got {ACTION_DIM}"

    # Verify mask shape
    assert mask.shape == (124,), f"Expected shape (124,), got {mask.shape}"
    assert mask.dtype == np.float32, f"Expected dtype float32, got {mask.dtype}"


def test_mask_legal_actions_only():
    """Masked actions (1.0) are actually legal."""
    # Create game state
    game_state = state.initial_state(seed=42)
    perspective = 0

    # Get legal actions from engine
    legal = list(legal_actions(game_state, perspective))
    legal_indices = {action_index(action) for action in legal}

    # Get action mask
    mask = legal_action_mask_tensor(game_state, perspective)

    # Find where mask is 1.0 (legal)
    masked_indices = set(np.where(mask == 1.0)[0])

    # Verify all masked actions are legal
    assert masked_indices == legal_indices, \
        f"Mask mismatch: mask has {masked_indices}, expected {legal_indices}"

    # Verify mask is binary (only 0.0 or 1.0)
    unique_values = set(mask)
    assert unique_values.issubset({0.0, 1.0}), \
        f"Mask contains non-binary values: {unique_values}"


def test_sample_masked_action():
    """Sampling respects mask (never samples illegal actions)."""
    # Create game state
    game_state = state.initial_state(seed=123)
    perspective = 0

    # Get legal actions
    legal = list(legal_actions(game_state, perspective))
    legal_indices = {action_index(action) for action in legal}

    # Get action mask
    mask = legal_action_mask_tensor(game_state, perspective)

    # Create random logits
    np.random.seed(456)
    logits = np.random.randn(ACTION_DIM).astype(np.float32)

    # Sample many times and verify all samples are legal
    samples = set()
    for _ in range(100):
        sampled_idx = sample_masked_action(logits, mask, temperature=1.0)
        samples.add(sampled_idx)
        assert sampled_idx in legal_indices, \
            f"Sampled illegal action index: {sampled_idx} not in {legal_indices}"

    # With 100 samples, we should see some diversity (at least 2 different actions)
    # unless there's only 1 legal action
    if len(legal_indices) > 1:
        assert len(samples) > 1, "Sampling should produce diverse actions"


def test_greedy_masked_action():
    """Greedy returns highest legal action logit."""
    # Create game state
    game_state = state.initial_state(seed=789)
    perspective = 0

    # Get legal actions
    legal = list(legal_actions(game_state, perspective))
    legal_indices = [action_index(action) for action in legal]

    # Get action mask
    mask = legal_action_mask_tensor(game_state, perspective)

    # Create logits with known maximum
    logits = np.random.randn(ACTION_DIM).astype(np.float32)

    # Set a specific legal action to have highest logit
    target_legal_idx = legal_indices[0]
    logits[target_legal_idx] = 100.0  # Make it clearly the highest

    # Greedy selection should return this action
    greedy_idx = greedy_masked_action(logits, mask)
    assert greedy_idx == target_legal_idx, \
        f"Greedy selected {greedy_idx}, expected {target_legal_idx}"

    # Verify greedy never selects illegal actions
    for _ in range(10):
        random_logits = np.random.randn(ACTION_DIM).astype(np.float32)
        greedy_idx = greedy_masked_action(random_logits, mask)
        assert greedy_idx in legal_indices, \
            f"Greedy selected illegal action: {greedy_idx}"


def test_temperature_scaling():
    """Higher temperature = more random sampling."""
    # Create game state with multiple legal actions
    game_state = state.initial_state(seed=999)
    perspective = 0

    # Get action mask
    mask = legal_action_mask_tensor(game_state, perspective)
    legal_indices = list(np.where(mask == 1.0)[0])

    # Skip test if only 1 legal action
    if len(legal_indices) <= 1:
        pytest.skip("Test requires multiple legal actions")

    # Create logits with clear preference
    logits = np.zeros(ACTION_DIM, dtype=np.float32)
    logits[legal_indices[0]] = 10.0  # Strong preference for first legal action
    for idx in legal_indices[1:]:
        logits[idx] = 0.0

    # Sample with low temperature (should be more deterministic)
    low_temp_samples = []
    for _ in range(50):
        idx = sample_masked_action(logits, mask, temperature=0.1)
        low_temp_samples.append(idx)

    # Sample with high temperature (should be more random)
    high_temp_samples = []
    for _ in range(50):
        idx = sample_masked_action(logits, mask, temperature=10.0)
        high_temp_samples.append(idx)

    # Low temp should concentrate on highest logit action
    low_temp_unique = len(set(low_temp_samples))
    high_temp_unique = len(set(high_temp_samples))

    # High temperature should explore more actions
    assert high_temp_unique >= low_temp_unique, \
        f"High temp diversity ({high_temp_unique}) should be >= low temp ({low_temp_unique})"


def test_batch_action_masks():
    """Batched mask generation."""
    # Create multiple states
    states = [state.initial_state(seed=i) for i in range(5)]
    perspectives = [0, 1, 0, 1, 0]

    # Generate batch
    batch_masks = batch_action_masks(states, perspectives)

    # Verify shape
    assert batch_masks.shape == (5, ACTION_DIM), \
        f"Expected shape (5, 124), got {batch_masks.shape}"
    assert batch_masks.dtype == np.float32

    # Verify each row matches individual mask
    for i, (game_state, persp) in enumerate(zip(states, perspectives, strict=False)):
        individual_mask = legal_action_mask_tensor(game_state, persp)
        np.testing.assert_array_equal(batch_masks[i], individual_mask)

    # Verify all masks are binary
    unique_values = set(batch_masks.flatten())
    assert unique_values.issubset({0.0, 1.0}), \
        f"Batch masks contain non-binary values: {unique_values}"


def test_action_index_roundtrip():
    """action_index ↔ index_to_action bijection."""
    # Test all valid indices
    for idx in range(ACTION_DIM):
        # Convert index to action
        action = index_to_action(idx)

        # Convert back to index
        recovered_idx = action_index(action)

        # Should be identical
        assert recovered_idx == idx, \
            f"Roundtrip failed: {idx} -> {action} -> {recovered_idx}"

    # Test with actual game actions
    game_state = state.initial_state(seed=42)
    legal = list(legal_actions(game_state, 0))

    for action in legal:
        # Convert to index
        idx = action_index(action)

        # Verify index is valid
        assert 0 <= idx < ACTION_DIM, \
            f"Invalid index {idx} for action {action}"

        # Convert back
        recovered_action = index_to_action(idx)

        # Should match
        assert recovered_action.hand_index == action.hand_index
        assert recovered_action.params == action.params
