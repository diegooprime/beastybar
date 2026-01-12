#!/usr/bin/env python3
"""Diagnostic script to identify Cython bugs vs Python implementation."""

import sys

import numpy as np

# Try to import Cython module
try:
    from _01_simulator._cython._cython_core import (
        PY_ACTION_DIM,
        PY_OBSERVATION_DIM,
        GameStateArray,
        python_state_to_c,
    )
    CYTHON_AVAILABLE = True
    print("Cython module loaded successfully")
except ImportError as e:
    print(f"Cython module not available: {e}")
    CYTHON_AVAILABLE = False
    sys.exit(1)

# Import Python implementations
from _01_simulator import engine, simulate
from _01_simulator.action_space import action_index, legal_action_mask_tensor
from _01_simulator.observations import (
    state_to_tensor,
)


def test_initial_state():
    """Test that initial game state matches between Python and Cython."""
    print("\n" + "="*60)
    print("TEST 1: Initial State Comparison")
    print("="*60)

    seed = 42

    # Create Python state
    py_state = simulate.new_game(seed)
    print("\nPython initial state:")
    print(f"  Active player: {py_state.active_player}")
    print(f"  Turn: {py_state.turn}")
    print(f"  Player 0 hand: {len(py_state.players[0].hand)} cards")
    print(f"  Player 1 hand: {len(py_state.players[1].hand)} cards")

    # Create Cython state
    arr = GameStateArray(1)
    arr.init_game(0, seed)

    print("\nCython initial state:")
    print(f"  Active player: {arr.get_active_player(0)}")
    print(f"  Is terminal: {arr.is_terminal(0)}")

    # Check legal actions for both players
    py_legal_p0 = list(engine.legal_actions(py_state, 0))
    py_legal_p1 = list(engine.legal_actions(py_state, 1))

    print("\nPython legal actions:")
    print(f"  Player 0: {len(py_legal_p0)} actions")
    print(f"  Player 1: {len(py_legal_p1)} actions")

    # Get Cython legal mask
    cy_mask = np.zeros((1, PY_ACTION_DIM), dtype=np.float32)
    active_indices = np.array([0], dtype=np.int64)

    from _01_simulator._cython._cython_core import get_legal_masks_parallel
    get_legal_masks_parallel(arr, active_indices, cy_mask)
    cy_legal_count = int(cy_mask[0].sum())

    print("\nCython legal actions (mask):")
    print(f"  Active player (0): {cy_legal_count} actions")

    # Compare
    if len(py_legal_p0) == cy_legal_count:
        print("\n[PASS] Initial legal action counts match")
    else:
        print(f"\n[FAIL] Legal action count mismatch: Python={len(py_legal_p0)}, Cython={cy_legal_count}")

    return py_state, arr


def test_after_one_step(py_state, arr):
    """Test state after one step."""
    print("\n" + "="*60)
    print("TEST 2: State After One Step")
    print("="*60)

    # Get first legal action for player 0
    py_legal = list(engine.legal_actions(py_state, 0))
    if not py_legal:
        print("[ERROR] No legal actions for player 0")
        return None, None

    action = py_legal[0]
    print(f"\nApplying action: hand_index={action.hand_index}, params={action.params}")

    # Step Python state
    py_state2 = engine.step(py_state, action)
    print("\nPython after step:")
    print(f"  Active player: {py_state2.active_player}")
    print(f"  Turn: {py_state2.turn}")
    print(f"  Queue length: {len(py_state2.zones.queue)}")

    # Step Cython state
    from _01_simulator._cython._cython_core import step_batch_parallel
    action_idx = action_index(action)
    print(f"  Action index: {action_idx}")

    active_indices = np.array([0], dtype=np.int64)
    actions = np.array([action_idx], dtype=np.int64)
    step_batch_parallel(arr, active_indices, actions)

    print("\nCython after step:")
    print(f"  Active player: {arr.get_active_player(0)}")
    print(f"  Is terminal: {arr.is_terminal(0)}")

    # Check legal actions for player 1 (should be active now)
    py_legal_p0 = list(engine.legal_actions(py_state2, 0))
    py_legal_p1 = list(engine.legal_actions(py_state2, 1))

    print("\nPython legal actions after step:")
    print(f"  Player 0: {len(py_legal_p0)} actions (should be 0, not active)")
    print(f"  Player 1: {len(py_legal_p1)} actions (should be >0, is active)")

    # Get Cython legal mask for new active player
    cy_mask = np.zeros((1, PY_ACTION_DIM), dtype=np.float32)
    from _01_simulator._cython._cython_core import get_legal_masks_parallel
    get_legal_masks_parallel(arr, active_indices, cy_mask)
    cy_legal_count = int(cy_mask[0].sum())

    print("\nCython legal actions after step:")
    print(f"  Legal action count: {cy_legal_count}")

    if py_state2.active_player == arr.get_active_player(0):
        print("\n[PASS] Active player matches after step")
    else:
        print(f"\n[FAIL] Active player mismatch: Python={py_state2.active_player}, Cython={arr.get_active_player(0)}")

    if len(py_legal_p1) == cy_legal_count:
        print("[PASS] Legal action counts match after step")
    else:
        print(f"[FAIL] Legal action count mismatch: Python={len(py_legal_p1)}, Cython={cy_legal_count}")

    return py_state2, arr


def test_observation_encoding(py_state):
    """Test observation encoding matches between Python and Cython."""
    print("\n" + "="*60)
    print("TEST 3: Observation Encoding")
    print("="*60)

    # Get Python observation tensor
    py_tensor = state_to_tensor(py_state, perspective=0)
    print("\nPython observation:")
    print(f"  Shape: {py_tensor.shape}")
    print(f"  Non-zero count: {np.count_nonzero(py_tensor)}")
    print(f"  Min: {py_tensor.min():.4f}, Max: {py_tensor.max():.4f}")

    # Convert to Cython and get observation
    arr = GameStateArray(1)
    python_state_to_c(py_state, arr, 0)

    from _01_simulator._cython._cython_core import encode_observations_parallel
    cy_obs = np.zeros((1, PY_OBSERVATION_DIM), dtype=np.float32)
    active_indices = np.array([0], dtype=np.int64)
    encode_observations_parallel(arr, active_indices, cy_obs)

    cy_tensor = cy_obs[0]
    print("\nCython observation:")
    print(f"  Shape: {cy_tensor.shape}")
    print(f"  Non-zero count: {np.count_nonzero(cy_tensor)}")
    print(f"  Min: {cy_tensor.min():.4f}, Max: {cy_tensor.max():.4f}")

    # Compare
    diff = np.abs(py_tensor - cy_tensor)
    max_diff = diff.max()
    diff_count = np.sum(diff > 0.001)

    print("\nComparison:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Elements with diff > 0.001: {diff_count}")

    if diff_count > 0:
        print("\n[FAIL] Observation encoding mismatch")
        # Find first mismatch
        mismatch_indices = np.where(diff > 0.001)[0][:5]
        for idx in mismatch_indices:
            print(f"  Index {idx}: Python={py_tensor[idx]:.4f}, Cython={cy_tensor[idx]:.4f}")
    else:
        print("\n[PASS] Observation encoding matches")

    return py_tensor, cy_tensor


def test_legal_mask_comparison(py_state):
    """Test legal action mask matches."""
    print("\n" + "="*60)
    print("TEST 4: Legal Action Mask Comparison")
    print("="*60)

    player = py_state.active_player

    # Get Python legal mask
    py_mask = legal_action_mask_tensor(py_state, player)
    print(f"\nPython legal mask (player {player}):")
    print(f"  Legal actions: {int(py_mask.sum())}")
    print(f"  Legal indices: {np.where(py_mask > 0)[0][:10]}...")

    # Get Cython legal mask
    arr = GameStateArray(1)
    python_state_to_c(py_state, arr, 0)

    cy_mask = np.zeros((1, PY_ACTION_DIM), dtype=np.float32)
    active_indices = np.array([0], dtype=np.int64)
    from _01_simulator._cython._cython_core import get_legal_masks_parallel
    get_legal_masks_parallel(arr, active_indices, cy_mask)

    print("\nCython legal mask:")
    print(f"  Legal actions: {int(cy_mask[0].sum())}")
    print(f"  Legal indices: {np.where(cy_mask[0] > 0)[0][:10]}...")

    # Compare
    diff = np.abs(py_mask - cy_mask[0])
    mismatch_count = np.sum(diff > 0)

    if mismatch_count > 0:
        print(f"\n[FAIL] Legal mask mismatch: {mismatch_count} differences")
        py_only = np.where((py_mask > 0) & (cy_mask[0] == 0))[0]
        cy_only = np.where((py_mask == 0) & (cy_mask[0] > 0))[0]
        if len(py_only) > 0:
            print(f"  Python only (missing in Cython): {py_only[:10]}")
        if len(cy_only) > 0:
            print(f"  Cython only (extra): {cy_only[:10]}")
    else:
        print("\n[PASS] Legal action masks match")


def main():
    """Run all diagnostic tests."""
    print("="*60)
    print("CYTHON BUG DIAGNOSTIC")
    print("="*60)

    # Test 1: Initial state
    py_state, arr = test_initial_state()

    # Test 2: After one step
    _py_state2, _arr2 = test_after_one_step(py_state, arr)

    # Test 3: Observation encoding
    test_observation_encoding(py_state)

    # Test 4: Legal mask
    test_legal_mask_comparison(py_state)

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
