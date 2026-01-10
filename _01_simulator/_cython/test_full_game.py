#!/usr/bin/env python3
"""Test full game simulation matches between Python and Cython."""

import numpy as np
from _01_simulator import simulate, engine
from _01_simulator.action_space import action_index, index_to_action
from _01_simulator._cython._cython_core import (
    GameStateArray,
    step_batch_parallel,
    get_legal_masks_parallel,
    encode_observations_parallel,
    PY_ACTION_DIM,
    PY_OBSERVATION_DIM,
)


def play_full_game(seed: int) -> tuple[list, list]:
    """Play a full game using both Python and Cython engines."""
    from _01_simulator._cython._cython_core import python_state_to_c

    # Initialize Python state
    py_state = simulate.new_game(seed)

    # Initialize Cython state FROM Python state (not native init)
    arr = GameStateArray(1)
    python_state_to_c(py_state, arr, 0)

    py_moves = []
    cy_moves = []

    move_count = 0
    max_moves = 100

    while not simulate.is_terminal(py_state) and move_count < max_moves:
        player = py_state.active_player

        # Get Python legal actions
        py_legal = list(engine.legal_actions(py_state, player))

        # Get Cython legal mask
        cy_mask = np.zeros((1, PY_ACTION_DIM), dtype=np.float32)
        active_indices = np.array([0], dtype=np.int64)
        get_legal_masks_parallel(arr, active_indices, cy_mask)

        cy_legal_indices = np.where(cy_mask[0] > 0)[0]
        py_legal_indices = sorted([action_index(a) for a in py_legal])

        # Compare legal actions
        if list(cy_legal_indices) != list(py_legal_indices):
            print(f"Move {move_count}: Legal action mismatch!")
            print(f"  Python: {py_legal_indices}")
            print(f"  Cython: {list(cy_legal_indices)}")
            return py_moves, cy_moves

        if not py_legal:
            print(f"Move {move_count}: No legal actions available (game should be terminal)")
            break

        # Choose first action
        action = py_legal[0]
        action_idx = action_index(action)

        py_moves.append(action_idx)
        cy_moves.append(action_idx)

        # Step Python
        py_state = engine.step(py_state, action)

        # Step Cython
        actions = np.array([action_idx], dtype=np.int64)
        step_batch_parallel(arr, active_indices, actions)

        # Compare observations
        py_obs = np.zeros(PY_OBSERVATION_DIM, dtype=np.float32)
        from _01_simulator.observations import state_to_tensor
        py_obs = state_to_tensor(py_state, py_state.active_player)

        cy_obs = np.zeros((1, PY_OBSERVATION_DIM), dtype=np.float32)
        encode_observations_parallel(arr, active_indices, cy_obs)

        diff = np.abs(py_obs - cy_obs[0]).max()
        if diff > 0.01:
            print(f"Move {move_count}: Observation mismatch! Max diff: {diff}")
            return py_moves, cy_moves

        move_count += 1

    # Compare final scores
    py_scores = simulate.score(py_state)
    cy_scores = arr.get_scores(0)

    return py_moves, cy_moves, py_scores, cy_scores, move_count


def main():
    print("="*60)
    print("FULL GAME COMPARISON TEST")
    print("="*60)

    num_games = 100
    passed = 0

    for seed in range(num_games):
        result = play_full_game(seed)
        if len(result) == 5:
            py_moves, cy_moves, py_scores, cy_scores, moves = result

            if py_scores == cy_scores and py_moves == cy_moves:
                print(f"Game {seed}: PASS ({moves} moves, scores {py_scores})")
                passed += 1
            else:
                print(f"Game {seed}: FAIL")
                print(f"  Python scores: {py_scores}")
                print(f"  Cython scores: {cy_scores}")
        else:
            print(f"Game {seed}: FAIL (early termination)")

    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{num_games} games passed")
    print("="*60)

    if passed == num_games:
        print("\n✅ Cython engine is READY FOR PRODUCTION")
    else:
        print(f"\n❌ {num_games - passed} games failed - needs debugging")


if __name__ == "__main__":
    main()
