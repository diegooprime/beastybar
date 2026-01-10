#!/usr/bin/env python3
"""Targeted edge case tests for Cython implementation.

These tests exercise specific code paths that may not be covered by random game seeds.
"""

import sys
import numpy as np
from _01_simulator import simulate, engine, state, cards, rules, actions
from _01_simulator.action_space import action_index, index_to_action, ACTION_DIM
from _01_simulator.observations import state_to_tensor, OBSERVATION_DIM

try:
    from _01_simulator._cython._cython_core import (
        GameStateArray,
        python_state_to_c,
        encode_single_observation,
        get_single_legal_mask,
        step_single,
        get_single_scores,
        PY_OBSERVATION_DIM,
        PY_ACTION_DIM,
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Cython module not available")
    sys.exit(1)


def create_test_state(
    queue_species: list[tuple[str, int]],  # (species, owner)
    hand0_species: list[str],
    hand1_species: list[str],
    deck0_species: list[str] = None,
    deck1_species: list[str] = None,
    active_player: int = 0,
    turn: int = 0,
):
    """Create a custom game state for testing."""
    deck0_species = deck0_species or []
    deck1_species = deck1_species or []

    # Create cards for queue
    queue_cards = tuple(
        state.Card(owner=owner, species=species, entered_turn=i - 10)
        for i, (species, owner) in enumerate(queue_species)
    )

    # Create cards for hands
    hand0 = tuple(state.Card(owner=0, species=s) for s in hand0_species)
    hand1 = tuple(state.Card(owner=1, species=s) for s in hand1_species)

    # Create cards for decks
    deck0 = tuple(state.Card(owner=0, species=s) for s in deck0_species)
    deck1 = tuple(state.Card(owner=1, species=s) for s in deck1_species)

    zones = state.Zones(queue=queue_cards)
    players = (
        state.PlayerState(deck=deck0, hand=hand0),
        state.PlayerState(deck=deck1, hand=hand1),
    )

    return state.State(
        seed=0,
        turn=turn,
        active_player=active_player,
        players=players,
        zones=zones,
    )


def compare_step(py_state, action):
    """Compare Python and Cython step results."""
    # Step Python
    py_result = engine.step(py_state, action)

    # Convert to Cython
    arr = GameStateArray(1)
    python_state_to_c(py_state, arr, 0)

    # Step Cython
    action_idx = action_index(action)
    step_single(arr, 0, action_idx)

    # Compare observations
    py_obs = state_to_tensor(py_result, py_result.active_player)
    cy_obs = encode_single_observation(arr, 0)

    obs_match = np.allclose(py_obs, cy_obs, atol=0.01)

    # Compare legal actions
    py_legal = sorted([action_index(a) for a in engine.legal_actions(py_result, py_result.active_player)])
    cy_mask = get_single_legal_mask(arr, 0)
    cy_legal = sorted(np.where(cy_mask > 0)[0].tolist())

    legal_match = py_legal == cy_legal

    return obs_match, legal_match, py_result, arr


def test_chameleon_copies_lion_into_multiple_lions():
    """Test chameleon copying lion when another lion exists (should go to thats_it)."""
    print("\nTest: Chameleon copies lion into multiple lions")

    # Queue has a lion, player plays chameleon targeting lion
    py_state = create_test_state(
        queue_species=[("lion", 1)],
        hand0_species=["chameleon"],
        hand1_species=["zebra"],
    )

    # Play chameleon copying the lion (target index 0)
    action = actions.Action(hand_index=0, params=(0,))
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    # After chameleon copies lion with another lion present:
    # - Chameleon-as-lion counts as 2nd lion, should go to thats_it
    # - Only the original lion should remain in queue

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_chameleon_copies_parrot():
    """Test chameleon copying parrot (needs 2 params: target, then parrot's target)."""
    print("\nTest: Chameleon copies parrot")

    py_state = create_test_state(
        queue_species=[("zebra", 1), ("parrot", 1)],
        hand0_species=["chameleon"],
        hand1_species=["zebra"],
    )

    # Play chameleon copying parrot, targeting zebra at index 0
    # params = (1, 0) -> copy parrot at index 1, then use parrot to send card at index 0 to thats_it
    action = actions.Action(hand_index=0, params=(1, 0))
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_chameleon_copies_kangaroo():
    """Test chameleon copying kangaroo (needs 2 params: target, hop distance)."""
    print("\nTest: Chameleon copies kangaroo")

    py_state = create_test_state(
        queue_species=[("zebra", 1), ("kangaroo", 1)],
        hand0_species=["chameleon"],
        hand1_species=["zebra"],
    )

    # Play chameleon copying kangaroo with hop=1
    action = actions.Action(hand_index=0, params=(1, 1))
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_chameleon_copies_monkey_triggers_ability():
    """Test chameleon copying monkey when 2+ monkeys (should expel hippos/crocs)."""
    print("\nTest: Chameleon copies monkey, triggers ability")

    py_state = create_test_state(
        queue_species=[("hippo", 1), ("monkey", 1)],  # Hippo and monkey in queue
        hand0_species=["chameleon"],  # Chameleon will copy monkey
        hand1_species=["zebra"],
    )

    # Chameleon copies monkey -> now 2 monkeys (chameleon-as-monkey + real monkey)
    # This should trigger monkey ability: expel hippos
    action = actions.Action(hand_index=0, params=(1,))  # Copy monkey at index 1
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_skunk_top_two_species():
    """Test skunk expelling top 2 strength species."""
    print("\nTest: Skunk expels top 2 species")

    py_state = create_test_state(
        queue_species=[
            ("lion", 1),      # strength 12 -> expelled
            ("hippo", 1),     # strength 11 -> expelled
            ("zebra", 1),     # strength 7 -> kept
            ("monkey", 1),    # strength 4 -> kept
        ],
        hand0_species=["skunk"],
        hand1_species=["zebra"],
    )

    action = actions.Action(hand_index=0)
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    # After skunk: lion and hippo should be in thats_it, zebra and monkey remain

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_seal_reverses_queue():
    """Test seal reversing queue order."""
    print("\nTest: Seal reverses queue")

    py_state = create_test_state(
        queue_species=[
            ("lion", 0),
            ("hippo", 1),
            ("zebra", 0),
            ("monkey", 1),
        ],
        hand0_species=["seal"],
        hand1_species=["zebra"],
    )

    action = actions.Action(hand_index=0)
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_snake_sorts_by_strength():
    """Test snake sorting queue by strength descending."""
    print("\nTest: Snake sorts by strength")

    py_state = create_test_state(
        queue_species=[
            ("monkey", 0),    # strength 4
            ("lion", 1),      # strength 12
            ("zebra", 0),     # strength 7
            ("hippo", 1),     # strength 11
        ],
        hand0_species=["snake"],
        hand1_species=["zebra"],
    )

    action = actions.Action(hand_index=0)
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    # After snake: should be lion(12), hippo(11), snake(9), zebra(7), monkey(4)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_crocodile_eats_weaker():
    """Test crocodile eating weaker cards ahead."""
    print("\nTest: Crocodile eats weaker")

    py_state = create_test_state(
        queue_species=[
            ("monkey", 1),    # strength 4 -> eaten
            ("kangaroo", 1),  # strength 3 -> eaten
            ("parrot", 1),    # strength 2 -> eaten
        ],
        hand0_species=["crocodile"],
        hand1_species=["zebra"],
    )

    action = actions.Action(hand_index=0)
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_crocodile_blocked_by_zebra():
    """Test crocodile blocked by zebra."""
    print("\nTest: Crocodile blocked by zebra")

    py_state = create_test_state(
        queue_species=[
            ("monkey", 1),    # strength 4 -> eaten
            ("zebra", 1),     # blocks
            ("parrot", 1),    # protected by zebra
        ],
        hand0_species=["crocodile"],
        hand1_species=["zebra"],
    )

    action = actions.Action(hand_index=0)
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_hippo_pushes_forward():
    """Test hippo pushing past weaker cards."""
    print("\nTest: Hippo pushes forward")

    py_state = create_test_state(
        queue_species=[
            ("monkey", 1),
            ("kangaroo", 1),
            ("parrot", 1),
        ],
        hand0_species=["hippo"],
        hand1_species=["zebra"],
    )

    action = actions.Action(hand_index=0)
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_giraffe_steps_forward():
    """Test giraffe stepping one position forward."""
    print("\nTest: Giraffe steps forward")

    py_state = create_test_state(
        queue_species=[
            ("monkey", 1),   # weaker than giraffe
            ("kangaroo", 1),
        ],
        hand0_species=["giraffe"],
        hand1_species=["zebra"],
    )

    action = actions.Action(hand_index=0)
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_kangaroo_hops():
    """Test kangaroo hopping forward."""
    print("\nTest: Kangaroo hops")

    py_state = create_test_state(
        queue_species=[
            ("zebra", 1),
            ("monkey", 1),
        ],
        hand0_species=["kangaroo"],
        hand1_species=["zebra"],
    )

    # Hop 2 positions
    action = actions.Action(hand_index=0, params=(2,))
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_five_card_check():
    """Test five card check (2 enter beasty bar, 1 bounces)."""
    print("\nTest: Five card check")

    py_state = create_test_state(
        queue_species=[
            ("lion", 0),      # -> beasty bar
            ("hippo", 1),     # -> beasty bar
            ("zebra", 0),     # stays
            ("monkey", 1),    # -> thats_it (bounce)
        ],
        hand0_species=["kangaroo"],
        hand1_species=["zebra"],
    )

    # Playing kangaroo will make queue length 5
    action = actions.Action(hand_index=0)
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_recurring_giraffe():
    """Test recurring giraffe steps forward each turn (except entry turn)."""
    print("\nTest: Recurring giraffe")

    # Giraffe already in queue from previous turn
    # Need enough cards to not be terminal (< 5 total cards is terminal)
    py_state = create_test_state(
        queue_species=[
            ("monkey", 1),
            ("giraffe", 0),   # entered_turn = -8 (not this turn)
        ],
        hand0_species=["zebra", "lion", "hippo", "snake"],
        hand1_species=["zebra", "lion", "hippo", "snake"],
        deck0_species=["seal", "crocodile"],
        deck1_species=["seal", "crocodile"],
        turn=1,
    )

    # Use reflection to set the giraffe's entered_turn to previous turn
    queue = list(py_state.zones.queue)
    giraffe_card = queue[1]
    # Create new card with entered_turn = 0 (previous turn)
    queue[1] = state.Card(owner=giraffe_card.owner, species="giraffe", entered_turn=0)
    py_state = state.replace_queue(py_state, queue)

    action = actions.Action(hand_index=0)
    obs_ok, legal_ok, py_result, arr = compare_step(py_state, action)

    if obs_ok and legal_ok:
        print("  PASS: Observations and legal actions match")
    else:
        print(f"  FAIL: obs_match={obs_ok}, legal_match={legal_ok}")

    return obs_ok and legal_ok


def test_empty_queue_legal_actions():
    """Test legal actions with empty queue."""
    print("\nTest: Empty queue legal actions")

    py_state = create_test_state(
        queue_species=[],
        hand0_species=["chameleon", "parrot", "kangaroo", "zebra"],
        hand1_species=["zebra", "lion", "hippo", "snake"],
        deck0_species=["seal", "crocodile", "giraffe", "monkey"],
        deck1_species=["seal", "crocodile", "giraffe", "monkey"],
    )

    # Get legal actions
    py_legal = list(engine.legal_actions(py_state, 0))

    arr = GameStateArray(1)
    python_state_to_c(py_state, arr, 0)
    cy_mask = get_single_legal_mask(arr, 0)

    py_indices = sorted([action_index(a) for a in py_legal])
    cy_indices = sorted(np.where(cy_mask > 0)[0].tolist())

    # Chameleon should have no legal actions (needs target)
    # Parrot should have no legal actions (needs target)
    # Kangaroo with empty queue should have 1 action (no params)
    # Zebra should have 1 action

    match = py_indices == cy_indices
    if match:
        print(f"  PASS: Legal actions match ({len(py_indices)} actions)")
    else:
        print(f"  FAIL: Python={py_indices}, Cython={cy_indices}")

    return match


def main():
    """Run all edge case tests."""
    print("=" * 60)
    print("CYTHON EDGE CASE TESTS")
    print("=" * 60)

    results = []

    # Card ability tests
    results.append(("Chameleon copies lion (multiple lions)", test_chameleon_copies_lion_into_multiple_lions()))
    results.append(("Chameleon copies parrot", test_chameleon_copies_parrot()))
    results.append(("Chameleon copies kangaroo", test_chameleon_copies_kangaroo()))
    results.append(("Chameleon copies monkey (triggers ability)", test_chameleon_copies_monkey_triggers_ability()))
    results.append(("Skunk expels top 2 species", test_skunk_top_two_species()))
    results.append(("Seal reverses queue", test_seal_reverses_queue()))
    results.append(("Snake sorts by strength", test_snake_sorts_by_strength()))
    results.append(("Crocodile eats weaker", test_crocodile_eats_weaker()))
    results.append(("Crocodile blocked by zebra", test_crocodile_blocked_by_zebra()))
    results.append(("Hippo pushes forward", test_hippo_pushes_forward()))
    results.append(("Giraffe steps forward", test_giraffe_steps_forward()))
    results.append(("Kangaroo hops", test_kangaroo_hops()))
    results.append(("Five card check", test_five_card_check()))
    results.append(("Recurring giraffe", test_recurring_giraffe()))
    results.append(("Empty queue legal actions", test_empty_queue_legal_actions()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll edge case tests passed!")
        return 0
    else:
        print(f"\n{total - passed} tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
