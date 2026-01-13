"""
Test suite for negative path (error handling) edge cases.

Tests that invalid inputs are properly rejected with appropriate errors:
invalid action indices, out-of-bounds hand indices, chameleon targeting
violations, and other error conditions. These tests ensure the game
engine fails fast with clear error messages.
"""

import numpy as np
import pytest

from _01_simulator import actions, engine, rules, state
from _01_simulator.action_space import (
    ACTION_DIM,
    action_index,
    greedy_masked_action,
    index_to_action,
    sample_masked_action,
)


def make_card(owner: int, species: str) -> state.Card:
    """Helper to create a card with given owner and species."""
    return state.Card(owner=owner, species=species)


def create_non_terminal_state_with_cards(
    hand_cards: tuple[state.Card, ...],
    queue_cards: tuple[state.Card, ...] = (),
    active_player: int = 0,
) -> state.State:
    """Create a non-terminal game state with specified hand for player 0.

    Ensures total cards >= MAX_QUEUE_LENGTH to avoid terminal state.
    Player 1 gets filler cards to meet the minimum.
    """
    # Calculate how many more cards we need to avoid terminal state
    # Total must be >= MAX_QUEUE_LENGTH (5)
    total_cards = len(hand_cards) + len(queue_cards)

    # Player 1 needs enough cards to make total >= 5
    player1_hand_needed = max(0, rules.MAX_QUEUE_LENGTH - total_cards)

    player1_hand = tuple(
        make_card(1, species)
        for species in ["lion", "hippo", "crocodile", "snake", "giraffe"][:player1_hand_needed]
    )

    # If still not enough, add more to player 1's deck
    remaining = rules.MAX_QUEUE_LENGTH - (total_cards + len(player1_hand))
    player1_deck = tuple(
        make_card(1, species)
        for species in ["zebra", "seal", "monkey", "kangaroo"][:max(0, remaining)]
    )

    players = (
        state.PlayerState(deck=(), hand=hand_cards),
        state.PlayerState(deck=player1_deck, hand=player1_hand),
    )

    game_state = state.State(
        seed=42,
        turn=5,
        active_player=active_player,
        players=players,
        zones=state.Zones(queue=queue_cards),
    )

    assert not engine.is_terminal(game_state), (
        f"Test state should not be terminal. "
        f"Total cards: {len(hand_cards) + len(queue_cards) + len(player1_hand) + len(player1_deck)}"
    )

    return game_state


class TestInvalidActionIndices:
    """Tests for invalid action index handling."""

    def test_negative_action_index_raises(self):
        """Negative action index should raise IndexError.

        Edge case: Invalid neural network output.
        """
        with pytest.raises(IndexError, match="out of range"):
            index_to_action(-1)

    def test_action_index_beyond_dim_raises(self):
        """Action index >= ACTION_DIM should raise IndexError.

        Edge case: Off-by-one error in action selection.
        """
        with pytest.raises(IndexError, match="out of range"):
            index_to_action(ACTION_DIM)

    def test_large_invalid_action_index_raises(self):
        """Very large action index should raise IndexError.

        Edge case: Corrupted action data.
        """
        with pytest.raises(IndexError, match="out of range"):
            index_to_action(999999)

    def test_action_not_in_catalog_raises(self):
        """Action not in catalog should raise ValueError.

        Edge case: Manually constructed invalid action.
        """
        # Create an action with impossible params
        invalid_action = actions.Action(
            hand_index=0,
            params=(100, 100, 100)  # Three params not in catalog
        )

        with pytest.raises(ValueError, match="not found in catalog"):
            action_index(invalid_action)


class TestOutOfBoundsHandIndex:
    """Tests for out-of-bounds hand index handling."""

    def test_hand_index_negative_raises(self):
        """Negative hand index should raise ValueError.

        Edge case: Invalid action construction.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "zebra"),),
            queue_cards=(make_card(0, "lion"),),
        )

        invalid_action = actions.Action(hand_index=-1)

        with pytest.raises(ValueError, match="Hand index out of range"):
            engine.step(game_state, invalid_action)

    def test_hand_index_beyond_hand_size_raises(self):
        """Hand index >= hand size should raise ValueError.

        Edge case: Off-by-one error in action selection.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "zebra"),),  # Size 1
            queue_cards=(make_card(0, "lion"),),
        )

        # Hand has 1 card, so index 1 is invalid
        invalid_action = actions.Action(hand_index=1)

        with pytest.raises(ValueError, match="Hand index out of range"):
            engine.step(game_state, invalid_action)

    def test_hand_index_for_empty_hand_raises(self):
        """Any hand index for empty hand should raise ValueError.

        Edge case: Playing from empty hand.
        Note: We need to make player 1 active since player 0 has empty hand.
        With empty hand, the game may be terminal, so we use a different approach.
        """
        # Create state with enough cards but player 0 has empty hand
        queue = (make_card(0, "lion"), make_card(0, "zebra"))
        players = (
            state.PlayerState(deck=(), hand=()),  # Empty hand
            state.PlayerState(
                deck=(make_card(1, "seal"), make_card(1, "monkey")),
                hand=(make_card(1, "parrot"), make_card(1, "giraffe")),
            ),
        )
        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,  # Player 0 with empty hand is active
            players=players,
            zones=state.Zones(queue=queue),
        )

        # Verify this is not terminal (total = 2 + 4 + 2 = 8 >= 5)
        assert not engine.is_terminal(game_state)

        invalid_action = actions.Action(hand_index=0)

        with pytest.raises(ValueError, match="Hand index out of range"):
            engine.step(game_state, invalid_action)


class TestChameleonInvalidTargets:
    """Tests for chameleon with invalid targets."""

    def test_chameleon_copy_self_raises(self):
        """Chameleon cannot copy itself.

        Edge case: Self-referential copy attempt.
        Note: The chameleon in hand cannot literally copy itself since it's
        not in the queue. This test checks copying another chameleon.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "chameleon"),),
            queue_cards=(make_card(1, "chameleon"),),  # Another chameleon in queue
        )

        # Try to copy the chameleon in queue (forbidden)
        invalid_action = actions.Action(hand_index=0, params=(0,))

        with pytest.raises(ValueError, match="cannot copy"):
            engine.step(game_state, invalid_action)

    def test_chameleon_copy_another_chameleon_raises(self):
        """Chameleon cannot copy another chameleon.

        Edge case: Chameleon-chameleon copy is explicitly forbidden.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "chameleon"),),
            queue_cards=(make_card(1, "chameleon"),),  # Different owner
        )

        invalid_action = actions.Action(hand_index=0, params=(0,))

        with pytest.raises(ValueError, match=r"[Cc]hameleon.*cannot copy.*chameleon"):
            engine.step(game_state, invalid_action)

    def test_chameleon_target_out_of_range_raises(self):
        """Chameleon with out-of-range target should raise ValueError.

        Edge case: Invalid queue index for chameleon.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "chameleon"),),
            queue_cards=(make_card(1, "lion"),),  # Only one card at index 0
        )

        # Try to target index 5 when queue only has 1 card
        invalid_action = actions.Action(hand_index=0, params=(5,))

        with pytest.raises(ValueError, match="out of range"):
            engine.step(game_state, invalid_action)

    def test_chameleon_missing_target_raises(self):
        """Chameleon without target params should raise ValueError.

        Edge case: Incomplete chameleon action.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "chameleon"),),
            queue_cards=(make_card(1, "lion"),),
        )

        # Missing target param
        invalid_action = actions.Action(hand_index=0, params=())

        with pytest.raises(ValueError, match="requires target"):
            engine.step(game_state, invalid_action)

    def test_chameleon_as_parrot_missing_target_raises(self):
        """Chameleon copying parrot without parrot target should raise.

        Edge case: Incomplete chameleon-as-parrot action.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "chameleon"),),
            queue_cards=(make_card(1, "parrot"), make_card(1, "zebra")),
        )

        # Copy parrot (index 0) but missing parrot's target param
        invalid_action = actions.Action(hand_index=0, params=(0,))

        with pytest.raises(ValueError, match=r"parrot.*requires.*target"):
            engine.step(game_state, invalid_action)


class TestParrotInvalidTargets:
    """Tests for parrot with invalid targets."""

    def test_parrot_missing_target_raises(self):
        """Parrot without target should raise ValueError.

        Edge case: Incomplete parrot action.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "parrot"),),
            queue_cards=(make_card(1, "zebra"),),
        )

        invalid_action = actions.Action(hand_index=0, params=())

        with pytest.raises(ValueError, match=r"[Pp]arrot requires.*target"):
            engine.step(game_state, invalid_action)

    def test_parrot_target_out_of_range_raises(self):
        """Parrot with out-of-range target should raise ValueError.

        Edge case: Invalid queue index for parrot.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "parrot"),),
            queue_cards=(make_card(1, "zebra"),),  # Only index 0 valid
        )

        invalid_action = actions.Action(hand_index=0, params=(10,))

        with pytest.raises(ValueError, match="out of range"):
            engine.step(game_state, invalid_action)


class TestKangarooInvalidHops:
    """Tests for kangaroo with invalid hop distances."""

    def test_kangaroo_hop_exceeds_max_raises(self):
        """Kangaroo hop > 2 should raise ValueError.

        Edge case: Kangaroo can only hop 1 or 2 spaces max.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "kangaroo"),),
            queue_cards=(make_card(1, "zebra"), make_card(1, "lion"), make_card(1, "giraffe")),
        )

        # Try to hop 3 (exceeds max of 2)
        invalid_action = actions.Action(hand_index=0, params=(3,))

        with pytest.raises(ValueError, match=r"hop.*out of range"):
            engine.step(game_state, invalid_action)

    def test_kangaroo_hop_with_empty_queue_and_params_raises(self):
        """Kangaroo with empty queue cannot have hop params.

        Edge case: Kangaroo hop when no cards to hop over.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "kangaroo"),),
            queue_cards=(),  # Empty queue
        )

        # Try to hop when there's nothing to hop over
        invalid_action = actions.Action(hand_index=0, params=(1,))

        with pytest.raises(ValueError, match="cannot hop"):
            engine.step(game_state, invalid_action)

    def test_kangaroo_hop_zero_raises(self):
        """Kangaroo hop of 0 should be invalid (min hop is 1).

        Edge case: Zero hop is not a valid hop distance.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "kangaroo"),),
            queue_cards=(make_card(1, "zebra"),),
        )

        # Hop of 0 is not valid (should be no params for no hop)
        invalid_action = actions.Action(hand_index=0, params=(0,))

        with pytest.raises(ValueError, match=r"hop.*out of range"):
            engine.step(game_state, invalid_action)


class TestNonParameterizedCardsWithParams:
    """Tests for non-parameterized cards given params."""

    def test_lion_with_params_raises(self):
        """Lion should not accept parameters.

        Edge case: Simple card given unexpected params.
        """
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "lion"),),
            queue_cards=(make_card(1, "zebra"),),
        )

        invalid_action = actions.Action(hand_index=0, params=(1,))

        with pytest.raises(ValueError, match="does not accept parameters"):
            engine.step(game_state, invalid_action)

    def test_zebra_with_params_raises(self):
        """Zebra should not accept parameters."""
        game_state = create_non_terminal_state_with_cards(
            hand_cards=(make_card(0, "zebra"),),
            queue_cards=(make_card(1, "lion"),),
        )

        invalid_action = actions.Action(hand_index=0, params=(0, 1))

        with pytest.raises(ValueError, match="does not accept parameters"):
            engine.step(game_state, invalid_action)


class TestMaskBasedSamplingErrors:
    """Tests for action sampling with invalid masks."""

    def test_sample_with_all_zero_mask_raises(self):
        """Sampling with all-zero mask should raise ValueError.

        Edge case: No legal actions available.
        """
        logits = np.random.randn(ACTION_DIM).astype(np.float32)
        mask = np.zeros(ACTION_DIM, dtype=np.float32)

        with pytest.raises(ValueError, match="[Nn]o legal actions"):
            sample_masked_action(logits, mask)

    def test_greedy_with_all_zero_mask_raises(self):
        """Greedy selection with all-zero mask should raise ValueError.

        Edge case: No legal actions for greedy selection.
        """
        logits = np.random.randn(ACTION_DIM).astype(np.float32)
        mask = np.zeros(ACTION_DIM, dtype=np.float32)

        with pytest.raises(ValueError, match="[Nn]o legal actions"):
            greedy_masked_action(logits, mask)

    def test_sample_with_negative_temperature_raises(self):
        """Negative temperature should raise ValueError.

        Edge case: Invalid temperature parameter.
        """
        logits = np.random.randn(ACTION_DIM).astype(np.float32)
        mask = np.ones(ACTION_DIM, dtype=np.float32)

        with pytest.raises(ValueError, match="[Tt]emperature"):
            sample_masked_action(logits, mask, temperature=-1.0)

    def test_sample_with_zero_temperature_raises(self):
        """Zero temperature should raise ValueError.

        Edge case: Division by zero prevention.
        """
        logits = np.random.randn(ACTION_DIM).astype(np.float32)
        mask = np.ones(ACTION_DIM, dtype=np.float32)

        with pytest.raises(ValueError, match="[Tt]emperature"):
            sample_masked_action(logits, mask, temperature=0.0)


class TestStateValidationErrors:
    """Tests for state construction validation errors."""

    def test_invalid_species_raises(self):
        """Unknown species should raise ValueError.

        Edge case: Typo or invalid species name.
        """
        with pytest.raises(ValueError, match="[Uu]nknown species"):
            make_card(0, "invalid_species")

    def test_invalid_owner_raises(self):
        """Owner out of range should raise ValueError.

        Edge case: Invalid player index.
        """
        with pytest.raises(ValueError, match="[Oo]wner.*out of range"):
            make_card(5, "lion")  # Only players 0 and 1 valid

    def test_negative_owner_raises(self):
        """Negative owner should raise ValueError."""
        with pytest.raises(ValueError, match="[Oo]wner.*out of range"):
            make_card(-1, "lion")

    def test_wrong_player_count_raises(self):
        """State with wrong number of players should raise ValueError.

        Edge case: Game configured for different player count.
        """
        players = (
            state.PlayerState(deck=(), hand=()),
            state.PlayerState(deck=(), hand=()),
            state.PlayerState(deck=(), hand=()),  # Third player
        )

        with pytest.raises(ValueError, match="two players"):
            state.State(
                seed=42,
                turn=0,
                active_player=0,
                players=players,
                zones=state.Zones(),
            )

    def test_invalid_active_player_raises(self):
        """Active player out of range should raise ValueError."""
        players = (
            state.PlayerState(deck=(), hand=()),
            state.PlayerState(deck=(), hand=()),
        )

        with pytest.raises(ValueError, match="[Aa]ctive player.*out of range"):
            state.State(
                seed=42,
                turn=0,
                active_player=5,  # Invalid
                players=players,
                zones=state.Zones(),
            )


class TestTerminalStateStepError:
    """Tests for stepping on terminal states."""

    def test_step_on_terminal_raises(self):
        """Stepping on terminal state should raise ValueError.

        Edge case: Attempting to continue finished game.
        """
        # Create terminal state
        queue = (make_card(0, "lion"),)
        players = (
            state.PlayerState(deck=(), hand=(make_card(0, "parrot"),)),
            state.PlayerState(deck=(), hand=(make_card(1, "zebra"),)),
        )
        # Total cards = 1 + 1 + 1 = 3 < MAX_QUEUE_LENGTH (5)
        game_state = state.State(
            seed=42,
            turn=10,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        assert engine.is_terminal(game_state), "State should be terminal"

        valid_action = actions.Action(hand_index=0, params=(0,))

        with pytest.raises(ValueError, match="[Cc]annot step.*finished"):
            engine.step(game_state, valid_action)
