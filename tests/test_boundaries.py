"""
Test suite for boundary condition edge cases.

Tests behavior at critical boundaries: empty hands, maximum queue length,
single-card queues, and score tie scenarios. These edge cases are essential
for ensuring game logic handles extreme states correctly.
"""

import numpy as np
import pytest

from _01_simulator import actions, engine, rules, state
from _01_simulator.action_space import (
    ACTION_DIM,
    action_index,
    index_to_action,
    legal_action_mask_tensor,
)
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor


def make_card(owner: int, species: str, entered_turn: int = -1) -> state.Card:
    """Helper to create a card with given owner and species."""
    return state.Card(owner=owner, species=species, entered_turn=entered_turn)


class TestEmptyHandBoundary:
    """Tests for empty hand (end-game) scenarios."""

    def test_empty_hand_no_legal_actions(self):
        """Player with empty hand should have no legal actions.

        Edge case: End-game when player exhausts all cards.
        """
        queue = (make_card(0, "lion"), make_card(1, "zebra"))
        players = (
            state.PlayerState(deck=(), hand=()),  # Empty hand
            state.PlayerState(deck=(), hand=(make_card(1, "parrot"),)),
        )
        game_state = state.State(
            seed=42,
            turn=10,
            active_player=0,  # Player with empty hand is active
            players=players,
            zones=state.Zones(queue=queue),
        )

        legal = list(engine.legal_actions(game_state, 0))
        assert len(legal) == 0, "Empty hand should yield no legal actions"

    def test_empty_hand_action_mask_all_zeros(self):
        """Action mask should be all zeros for empty hand."""
        queue = (make_card(0, "lion"), make_card(1, "zebra"))
        players = (
            state.PlayerState(deck=(), hand=()),
            state.PlayerState(deck=(), hand=(make_card(1, "parrot"),)),
        )
        game_state = state.State(
            seed=42,
            turn=10,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        mask = legal_action_mask_tensor(game_state, 0)

        assert mask.shape == (ACTION_DIM,)
        assert np.sum(mask) == 0.0, "Empty hand should have all-zero mask"

    def test_single_card_hand_limited_actions(self):
        """Player with single card should only have actions for that card.

        Edge case: Near end-game with minimal options.
        """
        queue = (make_card(0, "lion"), make_card(1, "zebra"))
        players = (
            state.PlayerState(deck=(), hand=(make_card(0, "monkey"),)),
            state.PlayerState(deck=(), hand=(make_card(1, "parrot"),)),
        )
        game_state = state.State(
            seed=42,
            turn=10,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        legal = list(engine.legal_actions(game_state, 0))

        # Monkey has no params, so exactly 1 action
        assert len(legal) == 1, "Single monkey card should have exactly 1 action"
        assert legal[0].hand_index == 0
        assert legal[0].params == ()

    def test_observation_with_empty_hand(self):
        """Observation encoding should handle empty hand gracefully."""
        queue = (make_card(0, "lion"),)
        players = (
            state.PlayerState(deck=(), hand=()),
            state.PlayerState(deck=(), hand=(make_card(1, "parrot"),)),
        )
        game_state = state.State(
            seed=42,
            turn=10,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        obs_tensor = state_to_tensor(game_state, perspective=0)

        assert obs_tensor.shape == (OBSERVATION_DIM,)
        assert np.all(np.isfinite(obs_tensor)), (
            "Observation should be valid even with empty hand"
        )


class TestMaxQueueLengthBoundary:
    """Tests for queue at exactly MAX_QUEUE_LENGTH."""

    def test_queue_at_max_length(self):
        """Verify behavior when queue is exactly at MAX_QUEUE_LENGTH.

        Edge case: Maximum capacity triggers five-card check.
        """
        assert rules.MAX_QUEUE_LENGTH == 5, "Test assumes MAX_QUEUE_LENGTH is 5"

        queue = tuple(make_card(i % 2, "lion") for i in range(5))
        assert len(queue) == rules.MAX_QUEUE_LENGTH

        players = (
            state.PlayerState(
                deck=(),
                hand=(make_card(0, "zebra"), make_card(0, "monkey")),
            ),
            state.PlayerState(
                deck=(),
                hand=(make_card(1, "parrot"), make_card(1, "giraffe")),
            ),
        )
        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        # Should still have legal actions (can still play cards)
        legal = list(engine.legal_actions(game_state, 0))
        assert len(legal) > 0, "Should have legal actions even at max queue"

    def test_five_card_check_triggered_on_max_queue(self):
        """Playing a card when queue hits 5 should trigger five-card check.

        Edge case: Cards enter beasty_bar and one bounces to thats_it.
        """
        # Start with 4 cards in queue
        queue = tuple(make_card(0, species) for species in ["lion", "zebra", "giraffe", "monkey"])
        assert len(queue) == 4

        players = (
            state.PlayerState(deck=(), hand=(make_card(0, "seal"),)),
            state.PlayerState(deck=(), hand=(make_card(1, "parrot"),)),
        )
        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        # Play the seal (no params)
        action = actions.Action(hand_index=0)
        next_state = engine.step(game_state, action)

        # After 5-card check, queue should be reduced
        # First 2 go to beasty_bar, last 1 goes to thats_it, 2 remain in queue
        assert len(next_state.zones.queue) == 2, (
            f"Queue should have 2 cards after 5-card check, got {len(next_state.zones.queue)}"
        )
        assert len(next_state.zones.beasty_bar) == 2, (
            f"Beasty bar should have 2 cards, got {len(next_state.zones.beasty_bar)}"
        )
        assert len(next_state.zones.thats_it) == 1, (
            f"That's it should have 1 card, got {len(next_state.zones.thats_it)}"
        )


class TestSingleCardQueueWithRecursiveActions:
    """Tests for single-card queue with recursive actions."""

    def test_parrot_single_card_queue(self):
        """Parrot with single-card queue has exactly one target.

        Edge case: Minimal queue for targeting abilities.
        """
        queue = (make_card(1, "zebra"),)  # Single card to target
        players = (
            state.PlayerState(deck=(), hand=(make_card(0, "parrot"),)),
            state.PlayerState(deck=(), hand=(make_card(1, "lion"),)),
        )
        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        legal = list(engine.legal_actions(game_state, 0))

        # Parrot should have exactly 1 target (index 0)
        assert len(legal) == 1
        assert legal[0].hand_index == 0
        assert legal[0].params == (0,), "Parrot should target index 0"

    def test_chameleon_single_card_queue(self):
        """Chameleon with single-card queue has one copy target.

        Edge case: Minimal queue for chameleon copying.
        """
        queue = (make_card(1, "lion"),)  # Single non-chameleon card
        players = (
            state.PlayerState(deck=(), hand=(make_card(0, "chameleon"),)),
            state.PlayerState(deck=(), hand=(make_card(1, "zebra"),)),
        )
        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        legal = list(engine.legal_actions(game_state, 0))

        # Chameleon can copy the lion (no params needed for lion)
        assert len(legal) == 1
        assert legal[0].hand_index == 0
        assert legal[0].params == (0,), "Chameleon should copy index 0"

    def test_kangaroo_single_card_queue_hop_options(self):
        """Kangaroo with single-card queue can hop 0 or 1 positions.

        Edge case: Minimal queue for hop ability.
        """
        queue = (make_card(1, "zebra"),)  # Single card ahead
        players = (
            state.PlayerState(deck=(), hand=(make_card(0, "kangaroo"),)),
            state.PlayerState(deck=(), hand=(make_card(1, "lion"),)),
        )
        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        legal = list(engine.legal_actions(game_state, 0))

        # Kangaroo with 1 card ahead: can hop 1
        # min(2, 1) = 1, so hop options are (1,)
        assert len(legal) == 1
        assert legal[0].params == (1,), "Kangaroo should be able to hop 1"

    def test_kangaroo_empty_queue_no_hop(self):
        """Kangaroo with empty queue cannot hop at all.

        Edge case: No cards to hop over.
        """
        queue = ()  # Empty queue
        players = (
            state.PlayerState(deck=(), hand=(make_card(0, "kangaroo"),)),
            state.PlayerState(deck=(), hand=(make_card(1, "lion"),)),
        )
        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,
            players=players,
            zones=state.Zones(queue=queue),
        )

        legal = list(engine.legal_actions(game_state, 0))

        # Kangaroo with empty queue: no hop, just play
        assert len(legal) == 1
        assert legal[0].params == (), "Kangaroo with empty queue should have no hop param"


class TestScoreTieConditions:
    """Tests for draw/tie scoring scenarios."""

    def test_zero_zero_draw(self):
        """Both players at 0 points is a draw.

        Edge case: No cards in beasty_bar.
        """
        players = (
            state.PlayerState(deck=(), hand=()),
            state.PlayerState(deck=(), hand=()),
        )
        game_state = state.State(
            seed=42,
            turn=20,
            active_player=0,
            players=players,
            zones=state.Zones(queue=(), beasty_bar=(), thats_it=()),
        )

        scores = engine.score(game_state)
        assert scores[0] == 0
        assert scores[1] == 0
        assert scores[0] == scores[1], "Both at 0 should be a draw"

    def test_equal_positive_scores_draw(self):
        """Both players with equal positive scores is a draw."""
        beasty_bar = (
            make_card(0, "lion"),    # P0: 2 points
            make_card(0, "hippo"),   # P0: 2 points = 4 total
            make_card(1, "zebra"),   # P1: 4 points = 4 total
        )
        players = (
            state.PlayerState(deck=(), hand=()),
            state.PlayerState(deck=(), hand=()),
        )
        game_state = state.State(
            seed=42,
            turn=20,
            active_player=0,
            players=players,
            zones=state.Zones(queue=(), beasty_bar=beasty_bar, thats_it=()),
        )

        scores = engine.score(game_state)
        assert scores[0] == 4, f"Player 0 should have 4 points, got {scores[0]}"
        assert scores[1] == 4, f"Player 1 should have 4 points, got {scores[1]}"
        assert scores[0] == scores[1], "Equal scores should result in draw"

    def test_one_point_difference(self):
        """Smallest possible score difference (1 point)."""
        beasty_bar = (
            make_card(0, "lion"),    # P0: 2 points
            make_card(1, "skunk"),   # P1: 4 points
            make_card(1, "monkey"),  # P1: 3 points = 7 total
        )
        players = (
            state.PlayerState(deck=(), hand=()),
            state.PlayerState(deck=(), hand=()),
        )
        game_state = state.State(
            seed=42,
            turn=20,
            active_player=0,
            players=players,
            zones=state.Zones(queue=(), beasty_bar=beasty_bar, thats_it=()),
        )

        scores = engine.score(game_state)
        assert scores[0] == 2
        assert scores[1] == 7
        assert scores[1] - scores[0] == 5, "Score difference should be 5"


class TestActionIndexBoundaries:
    """Tests for action index conversion at boundaries."""

    def test_action_index_zero(self):
        """Index 0 should be valid and reversible."""
        action = index_to_action(0)
        recovered_index = action_index(action)
        assert recovered_index == 0

    def test_action_index_max(self):
        """Maximum index (ACTION_DIM - 1) should be valid."""
        max_index = ACTION_DIM - 1
        action = index_to_action(max_index)
        recovered_index = action_index(action)
        assert recovered_index == max_index

    def test_action_index_out_of_bounds_negative(self):
        """Negative index should raise IndexError."""
        with pytest.raises(IndexError):
            index_to_action(-1)

    def test_action_index_out_of_bounds_too_large(self):
        """Index >= ACTION_DIM should raise IndexError."""
        with pytest.raises(IndexError):
            index_to_action(ACTION_DIM)

        with pytest.raises(IndexError):
            index_to_action(ACTION_DIM + 100)

    def test_all_indices_reversible(self):
        """All valid indices should round-trip correctly.

        Boundary check: Ensure complete action catalog is consistent.
        """
        for idx in range(ACTION_DIM):
            action = index_to_action(idx)
            recovered = action_index(action)
            assert recovered == idx, f"Index {idx} failed round-trip"


class TestQueueManipulationBoundaries:
    """Tests for queue state manipulation at boundaries."""

    def test_append_to_empty_queue(self):
        """Appending to empty queue should work."""
        game_state = state.State(
            seed=42,
            turn=0,
            active_player=0,
            players=(
                state.PlayerState(deck=(), hand=()),
                state.PlayerState(deck=(), hand=()),
            ),
            zones=state.Zones(queue=()),
        )

        card = make_card(0, "lion")
        new_state = state.append_queue(game_state, card)

        assert len(new_state.zones.queue) == 1
        assert new_state.zones.queue[0] == card

    def test_replace_queue_with_empty(self):
        """Replacing queue with empty tuple should work."""
        queue = (make_card(0, "lion"), make_card(1, "zebra"))
        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,
            players=(
                state.PlayerState(deck=(), hand=()),
                state.PlayerState(deck=(), hand=()),
            ),
            zones=state.Zones(queue=queue),
        )

        new_state = state.replace_queue(game_state, ())

        assert len(new_state.zones.queue) == 0

    def test_replace_queue_at_max_length(self):
        """Replacing queue at exactly MAX_QUEUE_LENGTH should work."""
        new_queue = tuple(make_card(0, "lion") for _ in range(rules.MAX_QUEUE_LENGTH))

        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,
            players=(
                state.PlayerState(deck=(), hand=()),
                state.PlayerState(deck=(), hand=()),
            ),
            zones=state.Zones(queue=()),
        )

        new_state = state.replace_queue(game_state, new_queue)
        assert len(new_state.zones.queue) == rules.MAX_QUEUE_LENGTH

    def test_replace_queue_exceeds_max_raises(self):
        """Replacing queue with more than MAX_QUEUE_LENGTH should raise."""
        oversized_queue = tuple(make_card(0, "lion") for _ in range(rules.MAX_QUEUE_LENGTH + 1))

        game_state = state.State(
            seed=42,
            turn=5,
            active_player=0,
            players=(
                state.PlayerState(deck=(), hand=()),
                state.PlayerState(deck=(), hand=()),
            ),
            zones=state.Zones(queue=()),
        )

        with pytest.raises(ValueError, match="exceeds maximum"):
            state.replace_queue(game_state, oversized_queue)
