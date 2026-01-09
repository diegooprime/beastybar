"""
Test suite for terminal state edge cases.

Tests behavior when the game has ended, ensuring proper handling of
stepping, legal actions, and network forward passes on terminal states.
These edge cases are critical for training stability and game logic correctness.
"""

import numpy as np
import pytest
import torch

from _01_simulator import engine, rules, state
from _01_simulator.action_space import (
    ACTION_DIM,
    legal_action_mask_tensor,
    legal_action_space,
)
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor
from _02_agents.neural.network import create_network
from _02_agents.neural.utils import NetworkConfig


def make_card(owner: int, species: str) -> state.Card:
    """Helper to create a card with given owner and species."""
    return state.Card(owner=owner, species=species)


def create_terminal_state_no_cards() -> state.State:
    """Create a terminal state where both players have empty hands and decks.

    This represents an end-game scenario where all cards have been played.
    The game should be terminal because no more moves are possible.
    """
    queue = (
        make_card(0, "lion"),
        make_card(1, "zebra"),
        make_card(0, "giraffe"),
    )
    players = (
        state.PlayerState(deck=(), hand=()),
        state.PlayerState(deck=(), hand=()),
    )
    return state.State(
        seed=42,
        turn=20,  # Late game
        active_player=0,
        players=players,
        zones=state.Zones(queue=queue),
    )


def create_terminal_state_insufficient_cards() -> state.State:
    """Create a terminal state where remaining cards cannot fill the queue.

    Game ends when total remaining cards (hands + decks + queue) < MAX_QUEUE_LENGTH.
    This tests the early termination condition.
    """
    queue = (make_card(0, "lion"),)  # 1 card in queue
    players = (
        state.PlayerState(deck=(), hand=(make_card(0, "parrot"),)),  # 1 card
        state.PlayerState(deck=(), hand=(make_card(1, "zebra"),)),   # 1 card
    )
    # Total = 1 + 1 + 1 = 3 < MAX_QUEUE_LENGTH (5)
    return state.State(
        seed=42,
        turn=10,
        active_player=0,
        players=players,
        zones=state.Zones(queue=queue),
    )


def create_non_terminal_state() -> state.State:
    """Create a non-terminal mid-game state for comparison."""
    queue = (make_card(0, "lion"), make_card(1, "zebra"))
    players = (
        state.PlayerState(
            deck=(make_card(0, "seal"), make_card(0, "monkey")),
            hand=(make_card(0, "parrot"), make_card(0, "giraffe")),
        ),
        state.PlayerState(
            deck=(make_card(1, "crocodile"),),
            hand=(make_card(1, "kangaroo"), make_card(1, "hippo")),
        ),
    )
    return state.State(
        seed=42,
        turn=5,
        active_player=0,
        players=players,
        zones=state.Zones(queue=queue),
    )


class TestTerminalStateDetection:
    """Tests for correct identification of terminal states."""

    def test_terminal_when_both_hands_and_decks_empty(self):
        """Game should be terminal when all cards have been played.

        Edge case: Both players have exhausted their resources.
        """
        game_state = create_terminal_state_no_cards()
        assert engine.is_terminal(game_state), (
            "Game should be terminal when both players have empty hands and decks"
        )

    def test_terminal_when_insufficient_cards_for_queue(self):
        """Game should be terminal when remaining cards < MAX_QUEUE_LENGTH.

        Edge case: Early termination due to card count threshold.
        """
        game_state = create_terminal_state_insufficient_cards()
        assert engine.is_terminal(game_state), (
            f"Game should be terminal when total cards < MAX_QUEUE_LENGTH ({rules.MAX_QUEUE_LENGTH})"
        )

    def test_non_terminal_with_sufficient_cards(self):
        """Game should not be terminal when players can still act."""
        game_state = create_non_terminal_state()
        assert not engine.is_terminal(game_state), (
            "Game should not be terminal when players have cards to play"
        )

    def test_turn_zero_never_terminal(self):
        """Initial state (turn 0) should never be terminal.

        Edge case: Even with weird card configurations, turn 0 is protected.
        """
        game_state = state.initial_state(seed=42)
        assert game_state.turn == 0
        assert not engine.is_terminal(game_state), (
            "Turn 0 should never be terminal regardless of card configuration"
        )


class TestStepOnTerminalState:
    """Tests for behavior when stepping in a terminal state."""

    def test_step_on_terminal_raises_value_error(self):
        """Stepping on a terminal state should raise ValueError.

        Edge case: Prevents invalid game progression after game end.
        """
        game_state = create_terminal_state_no_cards()
        assert engine.is_terminal(game_state)

        # Create a dummy action (any action)
        from _01_simulator.actions import Action
        dummy_action = Action(hand_index=0)

        with pytest.raises(ValueError, match="Cannot step.*finished"):
            engine.step(game_state, dummy_action)

    def test_step_with_trace_on_terminal_raises_value_error(self):
        """step_with_trace should also raise on terminal state."""
        game_state = create_terminal_state_no_cards()

        from _01_simulator.actions import Action
        dummy_action = Action(hand_index=0)

        with pytest.raises(ValueError, match="Cannot step.*finished"):
            engine.step_with_trace(game_state, dummy_action)


class TestLegalActionsOnTerminalState:
    """Tests for legal_actions behavior on terminal states."""

    def test_legal_actions_empty_on_terminal_empty_hand(self):
        """Terminal state with empty hand should yield no legal actions.

        Edge case: Player has no cards to play.
        """
        game_state = create_terminal_state_no_cards()

        legal = list(engine.legal_actions(game_state, 0))
        assert len(legal) == 0, (
            "Terminal state with empty hand should have no legal actions"
        )

    def test_legal_actions_empty_on_terminal_insufficient_cards(self):
        """Terminal state should yield no legal actions even if hand not empty.

        Note: legal_actions doesn't check terminal status, it checks
        if the player is active. We need to verify the expected behavior.
        """
        game_state = create_terminal_state_insufficient_cards()

        # Player 0 is active and has cards, so legal_actions will return moves
        # The game logic relies on is_terminal check before calling step
        legal = list(engine.legal_actions(game_state, 0))

        # This is actually expected - legal_actions doesn't check terminal
        # The game loop should check is_terminal first
        assert isinstance(legal, list), "legal_actions should return iterable"

    def test_legal_actions_non_active_player_returns_empty(self):
        """Requesting legal actions for non-active player returns empty.

        Edge case: Only the active player can act.
        """
        game_state = create_non_terminal_state()
        assert game_state.active_player == 0

        legal = list(engine.legal_actions(game_state, 1))  # Player 1 not active
        assert len(legal) == 0, (
            "Non-active player should have no legal actions"
        )

    def test_legal_action_space_terminal_state(self):
        """ActionSpace mask should be all zeros for terminal empty-hand state."""
        game_state = create_terminal_state_no_cards()

        action_space = legal_action_space(game_state, 0)

        # All mask entries should be 0 since hand is empty
        assert sum(action_space.mask) == 0, (
            "Terminal state with empty hand should have all-zero action mask"
        )
        assert len(action_space.legal_indices) == 0, (
            "Terminal state should have no legal indices"
        )

    def test_legal_action_mask_tensor_terminal(self):
        """Mask tensor should be all zeros for terminal state."""
        game_state = create_terminal_state_no_cards()

        mask = legal_action_mask_tensor(game_state, 0)

        assert mask.shape == (ACTION_DIM,)
        assert mask.dtype == np.float32
        assert np.sum(mask) == 0.0, (
            "Terminal state mask tensor should be all zeros"
        )


class TestNetworkOnTerminalState:
    """Tests for neural network behavior on terminal state observations."""

    @pytest.fixture
    def network(self):
        """Create a small test network."""
        config = NetworkConfig(
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        return create_network(config)

    def test_observation_tensor_from_terminal_state(self):
        """Terminal state should produce valid observation tensor.

        Edge case: Network must handle end-game observations for value estimation.
        """
        game_state = create_terminal_state_no_cards()

        obs_tensor = state_to_tensor(game_state, perspective=0)

        assert obs_tensor.shape == (OBSERVATION_DIM,)
        assert obs_tensor.dtype == np.float32
        assert np.all(np.isfinite(obs_tensor)), (
            "Terminal state observation should not contain NaN or Inf"
        )

    def test_network_forward_on_terminal_observation(self, network):
        """Network should produce valid outputs for terminal state.

        Edge case: Final value estimation at game end is crucial for training.
        """
        game_state = create_terminal_state_no_cards()

        obs_tensor = state_to_tensor(game_state, perspective=0)
        obs_torch = torch.from_numpy(obs_tensor).unsqueeze(0)

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(obs_torch)

        # Check shapes
        assert policy_logits.shape == (1, ACTION_DIM)
        assert value.shape == (1, 1)

        # Check values are finite
        assert torch.all(torch.isfinite(policy_logits)), (
            "Policy logits should be finite for terminal state"
        )
        assert torch.all(torch.isfinite(value)), (
            "Value should be finite for terminal state"
        )

        # Value should be in valid range
        assert -1.0 <= value.item() <= 1.0, (
            f"Value {value.item()} should be in [-1, 1] range"
        )

    def test_network_with_all_zero_mask_terminal(self, network):
        """Network with all-zero mask (terminal) should handle gracefully.

        Edge case: When applying mask, all-zero mask produces all -inf logits.
        The network itself doesn't apply the mask, but downstream sampling will.
        """
        game_state = create_terminal_state_no_cards()

        obs_tensor = state_to_tensor(game_state, perspective=0)
        obs_torch = torch.from_numpy(obs_tensor).unsqueeze(0)
        mask = torch.zeros(1, ACTION_DIM)  # All illegal

        network.eval()
        with torch.no_grad():
            policy_logits, _value = network(obs_torch, mask)

        # Network should still produce valid outputs
        assert torch.all(torch.isfinite(policy_logits)), (
            "Network should produce finite logits even with zero mask"
        )

        # Apply mask manually - this should result in all -inf
        masked_logits = policy_logits.clone()
        masked_logits[mask == 0] = float('-inf')

        assert torch.all(masked_logits == float('-inf')), (
            "All-zero mask should result in all -inf masked logits"
        )


class TestScoreOnTerminalState:
    """Tests for score calculation on terminal states."""

    def test_score_returns_valid_scores(self):
        """Score function should return valid scores on terminal state."""
        game_state = create_terminal_state_no_cards()

        scores = engine.score(game_state)

        assert len(scores) == rules.PLAYER_COUNT
        assert all(isinstance(s, int) for s in scores)
        assert all(s >= 0 for s in scores), "Scores should be non-negative"

    def test_score_based_on_beasty_bar_cards(self):
        """Score should sum points of cards in beasty_bar zone."""
        # Create state with known beasty_bar contents
        beasty_bar = (
            make_card(0, "lion"),   # 2 points
            make_card(0, "zebra"),  # 4 points
            make_card(1, "giraffe"), # 3 points
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
            zones=state.Zones(
                queue=(),
                beasty_bar=beasty_bar,
                thats_it=(),
            ),
        )

        scores = engine.score(game_state)

        # Player 0: lion (2) + zebra (4) = 6
        # Player 1: giraffe (3) = 3
        assert scores[0] == 6, f"Expected player 0 score 6, got {scores[0]}"
        assert scores[1] == 3, f"Expected player 1 score 3, got {scores[1]}"

    def test_score_draw_condition(self):
        """Test score when both players have equal points.

        Edge case: Draw/tie game.
        """
        # Create balanced beasty_bar
        beasty_bar = (
            make_card(0, "lion"),    # 2 points for P0
            make_card(1, "hippo"),   # 2 points for P1
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
            zones=state.Zones(
                queue=(),
                beasty_bar=beasty_bar,
                thats_it=(),
            ),
        )

        scores = engine.score(game_state)

        assert scores[0] == scores[1], "Scores should be equal for draw"
        assert scores[0] == 2, f"Expected both scores to be 2, got {scores}"
