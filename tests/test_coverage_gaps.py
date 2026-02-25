"""Tests targeting uncovered lines in _01_simulator modules.

Covers:
- state.py: insert_queue, remove_queue_card, push_to_zone, replace_queue,
  set_active_player, mask_state_for_player edge cases
- engine.py: _apply_five_card_check, step_with_trace chameleon resolve events,
  _validate_player_index, _validate_action edge paths
- rewards.py: win_loss draw, normalized_margin, shaped_reward jitter=0
- observations.py: as_dict, species_index, species_name, _encode_zone edge cases
- action_space.py: legal_action_space with invalid perspective
- simulate.py: run with games < 1
"""

import numpy as np
import pytest

from _01_simulator import (
    action_space,
    actions,
    cards,
    engine,
    formatting,
    observations,
    rewards,
    rules,
    simulate,
    state,
)


# ---------------------------------------------------------------------------
# state.py gaps
# ---------------------------------------------------------------------------


class TestStateEdgeCases:
    def _make_state(self, queue=(), hand0=("lion",), hand1=("zebra",)):
        """Helper to build a minimal State for edge-case testing."""

        def _cards(owner, species_list):
            return tuple(state.Card(owner=owner, species=s) for s in species_list)

        return state.State(
            seed=1,
            turn=1,
            active_player=0,
            players=(
                state.PlayerState(deck=(), hand=_cards(0, hand0)),
                state.PlayerState(deck=(), hand=_cards(1, hand1)),
            ),
            zones=state.Zones(queue=tuple(state.Card(owner=0, species=s) for s in queue)),
        )

    def test_card_post_init_bad_species(self):
        with pytest.raises(ValueError, match="Unknown species"):
            state.Card(owner=0, species="unicorn")

    def test_card_post_init_bad_owner(self):
        with pytest.raises(ValueError, match="Owner index out of range"):
            state.Card(owner=5, species="lion")

    def test_card_post_init_bad_entered_turn_type(self):
        with pytest.raises(TypeError, match="entered_turn must be an integer"):
            state.Card(owner=0, species="lion", entered_turn="x")

    def test_state_post_init_bad_player_count(self):
        with pytest.raises(ValueError, match="two players"):
            state.State(
                seed=0,
                turn=0,
                active_player=0,
                players=(state.PlayerState(deck=(), hand=()),),
                zones=state.Zones(),
            )

    def test_state_post_init_bad_active_player(self):
        p = state.PlayerState(deck=(), hand=())
        with pytest.raises(ValueError, match="Active player"):
            state.State(seed=0, turn=0, active_player=5, players=(p, p), zones=state.Zones())

    def test_initial_state_bad_starting_player(self):
        with pytest.raises(ValueError, match="Invalid starting player"):
            state.initial_state(seed=0, starting_player=9)

    def test_insert_queue_negative_index(self):
        gs = self._make_state(queue=("lion", "zebra"))
        card = state.Card(owner=0, species="seal")
        result = state.insert_queue(gs, -1, card)
        assert result.zones.queue[-1].species == "seal"

    def test_insert_queue_out_of_range(self):
        gs = self._make_state(queue=("lion", "zebra"))
        card = state.Card(owner=0, species="seal")
        with pytest.raises(IndexError, match="insertion index"):
            state.insert_queue(gs, 99, card)

    def test_remove_queue_card_out_of_range(self):
        gs = self._make_state(queue=("lion",))
        with pytest.raises(IndexError, match="Queue index out of range"):
            state.remove_queue_card(gs, 5)

    def test_push_to_zone_invalid_zone(self):
        gs = self._make_state()
        card = state.Card(owner=0, species="lion")
        with pytest.raises(ValueError, match="Unknown zone"):
            state.push_to_zone(gs, "nowhere", card)

    def test_replace_queue_too_long(self):
        gs = self._make_state()
        long_queue = [state.Card(owner=0, species="lion") for _ in range(rules.MAX_QUEUE_LENGTH + 1)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            state.replace_queue(gs, long_queue)

    def test_set_active_player_bad_index(self):
        gs = self._make_state()
        with pytest.raises(ValueError, match="Active player"):
            state.set_active_player(gs, 99)

    def test_set_active_player_advance_turn(self):
        gs = self._make_state()
        result = state.set_active_player(gs, 1, advance_turn=True)
        assert result.active_player == 1
        assert result.turn == gs.turn + 1

    def test_mask_state_for_player_invalid_perspective(self):
        gs = self._make_state()
        with pytest.raises(ValueError, match="Perspective"):
            state.mask_state_for_player(gs, 99)

    def test_mask_state_for_player_hides_opponent(self):
        gs = state.initial_state(seed=42)
        masked = state.mask_state_for_player(gs, 0)
        # Player 0's hand should be visible
        assert all(c.species != "unknown" for c in masked.players[0].hand)
        # Player 1's hand should be masked
        assert all(c.species == "unknown" for c in masked.players[1].hand)

    def test_remove_hand_card_out_of_range(self):
        gs = self._make_state()
        with pytest.raises(IndexError, match="Hand index out of range"):
            state.remove_hand_card(gs, 0, 99)

    def test_draw_card_empty_deck(self):
        gs = self._make_state()
        result, card = state.draw_card(gs, 0)
        assert card is None
        assert result is gs

    def test_append_queue_overflow(self):
        long_q = tuple(state.Card(owner=0, species="lion") for _ in range(rules.MAX_QUEUE_LENGTH + 1))
        p = state.PlayerState(deck=(), hand=())
        gs = state.State(seed=0, turn=1, active_player=0, players=(p, p), zones=state.Zones(queue=long_q))
        with pytest.raises(ValueError, match="maximum capacity"):
            state.append_queue(gs, state.Card(owner=0, species="seal"))


# ---------------------------------------------------------------------------
# engine.py gaps
# ---------------------------------------------------------------------------


class TestEngineEdgeCases:
    def test_step_on_terminal_raises(self):
        gs = state.initial_state(seed=1)
        # Fast-forward to terminal by emptying everything
        p = state.PlayerState(deck=(), hand=())
        terminal = state.State(
            seed=1,
            turn=10,
            active_player=0,
            players=(p, p),
            zones=state.Zones(
                queue=(state.Card(owner=0, species="lion"),),
            ),
        )
        assert engine.is_terminal(terminal)
        with pytest.raises(ValueError, match="already finished"):
            engine.step(terminal, actions.Action(hand_index=0))

    def test_legal_actions_wrong_player_yields_nothing(self):
        gs = state.initial_state(seed=5)
        # Active player is 0, so player 1 should have no legal actions
        result = list(engine.legal_actions(gs, 1))
        assert result == []

    def test_validate_player_index_out_of_range(self):
        gs = state.initial_state(seed=5)
        with pytest.raises(ValueError, match="Player index out of range"):
            list(engine.legal_actions(gs, 99))

    def test_step_with_trace_returns_five_phases(self):
        gs = state.initial_state(seed=42)
        legal = list(engine.legal_actions(gs, gs.active_player))
        assert legal
        new_state, trace = engine.step_with_trace(gs, legal[0])
        assert len(trace) == 5
        assert trace[0].name == "play"
        assert trace[1].name == "resolve"
        assert trace[2].name == "recurring"
        assert trace[3].name == "five-animal check"
        assert trace[4].name == "draw"

    def test_is_terminal_turn_zero(self):
        gs = state.initial_state(seed=1)
        assert gs.turn == 0
        assert not engine.is_terminal(gs)

    def test_score_empty_beasty_bar(self):
        gs = state.initial_state(seed=1)
        scores = engine.score(gs)
        assert scores == [0, 0]


# ---------------------------------------------------------------------------
# rewards.py gaps
# ---------------------------------------------------------------------------


class TestRewardsEdgeCases:
    def _terminal_state(self, p0_bar, p1_bar):
        """Build a terminal state with specific beasty_bar contents."""
        bar = tuple(state.Card(owner=0, species=s) for s in p0_bar) + tuple(
            state.Card(owner=1, species=s) for s in p1_bar
        )
        p = state.PlayerState(deck=(), hand=())
        return state.State(
            seed=0,
            turn=10,
            active_player=0,
            players=(p, p),
            zones=state.Zones(beasty_bar=bar, queue=(state.Card(owner=0, species="lion"),)),
        )

    def test_win_loss_draw(self):
        gs = self._terminal_state([], [])
        result = rewards.win_loss(gs)
        assert result == (0, 0)

    def test_win_loss_p0_wins(self):
        gs = self._terminal_state(["lion"], [])
        result = rewards.win_loss(gs)
        assert result == (1, -1)

    def test_win_loss_p1_wins(self):
        gs = self._terminal_state([], ["lion"])
        result = rewards.win_loss(gs)
        assert result == (-1, 1)

    def test_normalized_margin_symmetric(self):
        gs = self._terminal_state(["lion"], [])
        result = rewards.normalized_margin(gs)
        assert result[0] > 0
        assert result[1] < 0
        assert abs(result[0] + result[1]) < 1e-9

    def test_shaped_reward_no_jitter(self):
        gs = self._terminal_state(["lion"], [])
        result = rewards.shaped_reward(gs, jitter_scale=0)
        assert len(result) == 2
        # With no jitter, result is deterministic
        result2 = rewards.shaped_reward(gs, jitter_scale=0)
        assert result == result2

    def test_shaped_reward_with_explicit_seed(self):
        gs = self._terminal_state(["lion"], [])
        r1 = rewards.shaped_reward(gs, seed=999)
        r2 = rewards.shaped_reward(gs, seed=999)
        assert r1 == r2


# ---------------------------------------------------------------------------
# observations.py gaps
# ---------------------------------------------------------------------------


class TestObservationsEdgeCases:
    def test_as_dict_contains_all_fields(self):
        gs = state.initial_state(seed=10)
        obs = observations.build_observation(gs, 0)
        d = obs.as_dict()
        assert "queue" in d
        assert "hand" in d
        assert "turn" in d
        assert "perspective" in d

    def test_build_observation_invalid_perspective(self):
        gs = state.initial_state(seed=10)
        with pytest.raises(ValueError, match="Perspective"):
            observations.build_observation(gs, 99)

    def test_species_index_and_name_roundtrip(self):
        for name in rules.SPECIES:
            idx = observations.species_index(name)
            assert observations.species_name(idx) == name

    def test_species_index_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown species"):
            observations.species_index("unicorn")

    def test_species_name_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            observations.species_name(999)

    def test_build_observation_mask_hidden_false(self):
        gs = state.initial_state(seed=10)
        obs = observations.build_observation(gs, 0, mask_hidden=False)
        # When not masking, the observation still works
        assert obs.perspective == 0


# ---------------------------------------------------------------------------
# action_space.py gaps
# ---------------------------------------------------------------------------


class TestActionSpaceEdgeCases:
    def test_legal_action_space_invalid_perspective(self):
        gs = state.initial_state(seed=1)
        with pytest.raises(ValueError, match="Perspective"):
            action_space.legal_action_space(gs, 99)

    def test_action_index_unknown_action(self):
        bad_action = actions.Action(hand_index=99, params=(99, 99, 99))
        with pytest.raises(ValueError, match="not found in catalog"):
            action_space.action_index(bad_action)


# ---------------------------------------------------------------------------
# simulate.py gaps
# ---------------------------------------------------------------------------


class TestSimulateEdgeCases:
    def test_run_zero_games_raises(self):
        config = simulate.SimulationConfig(seed=1, games=0)
        with pytest.raises(ValueError, match="at least 1"):
            list(simulate.run(config))


# ---------------------------------------------------------------------------
# formatting.py coverage
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_card_label(self):
        card = state.Card(owner=0, species="kangaroo")
        assert formatting.card_label(card) == "Kangaroo (P0)"

    def test_card_list(self):
        c1 = state.Card(owner=0, species="lion")
        c2 = state.Card(owner=1, species="zebra")
        result = formatting.card_list([c1, c2])
        assert "Lion (P0)" in result
        assert "Zebra (P1)" in result
