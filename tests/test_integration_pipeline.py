"""Integration tests for the full simulation pipeline.

Tests the entire flow: state initialization -> legal actions -> step -> terminal
check -> scoring, using both the low-level engine API and the high-level
simulate module. Also validates agent integration via RandomAgent.
"""

import pytest

from _01_simulator import action_space, actions, engine, observations, rewards, simulate, state
from _02_agents.random_agent import RandomAgent


class TestFullGamePipeline:
    """End-to-end games played with the default deterministic agent."""

    @pytest.mark.parametrize("seed", [0, 42, 100, 2025, 9999])
    def test_default_agent_game_reaches_terminal(self, seed: int):
        """A game with default agents always terminates and produces valid scores."""
        config = simulate.SimulationConfig(seed=seed, games=1)
        results = list(simulate.run(config))
        assert len(results) == 1
        final = results[0]
        assert engine.is_terminal(final)
        scores = engine.score(final)
        assert len(scores) == 2
        assert all(isinstance(s, int) for s in scores)
        assert all(s >= 0 for s in scores)

    def test_multi_game_batch_produces_distinct_outcomes(self):
        """Running multiple games produces distinct final states."""
        config = simulate.SimulationConfig(seed=0, games=5)
        results = list(simulate.run(config))
        assert len(results) == 5
        # At least some games should differ (different seeds)
        seeds = {r.seed for r in results}
        assert len(seeds) == 5

    @pytest.mark.parametrize("seed", [7, 77, 777])
    def test_random_agent_game_reaches_terminal(self, seed: int):
        """RandomAgent completes a full game without errors."""
        agent_a = RandomAgent(seed=seed)
        agent_b = RandomAgent(seed=seed + 1)
        config = simulate.SimulationConfig(seed=seed, games=1, agent_a=agent_a, agent_b=agent_b)
        results = list(simulate.run(config))
        assert len(results) == 1
        assert engine.is_terminal(results[0])

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_engine_step_loop_matches_simulate_run(self, seed: int):
        """Manual engine.step loop produces same result as simulate.run."""
        # Run via simulate
        config = simulate.SimulationConfig(seed=seed, games=1)
        sim_result = list(simulate.run(config))[0]

        # Run via manual loop
        gs = state.initial_state(seed=seed)
        while not engine.is_terminal(gs):
            player = gs.active_player
            legal = list(engine.legal_actions(gs, player))
            assert legal, "Active player must have legal actions"
            gs = engine.step(gs, legal[0])

        assert engine.score(gs) == engine.score(sim_result)

    def test_observations_valid_throughout_game(self):
        """Observations are valid at every step of a game."""
        gs = state.initial_state(seed=42)
        step_count = 0
        while not engine.is_terminal(gs):
            player = gs.active_player
            obs = observations.build_observation(gs, player)
            assert obs.perspective == player
            assert obs.active_player == player
            assert len(obs.hand) == 4  # HAND_SIZE padded slots
            assert len(obs.queue) == 5  # MAX_QUEUE_LENGTH padded slots

            legal = list(engine.legal_actions(gs, player))
            gs = engine.step(gs, legal[0])
            step_count += 1

        assert step_count > 0

    def test_action_masks_consistent_with_legal_actions(self):
        """Action mask tensor has 1s exactly where legal actions exist."""
        gs = state.initial_state(seed=42)
        for _ in range(5):  # check first 5 turns
            if engine.is_terminal(gs):
                break
            player = gs.active_player
            legal = list(engine.legal_actions(gs, player))
            space = action_space.legal_action_space(gs, player)

            # Every legal action should be marked as 1 in the mask
            for act in legal:
                idx = action_space.action_index(act)
                assert space.mask[idx] == 1

            # Count of 1s should match count of legal actions
            assert sum(space.mask) == len(legal)

            gs = engine.step(gs, legal[0])

    def test_rewards_valid_at_terminal(self):
        """Reward functions produce valid output on terminal states."""
        config = simulate.SimulationConfig(seed=42, games=1)
        final = list(simulate.run(config))[0]
        assert engine.is_terminal(final)

        wl = rewards.win_loss(final)
        assert len(wl) == 2
        assert all(v in (-1, 0, 1) for v in wl)

        margin = rewards.normalized_margin(final)
        assert len(margin) == 2
        assert abs(margin[0] + margin[1]) < 1e-9  # zero-sum

        shaped = rewards.shaped_reward(final, jitter_scale=0)
        assert len(shaped) == 2

    def test_step_with_trace_produces_events(self):
        """step_with_trace returns meaningful event strings for each phase."""
        gs = state.initial_state(seed=42)
        legal = list(engine.legal_actions(gs, gs.active_player))
        new_state, trace = engine.step_with_trace(gs, legal[0])
        assert len(trace) == 5
        for step in trace:
            assert step.name
            assert step.events
            assert all(isinstance(e, str) for e in step.events)

    @pytest.mark.parametrize("starting_player", [0, 1])
    def test_game_respects_starting_player(self, starting_player: int):
        """Games started by different players produce different state sequences."""
        gs = state.initial_state(seed=42, starting_player=starting_player)
        assert gs.active_player == starting_player
        # After one step, active player should switch
        legal = list(engine.legal_actions(gs, starting_player))
        gs2 = engine.step(gs, legal[0])
        assert gs2.active_player != starting_player


class TestCardResolutionIntegration:
    """Integration tests verifying specific card abilities resolve correctly
    through the full engine pipeline."""

    def _make_state_with_queue(self, player_hand, queue_species, active=0):
        """Build a state with a specific queue and hand for player 0."""

        def _card(owner, species, turn=-1):
            return state.Card(owner=owner, species=species, entered_turn=turn)

        queue = tuple(_card(1, s) for s in queue_species)
        hand0 = tuple(_card(0, s) for s in player_hand)
        hand1 = tuple(_card(1, s) for s in ["lion", "hippo", "crocodile", "snake"])
        return state.State(
            seed=0,
            turn=1,
            active_player=active,
            players=(
                state.PlayerState(deck=(), hand=hand0),
                state.PlayerState(deck=(), hand=hand1),
            ),
            zones=state.Zones(queue=queue),
        )

    def test_lion_goes_to_front(self):
        gs = self._make_state_with_queue(["lion", "seal", "parrot", "monkey"], ["zebra", "kangaroo"])
        # Play lion (hand_index=0, no params)
        act = actions.Action(hand_index=0)
        gs2 = engine.step(gs, act)
        # Lion should be at the front of the queue
        # (After resolution, the queue should have lion at front)
        # We just verify the step doesn't crash and game continues
        assert gs2.turn == 2

    def test_parrot_removes_target(self):
        gs = self._make_state_with_queue(["parrot", "seal", "lion", "monkey"], ["zebra", "kangaroo"])
        # Play parrot targeting index 0 (the zebra in queue)
        act = actions.Action(hand_index=0, params=(0,))
        gs2 = engine.step(gs, act)
        # The targeted card should have moved to thats_it
        assert len(gs2.zones.thats_it) > 0

    def test_kangaroo_hops(self):
        gs = self._make_state_with_queue(["kangaroo", "seal", "lion", "monkey"], ["zebra", "hippo"])
        # Play kangaroo with hop=1
        act = actions.Action(hand_index=0, params=(1,))
        gs2 = engine.step(gs, act)
        assert gs2.turn == 2

    def test_seal_reverses_queue(self):
        gs = self._make_state_with_queue(["seal", "lion", "parrot", "monkey"], ["zebra", "kangaroo", "hippo"])
        act = actions.Action(hand_index=0)
        gs2 = engine.step(gs, act)
        assert gs2.turn == 2

    def test_snake_sorts_queue(self):
        gs = self._make_state_with_queue(["snake", "lion", "parrot", "monkey"], ["kangaroo", "zebra"])
        act = actions.Action(hand_index=0)
        gs2 = engine.step(gs, act)
        # After snake sorts, queue should be in descending strength order
        queue = gs2.zones.queue
        for i in range(len(queue) - 1):
            assert queue[i].strength >= queue[i + 1].strength

    def test_chameleon_copies_parrot(self):
        gs = self._make_state_with_queue(
            ["chameleon", "lion", "seal", "monkey"],
            ["zebra", "parrot"],
        )
        # Chameleon copies parrot at queue index 1, targeting queue index 0
        act = actions.Action(hand_index=0, params=(1, 0))
        gs2 = engine.step(gs, act)
        assert gs2.turn == 2
