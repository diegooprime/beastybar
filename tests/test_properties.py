from beastybar import engine, rules, simulate, state


def _total_cards(game_state: state.State) -> int:
    player_cards = sum(len(player.hand) + len(player.deck) for player in game_state.players)
    shared_cards = (
        len(game_state.zones.queue)
        + len(game_state.zones.beasty_bar)
        + len(game_state.zones.thats_it)
    )
    return player_cards + shared_cards


def test_invariants_hold_under_default_play():
    expected_total = rules.DECK_SIZE * rules.PLAYER_COUNT
    for seed in range(20):
        game = state.initial_state(seed)
        while True:
            assert len(game.zones.queue) <= rules.MAX_QUEUE_LENGTH
            assert _total_cards(game) == expected_total
            if engine.is_terminal(game):
                break
            legal = tuple(engine.legal_actions(game, game.active_player))
            assert legal, "Active player must always have a legal move"
            game = engine.step(game, legal[0])

        scores = engine.score(game)
        bar_points = [0 for _ in range(rules.PLAYER_COUNT)]
        for card in game.zones.beasty_bar:
            bar_points[card.owner] += card.points
        assert scores == bar_points


def test_deterministic_runs_for_fixed_seed():
    for seed in range(10):
        final_a = next(simulate.run(simulate.SimulationConfig(seed=seed)))
        final_b = next(simulate.run(simulate.SimulationConfig(seed=seed)))
        assert final_a == final_b
