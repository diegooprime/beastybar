from beastybar import rules, state


def test_initial_state_setup():
    game_state = state.initial_state(seed=42)

    assert game_state.turn == 0
    assert game_state.active_player == 0
    assert len(game_state.players) == rules.PLAYER_COUNT

    for player_state in game_state.players:
        assert len(player_state.hand) == rules.HAND_SIZE
        assert len(player_state.deck) == rules.DECK_SIZE - rules.HAND_SIZE

    assert game_state.zones.queue == ()
    assert game_state.zones.beasty_bar == ()
    assert game_state.zones.thats_it == ()
