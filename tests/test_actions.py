import pytest

from beastybar import actions, engine, state


def make_card(owner: int, species: str) -> state.Card:
    return state.Card(owner=owner, species=species)


def make_game(hand, queue):
    player0 = state.PlayerState(deck=(), hand=tuple(hand))
    player1 = state.PlayerState(deck=(), hand=())
    return state.State(
        seed=0,
        turn=0,
        active_player=0,
        players=(player0, player1),
        zones=state.Zones(queue=tuple(queue)),
    )


def test_legal_actions_yields_parrot_targets():
    parrot = make_card(0, "parrot")
    queue = [make_card(1, "lion"), make_card(1, "snake")]
    game = make_game([parrot], queue)

    acts = list(engine.legal_actions(game, 0))
    assert actions.Action(hand_index=0, params=(0,)) in acts
    assert actions.Action(hand_index=0, params=(1,)) in acts
    assert len(acts) == 2


def test_legal_actions_skips_chameleon_when_no_targets():
    chameleon_card = make_card(0, "chameleon")
    game = make_game([chameleon_card], [])
    acts = list(engine.legal_actions(game, 0))
    assert acts == []


def test_legal_actions_chameleon_on_non_param_card():
    chameleon_card = make_card(0, "chameleon")
    kangaroo = make_card(1, "kangaroo")
    game = make_game([chameleon_card], [kangaroo])

    acts = list(engine.legal_actions(game, 0))
    assert actions.Action(hand_index=0, params=(0,)) in acts
    assert len(acts) == 1


def test_legal_actions_chameleon_on_parrot_adds_targets():
    chameleon_card = make_card(0, "chameleon")
    parrot = make_card(1, "parrot")
    zebra = make_card(1, "zebra")
    queue = [parrot, zebra]
    game = make_game([chameleon_card], queue)

    acts = set(engine.legal_actions(game, 0))
    expected = {
        actions.Action(hand_index=0, params=(0, 0)),
        actions.Action(hand_index=0, params=(0, 1)),
        actions.Action(hand_index=0, params=(1,)),
    }
    assert acts == expected


def test_validate_rejects_parrot_without_params():
    parrot = make_card(0, "parrot")
    queue = [make_card(1, "lion")]
    game = make_game([parrot], queue)

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=0))


def test_validate_rejects_chameleon_copy_self():
    chameleon_card = make_card(0, "chameleon")
    queue = [chameleon_card]
    game = make_game([chameleon_card], queue)

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=0, params=(0,)))


def test_validate_rejects_chameleon_missing_params_for_parrot():
    chameleon_card = make_card(0, "chameleon")
    parrot = make_card(1, "parrot")
    queue = [parrot]
    game = make_game([chameleon_card], queue)

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=0, params=(0,)))


def test_validate_rejects_extra_params_for_non_param_cards():
    lion = make_card(0, "lion")
    queue = []
    game = make_game([lion], queue)

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=0, params=(1,)))


def test_validate_rejects_parrot_target_out_of_range():
    parrot = make_card(0, "parrot")
    queue = [make_card(1, "lion")]
    game = make_game([parrot], queue)

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=0, params=(2,)))


def test_validate_rejects_chameleon_target_out_of_range():
    chameleon_card = make_card(0, "chameleon")
    zebra = make_card(1, "zebra")
    game = make_game([chameleon_card], [zebra])

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=0, params=(5,)))


def test_validate_rejects_chameleon_parrot_extra_param_out_of_range():
    chameleon_card = make_card(0, "chameleon")
    parrot = make_card(1, "parrot")
    zebra = make_card(1, "zebra")
    queue = [parrot, zebra]
    game = make_game([chameleon_card], queue)

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=0, params=(0, 99)))


def test_validate_rejects_chameleon_extra_params_for_non_param_card():
    chameleon_card = make_card(0, "chameleon")
    zebra = make_card(1, "zebra")
    game = make_game([chameleon_card], [zebra])

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=0, params=(0, 1)))


def test_validate_rejects_invalid_hand_index():
    lion = make_card(0, "lion")
    game = make_game([lion], [])

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=1))


def test_validate_rejects_chameleon_copying_chameleon():
    chameleon_new = make_card(0, "chameleon")
    chameleon_old = make_card(1, "chameleon")
    game = make_game([chameleon_new], [chameleon_old])

    with pytest.raises(ValueError):
        engine.step(game, actions.Action(hand_index=0, params=(0,)))
