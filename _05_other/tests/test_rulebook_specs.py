from _01_simulator import actions, engine, state


def make_card(owner: int, species: str) -> state.Card:
    return state.Card(owner=owner, species=species)


def build_state(hand, queue):
    players = (
        state.PlayerState(deck=(), hand=tuple(hand)),
        state.PlayerState(deck=(), hand=()),
    )
    zones = state.Zones(queue=tuple(queue))
    return state.State(seed=99, turn=0, active_player=0, players=players, zones=zones)


def test_rulebook_lion_scares_monkey_troop():
    lion = make_card(0, "lion")
    queue = [
        make_card(1, "monkey"),
        make_card(1, "hippo"),
        make_card(0, "monkey"),
        make_card(1, "zebra"),
        make_card(0, "monkey"),
    ]
    game_state = build_state([lion], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert [c.species for c in next_state.zones.queue] == ["lion", "hippo", "zebra"]
    assert [c.species for c in next_state.zones.thats_it] == ["monkey", "monkey", "monkey"]


def test_rulebook_monkey_pair_bounces_big_carnivores():
    new_monkey = make_card(0, "monkey")
    queue = [
        make_card(1, "monkey"),
        make_card(1, "hippo"),
        make_card(1, "zebra"),
        make_card(0, "crocodile"),
    ]
    game_state = build_state([new_monkey], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert [c.species for c in next_state.zones.queue] == ["monkey", "monkey", "zebra"]
    assert [c.species for c in next_state.zones.thats_it] == ["hippo", "crocodile"]


def test_rulebook_zebra_blocks_hippo_and_croc_chain():
    hippo = make_card(0, "hippo")
    queue = [
        make_card(1, "parrot"),
        make_card(1, "zebra"),
        make_card(0, "crocodile"),
    ]
    game_state = build_state([hippo], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert [c.species for c in next_state.zones.queue] == ["parrot", "zebra", "hippo", "crocodile"]
    assert [c.species for c in next_state.zones.thats_it] == []


def test_rulebook_crocodile_eats_until_zebra():
    crocodile = make_card(0, "crocodile")
    queue = [
        make_card(1, "parrot"),
        make_card(1, "kangaroo"),
        make_card(1, "zebra"),
        make_card(0, "monkey"),
    ]
    game_state = build_state([crocodile], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert [c.species for c in next_state.zones.queue] == ["parrot", "kangaroo", "zebra", "crocodile"]
    assert [c.species for c in next_state.zones.thats_it] == ["monkey"]


def test_rulebook_crocodile_eats_skunk_and_weaker_cards():
    crocodile = make_card(0, "crocodile")
    queue = [
        make_card(1, "giraffe"),
        make_card(1, "skunk"),
    ]
    game_state = build_state([crocodile], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert [c.species for c in next_state.zones.queue] == ["crocodile"]
    assert {c.species for c in next_state.zones.thats_it} == {"giraffe", "skunk"}


def test_rulebook_snake_orders_by_strength():
    snake = make_card(0, "snake")
    queue = [
        make_card(1, "kangaroo"),
        make_card(0, "zebra"),
        make_card(1, "parrot"),
    ]
    game_state = build_state([snake], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert [c.species for c in next_state.zones.queue] == ["snake", "zebra", "kangaroo", "parrot"]
    assert [c.species for c in next_state.zones.thats_it] == []


def test_rulebook_giraffe_threads_forward_through_weaker_cards():
    giraffe = make_card(0, "giraffe")
    queue = [
        make_card(1, "parrot"),
        make_card(1, "monkey"),
    ]
    game_state = build_state([giraffe], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert [c.species for c in next_state.zones.queue] == ["parrot", "giraffe", "monkey"]


def test_rulebook_kangaroo_hop_two_slots():
    kangaroo = make_card(0, "kangaroo")
    queue = [
        make_card(1, "parrot"),
        make_card(1, "zebra"),
        make_card(0, "chameleon"),
    ]
    game_state = build_state([kangaroo], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0, params=(2,)))

    assert [c.species for c in next_state.zones.queue] == ["parrot", "kangaroo", "zebra", "chameleon"]


def test_rulebook_parrot_knocks_selected_target():
    parrot = make_card(0, "parrot")
    queue = [
        make_card(1, "lion"),
        make_card(1, "zebra"),
    ]
    game_state = build_state([parrot], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0, params=(0,)))

    # Parrot stays where it entered (at back), doesn't reposition
    assert [c.species for c in next_state.zones.queue] == ["zebra", "parrot"]
    assert [c.species for c in next_state.zones.thats_it] == ["lion"]


def test_rulebook_seal_reverses_entire_queue():
    seal = make_card(0, "seal")
    queue = [
        make_card(1, "parrot"),
        make_card(0, "lion"),
        make_card(1, "zebra"),
    ]
    game_state = build_state([seal], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert [c.species for c in next_state.zones.queue] == ["seal", "zebra", "lion", "parrot"]


def test_rulebook_chameleon_copy_parrot_to_bounce_zebra():
    chameleon = make_card(0, "chameleon")
    queue = [
        make_card(1, "parrot"),
        make_card(1, "zebra"),
        make_card(0, "kangaroo"),
    ]
    game_state = build_state([chameleon], queue)

    # Chameleon copies parrot at index 0, targets zebra at index 1
    next_state = engine.step(game_state, actions.Action(hand_index=0, params=(0, 1)))

    # Chameleon stays where it entered (at back), doesn't reposition
    assert [c.species for c in next_state.zones.queue] == ["parrot", "kangaroo", "chameleon"]
    assert [c.species for c in next_state.zones.thats_it] == ["zebra"]


def test_rulebook_skunk_removes_top_two_strength_bands():
    skunk = make_card(0, "skunk")
    queue = [
        make_card(1, "lion"),
        make_card(1, "hippo"),
        make_card(0, "crocodile"),
    ]
    game_state = build_state([skunk], queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    # Skunk stays in place after expelling (no repositioning)
    assert [c.species for c in next_state.zones.queue] == ["crocodile", "skunk"]
    assert [c.species for c in next_state.zones.thats_it] == ["lion", "hippo"]
