from beastybar.simulator import actions, cards, engine, state


def make_card(owner: int, species: str) -> state.Card:
    return state.Card(owner=owner, species=species)


def make_state(active_hand, queue, *, beasty=None, thats_it=None):
    player0 = state.PlayerState(deck=(), hand=tuple(active_hand))
    player1 = state.PlayerState(deck=(), hand=())
    zones = state.Zones(
        queue=tuple(queue),
        beasty_bar=tuple(beasty or ()),
        thats_it=tuple(thats_it or ()),
    )
    return state.State(seed=0, turn=0, active_player=0, players=(player0, player1), zones=zones)


def test_lion_scares_monkeys_and_takes_front():
    lion = make_card(0, "lion")
    monkey_a = make_card(0, "monkey")
    hippo = make_card(1, "hippo")
    monkey_b = make_card(1, "monkey")

    game_state = make_state([lion], [monkey_a, hippo, monkey_b])

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert next_state.zones.queue[0] is lion
    assert next_state.zones.queue[1] is hippo
    assert next_state.zones.thats_it == (monkey_a, monkey_b)


def test_new_lion_bounced_if_one_exists():
    existing_lion = make_card(1, "lion")
    hippo = make_card(1, "hippo")
    new_lion = make_card(0, "lion")

    game_state = make_state([new_lion], [existing_lion, hippo])

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert next_state.zones.queue == (existing_lion, hippo)
    assert next_state.zones.thats_it[-1] is new_lion


def test_snake_sorts_by_strength_descending():
    snake = make_card(0, "snake")
    parrot = make_card(0, "parrot")
    zebra = make_card(1, "zebra")
    crocodile = make_card(1, "crocodile")

    game_state = make_state([snake], [parrot, zebra, crocodile])

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert next_state.zones.queue == (crocodile, snake, zebra, parrot)


def test_giraffe_jumps_over_one_weaker_animal():
    giraffe = make_card(0, "giraffe")
    parrot = make_card(1, "parrot")

    game_state = make_state([giraffe], [parrot])

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert next_state.zones.queue == (giraffe, parrot)


def test_kangaroo_jumps_up_to_two_positions():
    kangaroo = make_card(0, "kangaroo")
    parrot = make_card(1, "parrot")
    zebra = make_card(1, "zebra")
    chameleon = make_card(1, "chameleon")

    game_state = make_state([kangaroo], [parrot, zebra, chameleon])

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert next_state.zones.queue == (parrot, kangaroo, zebra, chameleon)


def test_hippo_passes_all_weaker_animals():
    hippo = make_card(0, "hippo")
    parrot = make_card(1, "parrot")
    chameleon = make_card(1, "chameleon")
    queue_state = make_state([], [parrot, chameleon, hippo])

    next_state = cards.process_recurring(queue_state)

    assert next_state.zones.queue == (hippo, parrot, chameleon)


def test_hippo_stopped_by_zebra():
    hippo = make_card(0, "hippo")
    zebra = make_card(1, "zebra")
    parrot = make_card(1, "parrot")
    queue_state = make_state([], [parrot, zebra, hippo])

    next_state = cards.process_recurring(queue_state)

    assert next_state.zones.queue == (parrot, zebra, hippo)


def test_crocodile_eats_until_blocked():
    crocodile = make_card(0, "crocodile")
    parrot = make_card(1, "parrot")
    chameleon = make_card(1, "chameleon")
    queue_state = make_state([], [parrot, chameleon, crocodile])

    next_state = cards.process_recurring(queue_state)

    assert next_state.zones.queue == (crocodile,)
    assert next_state.zones.thats_it == (chameleon, parrot)


def test_crocodile_blocked_by_zebra():
    crocodile = make_card(0, "crocodile")
    parrot = make_card(1, "parrot")
    zebra = make_card(1, "zebra")
    queue_state = make_state([], [parrot, zebra, crocodile])

    next_state = cards.process_recurring(queue_state)

    assert next_state.zones.queue == (parrot, zebra, crocodile)
    assert next_state.zones.thats_it == ()


def test_monkey_bounces_hippos_and_crocs():
    monkey_old = make_card(1, "monkey")
    hippo = make_card(0, "hippo")
    zebra = make_card(1, "zebra")
    crocodile = make_card(0, "crocodile")
    monkey_new = make_card(0, "monkey")

    game_state = make_state([monkey_new], [monkey_old, hippo, zebra, crocodile])

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert next_state.zones.queue == (monkey_new, monkey_old, zebra)
    assert tuple(next_state.zones.thats_it[-2:]) == (hippo, crocodile)


def test_parrot_bounces_selected_target():
    parrot = make_card(0, "parrot")
    lion = make_card(1, "lion")

    game_state = make_state([parrot], [lion])

    next_state = engine.step(game_state, actions.Action(hand_index=0, params=(0,)))

    assert next_state.zones.queue == (parrot,)
    assert next_state.zones.thats_it[-1] is lion


def test_seal_reverses_queue():
    seal = make_card(0, "seal")
    parrot = make_card(1, "parrot")
    zebra = make_card(1, "zebra")

    game_state = make_state([seal], [parrot, zebra])

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert next_state.zones.queue == (seal, zebra, parrot)
    assert next_state.zones.thats_it == ()


def test_skunk_expels_top_two_strength_bands():
    skunk = make_card(0, "skunk")
    lion = make_card(1, "lion")
    crocodile_a = make_card(1, "crocodile")
    crocodile_b = make_card(0, "crocodile")
    snake = make_card(0, "snake")

    game_state = make_state([skunk], [lion, crocodile_a, snake, crocodile_b])

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert set(next_state.zones.queue) == {skunk, snake}
    assert set(next_state.zones.thats_it) == {lion, crocodile_a, crocodile_b}


def test_chameleon_copy_lion_causes_self_bounce():
    chameleon_card = make_card(0, "chameleon")
    lion = make_card(1, "lion")

    game_state = make_state([chameleon_card], [lion])

    next_state = engine.step(game_state, actions.Action(hand_index=0, params=(0,)))

    assert next_state.zones.queue == (lion,)
    assert chameleon_card in next_state.zones.thats_it


def test_chameleon_copy_parrot_uses_extra_params():
    chameleon_card = make_card(0, "chameleon")
    parrot = make_card(1, "parrot")
    zebra = make_card(1, "zebra")

    game_state = make_state([chameleon_card], [parrot, zebra])

    next_state = engine.step(game_state, actions.Action(hand_index=0, params=(0, 1)))

    assert next_state.zones.queue == (parrot, chameleon_card)
    assert next_state.zones.thats_it[-1] is zebra


def test_five_card_check_triggers_after_resolution():
    cards_in_queue = [
        make_card(0, "parrot"),
        make_card(0, "snake"),
        make_card(1, "kangaroo"),
        make_card(1, "chameleon"),
    ]
    lion = make_card(0, "lion")
    game_state = make_state([lion], cards_in_queue)

    next_state = engine.step(game_state, actions.Action(hand_index=0))

    assert len(next_state.zones.beasty_bar) == 2
    assert len(next_state.zones.thats_it) >= 1
