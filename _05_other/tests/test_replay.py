from _01_simulator import engine, state


PLAN = [
    ("kangaroo", None),
    ("monkey", None),
    ("seal", None),
    ("crocodile", None),
]

EXPECTED_SNAPSHOTS = [
    {
        "turn": 0,
        "active_player": 0,
        "queue": [],
        "thats_it": [],
        "beasty_bar": [],
        "hands": [
            ["seal", "zebra", "kangaroo", "parrot"],
            ["monkey", "giraffe", "crocodile", "zebra"],
        ],
    },
    {
        "turn": 1,
        "active_player": 1,
        "queue": ["kangaroo"],
        "thats_it": [],
        "beasty_bar": [],
        "hands": [
            ["seal", "zebra", "parrot", "giraffe"],
            ["monkey", "giraffe", "crocodile", "zebra"],
        ],
    },
    {
        "turn": 2,
        "active_player": 0,
        "queue": ["kangaroo", "monkey"],
        "thats_it": [],
        "beasty_bar": [],
        "hands": [
            ["seal", "zebra", "parrot", "giraffe"],
            ["giraffe", "crocodile", "zebra", "skunk"],
        ],
    },
    {
        "turn": 3,
        "active_player": 1,
        "queue": ["seal", "monkey", "kangaroo"],
        "thats_it": [],
        "beasty_bar": [],
        "hands": [
            ["zebra", "parrot", "giraffe", "snake"],
            ["giraffe", "crocodile", "zebra", "skunk"],
        ],
    },
    {
        "turn": 4,
        "active_player": 0,
        "queue": ["crocodile"],
        "thats_it": ["kangaroo", "monkey", "seal"],
        "beasty_bar": [],
        "hands": [
            ["zebra", "parrot", "giraffe", "snake"],
            ["giraffe", "zebra", "skunk", "kangaroo"],
        ],
    },
]


def test_seeded_two_turn_replay_matches_golden_expectations():
    _, snapshots = _run_plan(state.initial_state(seed=2025), PLAN)
    assert snapshots == EXPECTED_SNAPSHOTS


def _run_plan(game: state.State, plan):
    snapshots = [_snapshot(game)]
    current = game
    for species, params in plan:
        player = current.active_player
        hand = current.players[player].hand
        index = _find_hand_index(hand, species)
        legal = [act for act in engine.legal_actions(current, player) if act.hand_index == index]
        if not legal:
            raise AssertionError(f"No legal actions for {species}")
        if params is None:
            action = legal[0]
        else:
            try:
                action = next(act for act in legal if act.params == params)
            except StopIteration as exc:
                raise AssertionError(f"No legal action for {species} with params {params}") from exc
        current = engine.step(current, action)
        snapshots.append(_snapshot(current))
    return current, snapshots

def _snapshot(game: state.State) -> dict:
    return {
        "turn": game.turn,
        "active_player": game.active_player,
        "queue": [card.species for card in game.zones.queue],
        "thats_it": [card.species for card in game.zones.thats_it],
        "beasty_bar": [card.species for card in game.zones.beasty_bar],
        "hands": [
            [card.species for card in player.hand]
            for player in game.players
        ],
    }


def _find_hand_index(hand, species: str) -> int:
    for idx, card in enumerate(hand):
        if card.species == species:
            return idx
    raise AssertionError(f"Species {species} not found in hand")



def _make_card(owner: int, species: str) -> state.Card:
    return state.Card(owner=owner, species=species)


def _custom_state(p0_hand, p1_hand, queue, *, seed: int = 0):
    return state.State(
        seed=seed,
        turn=0,
        active_player=0,
        players=(
            state.PlayerState(deck=(), hand=tuple(p0_hand)),
            state.PlayerState(deck=(), hand=tuple(p1_hand)),
        ),
        zones=state.Zones(queue=tuple(queue)),
    )



def test_replay_monkeys_clear_heavies():
    game = _custom_state(
        p0_hand=[_make_card(0, "monkey"), _make_card(0, "monkey")],
        p1_hand=[_make_card(1, "hippo")],
        queue=[_make_card(1, "crocodile"), _make_card(0, "zebra")],
    )
    plan = [("monkey", None), ("hippo", None), ("monkey", None)]

    _, snapshots = _run_plan(game, plan)
    expected = [
        {
            "turn": 0,
            "active_player": 0,
            "queue": ["crocodile", "zebra"],
            "thats_it": [],
            "beasty_bar": [],
            "hands": [["monkey", "monkey"], ["hippo"]],
        },
        {
            "turn": 1,
            "active_player": 1,
            "queue": ["crocodile", "zebra", "monkey"],
            "thats_it": [],
            "beasty_bar": [],
            "hands": [["monkey"], ["hippo"]],
        },
        {
            "turn": 2,
            "active_player": 0,
            "queue": ["crocodile", "zebra", "hippo", "monkey"],
            "thats_it": [],
            "beasty_bar": [],
            "hands": [["monkey"], []],
        },
        {
            "turn": 3,
            "active_player": 1,
            "queue": ["monkey", "monkey", "zebra"],
            "thats_it": ["crocodile", "hippo"],
            "beasty_bar": [],
            "hands": [[], []],
        },
    ]
    assert snapshots == expected



def test_replay_chameleon_parrot_deep_queue():
    game = _custom_state(
        p0_hand=[_make_card(0, "chameleon")],
        p1_hand=[_make_card(1, "skunk")],
        queue=[
            _make_card(1, "parrot"),
            _make_card(0, "zebra"),
            _make_card(1, "kangaroo"),
            _make_card(0, "giraffe"),
        ],
    )
    plan = [("chameleon", (0, 3))]

    _, snapshots = _run_plan(game, plan)
    expected = [
        {
            "turn": 0,
            "active_player": 0,
            "queue": ["parrot", "zebra", "kangaroo", "giraffe"],
            "thats_it": [],
            "beasty_bar": [],
            "hands": [["chameleon"], ["skunk"]],
        },
        {
            "turn": 1,
            "active_player": 1,
            "queue": ["parrot", "zebra", "kangaroo", "chameleon"],
            "thats_it": ["giraffe"],
            "beasty_bar": [],
            "hands": [[], ["skunk"]],
        },
    ]
    assert snapshots == expected



def test_replay_seal_triggers_recurring_after_flip():
    game = _custom_state(
        p0_hand=[_make_card(0, "seal")],
        p1_hand=[_make_card(1, "giraffe")],
        queue=[
            _make_card(1, "zebra"),
            _make_card(1, "parrot"),
            _make_card(0, "hippo"),
            _make_card(0, "crocodile"),
        ],
    )
    plan = [("seal", None)]

    _, snapshots = _run_plan(game, plan)
    expected = [
        {
            "turn": 0,
            "active_player": 0,
            "queue": ["zebra", "parrot", "hippo", "crocodile"],
            "thats_it": [],
            "beasty_bar": [],
            "hands": [["seal"], ["giraffe"]],
        },
        {
            "turn": 1,
            "active_player": 1,
            "queue": ["hippo", "crocodile", "parrot", "zebra"],
            "thats_it": ["seal"],
            "beasty_bar": [],
            "hands": [[], ["giraffe"]],
        },
    ]
    assert snapshots == expected
