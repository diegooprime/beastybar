from beastybar import engine, state


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
        "bounced": [],
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
        "bounced": [],
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
        "bounced": [],
        "beasty_bar": [],
        "hands": [
            ["seal", "zebra", "parrot", "giraffe"],
            ["giraffe", "crocodile", "zebra", "skunk"],
        ],
    },
    {
        "turn": 3,
        "active_player": 1,
        "queue": ["kangaroo", "monkey", "seal"],
        "thats_it": [],
        "bounced": [],
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
        "thats_it": ["seal", "monkey", "kangaroo"],
        "bounced": [],
        "beasty_bar": [],
        "hands": [
            ["zebra", "parrot", "giraffe", "snake"],
            ["giraffe", "zebra", "skunk", "kangaroo"],
        ],
    },
]


def test_seeded_two_turn_replay_matches_golden_expectations():
    game = state.initial_state(seed=2025)
    snapshots = [_snapshot(game)]

    for species, params in PLAN:
        player = game.active_player
        hand = game.players[player].hand
        index = _find_hand_index(hand, species)
        legal = [act for act in engine.legal_actions(game, player) if act.hand_index == index]
        if not legal:
            raise AssertionError(f"No legal actions for {species}")

        if params is None:
            action = legal[0]
        else:
            try:
                action = next(act for act in legal if act.params == params)
            except StopIteration as exc:
                raise AssertionError(f"No legal action for {species} with params {params}") from exc

        game = engine.step(game, action)
        snapshots.append(_snapshot(game))

    assert snapshots == EXPECTED_SNAPSHOTS


def _snapshot(game: state.State) -> dict:
    return {
        "turn": game.turn,
        "active_player": game.active_player,
        "queue": [card.species for card in game.zones.queue],
        "thats_it": [card.species for card in game.zones.thats_it],
        "bounced": [card.species for card in game.zones.bounced],
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
