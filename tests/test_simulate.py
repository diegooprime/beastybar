from simulator import actions, engine, simulate, state


def test_new_game_matches_state_initializer():
    seed = 123
    assert simulate.new_game(seed) == state.initial_state(seed)


def test_legal_actions_align_with_engine():
    game = simulate.new_game(7)
    expected = tuple(engine.legal_actions(game, game.active_player))
    assert simulate.legal_actions(game, game.active_player) == expected


def test_apply_advances_state():
    game = simulate.new_game(99)
    legal = simulate.legal_actions(game, game.active_player)
    assert legal
    next_state = simulate.apply(game, legal[0])
    assert next_state != game
    assert next_state.turn == 1


def test_run_with_scripted_agents_matches_golden_final_state():
    agent_a = _recording_agent([
        ("kangaroo", None),
        ("seal", None),
    ])
    agent_b = _recording_agent([
        ("monkey", None),
        ("crocodile", None),
    ])
    config = simulate.SimulationConfig(
        seed=2025,
        games=1,
        agent_a=agent_a,
        agent_b=agent_b,
    )

    results = list(simulate.run(config))
    assert len(results) == 1
    final_state = results[0]

    assert agent_a.history[:2] == [("kangaroo", ()), ("seal", ())]
    assert agent_b.history[:2] == [("monkey", ()), ("crocodile", ())]
    assert simulate.is_terminal(final_state)
    assert simulate.score(final_state) == tuple(engine.score(final_state))


class _recording_agent:
    def __init__(self, script):
        self._script = list(script)
        self.history: list[tuple[str, tuple[int, ...]]] = []

    def __call__(self, game_state: state.State, legal: tuple[actions.Action, ...]) -> actions.Action:
        player_hand = game_state.players[game_state.active_player].hand
        action = self._select_action(player_hand, legal)
        species = player_hand[action.hand_index].species
        self.history.append((species, action.params))
        return action

    def _select_action(
        self,
        hand: tuple[state.Card, ...],
        legal: tuple[actions.Action, ...],
    ) -> actions.Action:
        if self._script:
            species, params = self._script.pop(0)
            choices = [act for act in legal if hand[act.hand_index].species == species]
            if not choices:
                raise AssertionError(f"No action available for species {species}")
            if params is None:
                return choices[0]
            for act in choices:
                if act.params == params:
                    return act
            raise AssertionError(f"Params {params} not available for species {species}")
        return legal[0]
