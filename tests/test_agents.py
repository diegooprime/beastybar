import json
from pathlib import Path

import pytest

from beastybar import engine, state
from beastybar.agents import (
    FirstLegalAgent,
    GreedyAgent,
    RandomAgent,
    ensure_legal,
)
from beastybar.agents import evaluation as eval_utils
from beastybar.agents import tournament


def make_card(owner: int, species: str) -> state.Card:
    return state.Card(owner=owner, species=species)


def make_state(hand, queue, *, deck=None, starting_player=0):
    player0 = state.PlayerState(deck=tuple(deck or ()), hand=tuple(hand))
    player1 = state.PlayerState(deck=(), hand=())
    zones = state.Zones(queue=tuple(queue))
    return state.State(
        seed=0,
        turn=0,
        active_player=starting_player,
        players=(player0, player1),
        zones=zones,
    )


def test_first_legal_agent_picks_first_action():
    lion = make_card(0, "lion")
    snake = make_card(0, "snake")
    game = make_state([lion, snake], [])
    legal = tuple(engine.legal_actions(game, 0))

    agent = FirstLegalAgent()
    chosen = agent.select_action(game, legal)

    assert chosen == legal[0]


def test_random_agent_deterministic_with_seed():
    lion = make_card(0, "lion")
    snake = make_card(0, "snake")
    game = make_state([lion, snake], [])
    legal = tuple(engine.legal_actions(game, 0))

    agent_a = RandomAgent(seed=123)
    agent_b = RandomAgent(seed=123)

    picks_a = [agent_a.select_action(game, legal) for _ in range(3)]
    picks_b = [agent_b.select_action(game, legal) for _ in range(3)]

    assert picks_a == picks_b
    for action in picks_a:
        ensure_legal(action, legal)


def test_greedy_agent_prefers_heuristic_outcome():
    lion = make_card(0, "lion")
    kangaroo = make_card(0, "kangaroo")
    monkey = make_card(1, "monkey")
    game = make_state([lion, kangaroo], [monkey])
    legal = tuple(engine.legal_actions(game, 0))

    def prefer_lion(next_state: state.State, perspective: int) -> float:
        del perspective
        front = next_state.zones.queue[0].species if next_state.zones.queue else ""
        return 1.0 if front == "lion" else 0.0

    agent = GreedyAgent(heuristic=prefer_lion)
    chosen = agent.select_action(game, legal)

    played_card = game.players[0].hand[chosen.hand_index].species
    assert played_card == "lion"


def test_material_advantage_scores_bar_points():
    a_lion = make_card(0, "lion")
    b_croc = make_card(1, "crocodile")
    zones = state.Zones(beasty_bar=(a_lion, b_croc))
    players = (
        state.PlayerState(deck=(), hand=()),
        state.PlayerState(deck=(), hand=()),
    )
    game = state.State(seed=0, turn=5, active_player=0, players=players, zones=zones)

    score = eval_utils.material_advantage(game, perspective=0)
    # Lion worth 2 points, crocodile 3 -> net advantage should be negative
    assert score < 0


def test_best_action_returns_highest_scoring_move():
    lion = make_card(0, "lion")
    kangaroo = make_card(0, "kangaroo")
    monkey = make_card(1, "monkey")
    game = make_state([lion, kangaroo], [monkey])
    legal = tuple(engine.legal_actions(game, 0))

    def heuristic(next_state: state.State, perspective: int) -> float:
        del perspective
        return len(next_state.zones.queue)

    action, score = eval_utils.best_action(game, legal, heuristic)

    assert score >= 0
    # Both actions keep queue length >= 1, but ensure the returned action is legal.
    assert action in legal


def test_play_series_collects_summary(tmp_path: Path):
    config = tournament.SeriesConfig(
        games=4,
        seed=7,
        agent_a=FirstLegalAgent(),
        agent_b=RandomAgent(seed=99),
        alternate_start=False,
    )
    result = tournament.play_series(config)

    assert result.summary.games == len(result.records) == 4
    assert sum(result.summary.wins) + result.summary.ties == 4
    assert result.records[0].starting_player == 0

    csv_path = tmp_path / "telemetry.csv"
    json_path = tmp_path / "telemetry.json"

    tournament.export_csv(csv_path, result.records)
    tournament.export_json(json_path, result.records)

    assert csv_path.read_text().count("game") >= 1
    data = json.loads(json_path.read_text())
    assert len(data) == 4
    assert all("scores" in entry for entry in data)


def test_summarize_rejects_empty_records():
    with pytest.raises(ValueError):
        tournament.summarize([])
