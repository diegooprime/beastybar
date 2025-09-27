import json
from pathlib import Path

import pytest

from beastybar.simulator import engine, state
from beastybar.agents import (
    DiegoAgent,
    FirstLegalAgent,
    FrontRunnerAgent,
    GreedyAgent,
    KillerAgent,
    RandomAgent,
    ensure_legal,
)
from beastybar.agents import evaluation as eval_utils
from beastybar.training import tournament


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


def test_diego_skunk_skips_low_value_removal():
    skunk = make_card(0, "skunk")
    lion = make_card(0, "lion")
    opponent_monkey = make_card(1, "monkey")
    game = make_state([skunk, lion], [opponent_monkey])
    legal = tuple(engine.legal_actions(game, 0))

    agent = DiegoAgent()
    action = agent.select_action(game, legal)

    chosen_species = game.players[0].hand[action.hand_index].species
    assert chosen_species == "lion"


def test_diego_skunk_takes_high_value_removal():
    skunk = make_card(0, "skunk")
    lion = make_card(0, "lion")
    opponent_zebra = make_card(1, "zebra")
    game = make_state([skunk, lion], [opponent_zebra])
    legal = tuple(engine.legal_actions(game, 0))

    agent = DiegoAgent()
    action = agent.select_action(game, legal)

    chosen_species = game.players[0].hand[action.hand_index].species
    assert chosen_species == "skunk"


def test_diego_parrot_targets_high_value_card():
    parrot = make_card(0, "parrot")
    low_target = make_card(1, "monkey")
    high_target = make_card(1, "zebra")
    game = make_state([parrot], [low_target, high_target])
    legal = tuple(engine.legal_actions(game, 0))

    agent = DiegoAgent()
    action = agent.select_action(game, legal)

    assert action.params == (1,)


def test_diego_crocodile_prefers_threshold_removal():
    agent = DiegoAgent()

    crocodile_a = make_card(0, "crocodile")
    lion_a = make_card(0, "lion")
    queue_high = [make_card(1, "seal"), make_card(1, "monkey")]
    game_high = make_state([crocodile_a, lion_a], queue_high)
    legal_high = tuple(engine.legal_actions(game_high, 0))
    action_high = agent.select_action(game_high, legal_high)
    chosen_high = game_high.players[0].hand[action_high.hand_index].species
    assert chosen_high == "crocodile"

    crocodile_b = make_card(0, "crocodile")
    lion_b = make_card(0, "lion")
    queue_low = [make_card(1, "monkey")]
    game_low = make_state([crocodile_b, lion_b], queue_low)
    legal_low = tuple(engine.legal_actions(game_low, 0))
    action_low = agent.select_action(game_low, legal_low)
    chosen_low = game_low.players[0].hand[action_low.hand_index].species
    assert chosen_low == "lion"


def test_diego_seal_requires_value_delivery():
    agent = DiegoAgent()

    seal_a = make_card(0, "seal")
    lion_a = make_card(0, "lion")
    queue_value = [
        make_card(1, "monkey"),
        make_card(1, "zebra"),
        make_card(1, "snake"),
        make_card(0, "kangaroo"),
    ]
    game_value = make_state([seal_a, lion_a], queue_value)
    legal_value = tuple(engine.legal_actions(game_value, 0))
    action_value = agent.select_action(game_value, legal_value)
    chosen_value = game_value.players[0].hand[action_value.hand_index].species
    assert chosen_value == "seal"

    seal_b = make_card(0, "seal")
    lion_b = make_card(0, "lion")
    queue_neutral = [make_card(1, "monkey"), make_card(1, "snake")]
    game_neutral = make_state([seal_b, lion_b], queue_neutral)
    legal_neutral = tuple(engine.legal_actions(game_neutral, 0))
    action_neutral = agent.select_action(game_neutral, legal_neutral)
    chosen_neutral = game_neutral.players[0].hand[action_neutral.hand_index].species
    assert chosen_neutral == "lion"


def test_diego_zebra_falls_back_when_blocked():
    zebra = make_card(0, "zebra")
    opponent_hippo = make_card(1, "hippo")
    game = make_state([zebra], [opponent_hippo])
    legal = tuple(engine.legal_actions(game, 0))

    agent = DiegoAgent()
    action = agent.select_action(game, legal)

    assert action == legal[0]


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


def test_play_series_collects_action_telemetry():
    config = tournament.SeriesConfig(
        games=1,
        seed=5,
        agent_a=FirstLegalAgent(),
        agent_b=RandomAgent(seed=321),
        collect_actions=True,
    )

    result = tournament.play_series(config)
    record = result.records[0]

    assert record.actions is not None
    assert len(record.actions) > 0
    first_action = record.actions[0]
    assert first_action.was_pass is False
    assert first_action.hand_before
    assert first_action.queue_after != ()


def test_round_robin_cli_writes_logs(tmp_path: Path):
    log_dir = tmp_path / "logs"
    result = tournament.main(
        ["--round-robin", "--games", "1", "--seed", "42", "--log-dir", str(log_dir)]
    )

    assert result.summary.games > 0
    assert log_dir.exists()
    files = list(log_dir.glob("*.json"))
    assert files
    sample = json.loads(files[0].read_text())
    assert sample and "actions" in sample[0]


def test_summarize_rejects_empty_records():
    with pytest.raises(ValueError):
        tournament.summarize([])


def test_frontrunner_prefers_queue_control():
    crocodile = make_card(0, "crocodile")
    kangaroo = make_card(0, "kangaroo")
    monkey = make_card(1, "monkey")
    snake = make_card(1, "snake")
    game = make_state([crocodile, kangaroo], [monkey, snake])
    legal = tuple(engine.legal_actions(game, 0))

    agent = FrontRunnerAgent()
    chosen = agent.select_action(game, legal)

    species = game.players[0].hand[chosen.hand_index].species
    assert species == "crocodile"


def test_frontrunner_rejects_lion_when_blocked():
    lion = make_card(0, "lion")
    snake = make_card(0, "snake")
    opposing_lion = make_card(1, "lion")
    game = make_state([lion, snake], [opposing_lion])
    legal = tuple(engine.legal_actions(game, 0))

    agent = FrontRunnerAgent()
    chosen = agent.select_action(game, legal)

    species = game.players[0].hand[chosen.hand_index].species
    assert species != "lion"


def test_frontrunner_falls_back_to_first_legal_when_all_rejected():
    lion = make_card(0, "lion")
    opposing_lion = make_card(1, "lion")
    game = make_state([lion], [opposing_lion])
    legal = tuple(engine.legal_actions(game, 0))

    agent = FrontRunnerAgent()
    chosen = agent.select_action(game, legal)

    fallback = FirstLegalAgent().select_action(game, legal)
    assert chosen == fallback


def test_killer_prioritizes_high_point_removal():
    skunk = make_card(0, "skunk")
    parrot = make_card(0, "parrot")
    opponent_zebra = make_card(1, "zebra")
    opponent_parrot = make_card(1, "parrot")
    game = make_state([skunk, parrot], [opponent_zebra, opponent_parrot])
    legal = tuple(engine.legal_actions(game, 0))

    agent = KillerAgent()
    action = agent.select_action(game, legal)

    chosen_species = game.players[0].hand[action.hand_index].species
    assert chosen_species == "skunk"


def test_killer_ignores_own_losses():
    skunk = make_card(0, "skunk")
    kangaroo = make_card(0, "kangaroo")
    our_seal = make_card(0, "seal")
    opponent_zebra = make_card(1, "zebra")
    opponent_monkey = make_card(1, "monkey")
    game = make_state([skunk, kangaroo], [our_seal, opponent_zebra, opponent_monkey])
    legal = tuple(engine.legal_actions(game, 0))

    agent = KillerAgent()
    action = agent.select_action(game, legal)

    chosen_species = game.players[0].hand[action.hand_index].species
    assert chosen_species == "skunk"


def test_killer_falls_back_when_no_opponent_loss():
    lion = make_card(0, "lion")
    snake = make_card(0, "snake")
    game = make_state([lion, snake], [])
    legal = tuple(engine.legal_actions(game, 0))

    agent = KillerAgent()
    chosen = agent.select_action(game, legal)

    assert chosen == legal[0]


def test_should_stop_early_when_wilson_confident():
    config = tournament.EarlyStopConfig(min_games=5)
    assert tournament._should_stop_early(wins_a=5, wins_b=0, ties=0, config=config)
    assert not tournament._should_stop_early(wins_a=3, wins_b=2, ties=0, config=config)


def test_play_series_respects_early_stop(monkeypatch):
    triggered_at = 3

    def fake_should_stop(*, wins_a: int, wins_b: int, ties: int, config):
        total = wins_a + wins_b + ties
        return total >= triggered_at

    monkeypatch.setattr(tournament, "_should_stop_early", fake_should_stop)

    config = tournament.SeriesConfig(
        games=20,
        seed=11,
        agent_a=FirstLegalAgent(),
        agent_b=RandomAgent(seed=99),
        alternate_start=False,
        early_stop=tournament.EarlyStopConfig(min_games=1),
    )

    result = tournament.play_series(config)

    assert len(result.records) == triggered_at
    assert result.summary.games == triggered_at
