from fastapi.testclient import TestClient

from _04_ui import create_app


def test_new_game_and_action_cycle():
    app = create_app()
    client = TestClient(app)

    response = client.post("/api/new-game", json={"seed": 2025})
    assert response.status_code == 200
    initial = response.json()
    assert initial["turn"] == 0
    assert initial["legalActions"], "Expected initial legal actions"

    action_payload = initial["legalActions"][0]
    play = client.post("/api/action", json=action_payload)
    assert play.status_code == 200
    updated = play.json()
    assert updated["turn"] == 1
    assert updated["activePlayer"] == 1

    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200
    assert state_resp.json() == updated


def test_static_index_served():
    app = create_app()
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Beasty Bar" in response.text


def test_turn_flow_and_hidden_information():
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/new-game",
        json={"seed": 314, "opponent": "random", "humanPlayer": 0},
    )
    assert response.status_code == 200
    state = response.json()

    assert state["turn"] == 0
    assert state["turnFlow"] == []

    # Human player's hand remains visible; opponent hand masked as "unknown".
    human_index = state["humanPlayer"]
    human_hand = state["hands"][human_index]
    opponent_hand = state["hands"][1 - human_index]
    assert any(card["species"] != "unknown" for card in human_hand)
    assert all(card["species"] == "unknown" for card in opponent_hand)

    action_payload = state["legalActions"][0]
    play = client.post("/api/action", json=action_payload)
    assert play.status_code == 200
    after_play = play.json()

    flow = after_play.get("turnFlow")
    assert isinstance(flow, list) and len(flow) == 5
    step_names = [step.get("name") for step in flow]
    assert step_names == ["play", "resolve", "recurring", "five-animal check", "draw"]
    assert any("Played" in event for event in flow[0]["events"])


def test_human_vs_human_hands_visible():
    app = create_app()
    client = TestClient(app)

    response = client.post("/api/new-game", json={"seed": 8080})
    assert response.status_code == 200
    state = response.json()

    assert state["humanPlayer"] == 0  # default selection
    player0, player1 = state["hands"]
    assert any(card["species"] != "unknown" for card in player0)
    assert any(card["species"] != "unknown" for card in player1)


def test_replay_endpoint_matches_recorded_sequence():
    app = create_app()
    client = TestClient(app)

    start = client.post("/api/new-game", json={"seed": 2718})
    assert start.status_code == 200
    state = start.json()

    first = client.post("/api/action", json=state["legalActions"][0])
    assert first.status_code == 200
    second_state = first.json()

    second = client.post("/api/action", json=second_state["legalActions"][0])
    assert second.status_code == 200
    latest_state = second.json()

    replay = client.post("/api/replay")
    assert replay.status_code == 200
    assert replay.json() == latest_state
