from fastapi.testclient import TestClient

from beastybar.user_interface import create_app


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
