"""FastAPI application exposing the Beasty Bar simulator."""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from .. import actions, simulate, state


@dataclass
class SessionStore:
    """In-memory holder for the current game state."""

    state: Optional[state.State] = None
    seed: Optional[int] = None

    def require_state(self) -> state.State:
        if self.state is None:
            raise HTTPException(status_code=404, detail="No active game")
        return self.state


class NewGameRequest(BaseModel):
    seed: Optional[int] = None
    starting_player: int = Field(default=0, ge=0, le=1)


class ActionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    hand_index: int = Field(alias="handIndex")
    params: List[int] = Field(default_factory=list)


def create_app() -> FastAPI:
    store = SessionStore()
    app = FastAPI(title="Beasty Bar Simulator", version="0.1.0")

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.post("/api/new-game")
    def api_new_game(request: NewGameRequest) -> dict:
        seed = request.seed if request.seed is not None else secrets.randbits(32)
        store.seed = seed
        store.state = simulate.new_game(seed, starting_player=request.starting_player)
        return _serialize(store.require_state(), store.seed)

    @app.get("/api/state")
    def api_state() -> dict:
        return _serialize(store.require_state(), store.seed)

    @app.get("/api/legal-actions")
    def api_legal_actions() -> List[dict]:
        game_state = store.require_state()
        player = game_state.active_player
        legal = simulate.legal_actions(game_state, player)
        return [_serialize_action(game_state, action) for action in legal]

    @app.post("/api/action")
    def api_action(payload: ActionPayload) -> dict:
        game_state = store.require_state()
        player = game_state.active_player
        legal = simulate.legal_actions(game_state, player)
        params_tuple = tuple(payload.params)
        try:
            action = next(
                act for act in legal if act.hand_index == payload.hand_index and act.params == params_tuple
            )
        except StopIteration as exc:
            raise HTTPException(status_code=400, detail="Illegal action") from exc

        store.state = simulate.apply(game_state, action)
        return _serialize(store.require_state(), store.seed)

    return app


def _serialize(game_state: state.State, seed: Optional[int]) -> dict:
    active_player = game_state.active_player
    legal = simulate.legal_actions(game_state, active_player) if not simulate.is_terminal(game_state) else ()
    return {
        "seed": seed,
        "turn": game_state.turn,
        "activePlayer": active_player,
        "isTerminal": simulate.is_terminal(game_state),
        "score": simulate.score(game_state) if simulate.is_terminal(game_state) else None,
        "queue": [_card_view(card) for card in game_state.zones.queue],
        "zones": {
            "beastyBar": [_card_view(card) for card in game_state.zones.beasty_bar],
            "thatsIt": [_card_view(card) for card in game_state.zones.thats_it],
        },
        "hands": [
            [_card_view(card) for card in player_state.hand]
            for player_state in game_state.players
        ],
        "legalActions": [_serialize_action(game_state, action) for action in legal],
    }


def _serialize_action(game_state: state.State, action: actions.Action) -> dict:
    player = game_state.active_player
    card = game_state.players[player].hand[action.hand_index]
    return {
        "handIndex": action.hand_index,
        "params": list(action.params),
        "card": _card_view(card),
        "label": _action_label(card, action),
    }


def _card_view(card: state.Card) -> dict:
    return {
        "owner": card.owner,
        "species": card.species,
        "strength": card.strength,
        "points": card.points,
    }


def _action_label(card: state.Card, action: actions.Action) -> str:
    species = card.species
    if species == "kangaroo":
        if action.params:
            hop = action.params[0]
            return f"Play {species} (hop {hop})"
        return f"Play {species}"
    if action.params:
        params = ",".join(str(p) for p in action.params)
        return f"Play {species} ({params})"
    return f"Play {species}"


__all__ = ["create_app"]
