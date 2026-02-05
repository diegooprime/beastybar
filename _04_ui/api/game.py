"""Game-related API routes."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException

from _01_simulator import engine, simulate
from _04_ui.core.session import GameSession  # noqa: TC001
from _04_ui.models.requests import NewGameRequest  # noqa: TC001
from _04_ui.services.game import log_new_game
from _04_ui.services.serializer import serialize

router = APIRouter(prefix="/api", tags=["game"])

# Store reference - will be set by app factory
_store: GameSession | None = None


def set_store(store: GameSession) -> None:
    """Set the session store for this router."""
    global _store
    _store = store


def get_store() -> GameSession:
    """Get the session store."""
    if _store is None:
        raise RuntimeError("Store not initialized")
    return _store


@router.post("/new-game")
def api_new_game(request: NewGameRequest) -> dict:
    """Start a new game."""
    store = get_store()
    seed = request.seed if request.seed is not None else secrets.randbits(32)
    store.seed = seed
    store.human_player = request.human_player
    store.starting_player = request.starting_player
    store.ai_opponent = request.ai_opponent
    store.state = simulate.new_game(seed, starting_player=request.starting_player)
    store.log.clear()
    store.history.clear()
    log_new_game(store, starting_player=request.starting_player)

    return serialize(store.require_state(), store.seed, store)


@router.get("/state")
def api_state() -> dict:
    """Get current game state."""
    store = get_store()
    return serialize(store.require_state(), store.seed, store)


@router.get("/legal-actions")
def api_legal_actions() -> list[dict]:
    """Get legal actions for the current player."""
    from _04_ui.services.serializer import serialize_action

    store = get_store()
    game_state = store.require_state()
    player = game_state.active_player
    legal = simulate.legal_actions(game_state, player)
    return [serialize_action(game_state, action) for action in legal]


@router.post("/replay")
def api_replay() -> dict:
    """Verify the game history by replaying it."""
    store = get_store()
    current = store.require_state()
    if store.seed is None:
        raise HTTPException(status_code=400, detail="No seed available to replay")

    candidate = simulate.new_game(store.seed, starting_player=store.starting_player)
    for index, action in enumerate(store.history, start=1):
        try:
            candidate = engine.step(candidate, action)
        except Exception as exc:  # pragma: no cover - defensive safeguard
            raise HTTPException(
                status_code=400,
                detail=f"Replay failed while applying action {index}",
            ) from exc

    if candidate != current:
        raise HTTPException(status_code=409, detail="Replay diverged from recorded state")

    return serialize(current, store.seed, store)
