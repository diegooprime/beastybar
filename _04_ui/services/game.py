"""Game logic and session management for the Beasty Bar web UI."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException, Request, Response

from _01_simulator import actions, engine, simulate
from _04_ui.core.session import GameSession, TurnLogEntry
from _04_ui.services.serializer import (
    action_label,
    describe_action_effects,
    turn_flow_summary,
)


def get_session_id(request: Request) -> str:
    """Extract or generate session ID from request."""
    # Try to get from cookie first
    session_id = request.cookies.get("session_id")
    if session_id:
        return session_id
    # Fall back to header
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id
    # Generate new one (will be set in response)
    return uuid.uuid4().hex


def set_session_cookie(response: Response, session_id: str) -> None:
    """Set session cookie on response."""
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,  # 7 days
    )


def apply_action(
    store: GameSession,
    action: actions.Action,
    *,
    record_history: bool = True,
) -> None:
    """Apply an action to the game state and update the log."""
    before = store.require_state()
    player = before.active_player
    try:
        card = before.players[player].hand[action.hand_index]
    except IndexError as exc:  # pragma: no cover - validation upstream should prevent this
        raise HTTPException(status_code=400, detail="Invalid hand index") from exc

    after, steps = engine.step_with_trace(before, action)
    store.state = after
    if record_history:
        store.history.append(action)

    effects = turn_flow_summary(steps)
    effects.extend(describe_action_effects(store, before, after, player, include_draw=False))
    if simulate.is_terminal(after):
        final_score = simulate.score(after)
        summary = "Game over: " + " - ".join(f"P{idx} {score}" for idx, score in enumerate(final_score))
        effects.append(summary)
    entry = TurnLogEntry(
        id=make_log_id(),
        timestamp=datetime.now(timezone.utc),
        turn=after.turn,
        player=player,
        action=action_label(card, action),
        effects=tuple(effects),
        steps=steps,
    )
    append_log_entry(store, entry)


def log_new_game(store: GameSession, *, starting_player: int) -> None:
    """Log the start of a new game."""
    details: list[str] = []
    if store.seed is not None:
        details.append(f"Seed {store.seed}")
    details.append(f"Starting player P{starting_player}")
    details.append(f"You are P{store.human_player}")

    entry = TurnLogEntry(
        id=make_log_id(),
        timestamp=datetime.now(timezone.utc),
        turn=0,
        player=None,
        action="New game started",
        effects=tuple(details),
        steps=(),
    )
    append_log_entry(store, entry)


def append_log_entry(store: GameSession, entry: TurnLogEntry, *, max_entries: int = 100) -> None:
    """Append a log entry, maintaining the max size."""
    store.log.append(entry)
    if len(store.log) > max_entries:
        store.log[:] = store.log[-max_entries:]


def make_log_id() -> str:
    """Generate a unique log entry ID."""
    return uuid.uuid4().hex
