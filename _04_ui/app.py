"""FastAPI application exposing the Beasty Bar simulator."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from _01_simulator import actions, engine, simulate, state


@dataclass
class TurnLogEntry:
    """Structured record of a single turn action."""

    id: str
    timestamp: datetime
    turn: int
    player: Optional[int]
    action: str
    effects: Tuple[str, ...]
    steps: Tuple[engine.TurnStep, ...] = field(default_factory=tuple)


@dataclass
class SessionStore:
    """In-memory holder for the current game state."""

    state: Optional[state.State] = None
    seed: Optional[int] = None
    human_player: int = 0
    log: List[TurnLogEntry] = field(default_factory=list)
    history: List[actions.Action] = field(default_factory=list)
    starting_player: int = 0

    def require_state(self) -> state.State:
        if self.state is None:
            raise HTTPException(status_code=404, detail="No active game")
        return self.state


class NewGameRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    seed: Optional[int] = None
    starting_player: int = Field(default=0, ge=0, le=1, alias="startingPlayer")
    human_player: int = Field(default=0, ge=0, le=1, alias="humanPlayer")


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
        store.human_player = request.human_player
        store.starting_player = request.starting_player
        store.state = simulate.new_game(seed, starting_player=request.starting_player)
        store.log.clear()
        store.history.clear()
        _log_new_game(store, starting_player=request.starting_player)

        return _serialize(store.require_state(), store.seed, store)

    @app.get("/api/state")
    def api_state() -> dict:
        return _serialize(store.require_state(), store.seed, store)

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

        _apply_action(store, action)

        return _serialize(store.require_state(), store.seed, store)

    @app.post("/api/replay")
    def api_replay() -> dict:
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

        return _serialize(current, store.seed, store)

    return app


def _serialize(game_state: state.State, seed: Optional[int], store: SessionStore) -> dict:
    active_player = game_state.active_player
    legal = simulate.legal_actions(game_state, active_player) if not simulate.is_terminal(game_state) else ()
    log_entries = list(store.log)
    visible_state = game_state
    return {
        "seed": seed,
        "turn": game_state.turn,
        "activePlayer": active_player,
        "isTerminal": simulate.is_terminal(game_state),
        "score": simulate.score(game_state) if simulate.is_terminal(game_state) else None,
        "humanPlayer": store.human_player,
        "queue": [_card_view(card) for card in visible_state.zones.queue],
        "zones": {
            "beastyBar": [_card_view(card) for card in visible_state.zones.beasty_bar],
            "thatsIt": [_card_view(card) for card in visible_state.zones.thats_it],
        },
        "hands": [
            [_card_view(card) for card in player_state.hand]
            for player_state in visible_state.players
        ],
        "legalActions": [_serialize_action(game_state, action) for action in legal],
        "log": [_serialize_log_entry(entry) for entry in log_entries],
        "logText": _format_log_text(store, log_entries),
        "turnFlow": _serialize_turn_steps(log_entries[-1].steps) if log_entries else [],
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


def _apply_action(
    store: SessionStore,
    action: actions.Action,
    *,
    record_history: bool = True,
) -> None:
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

    effects = _turn_flow_summary(steps)
    effects.extend(_describe_action_effects(store, before, after, player, include_draw=False))
    if simulate.is_terminal(after):
        final_score = simulate.score(after)
        summary = "Game over: " + " – ".join(
            f"P{idx} {score}" for idx, score in enumerate(final_score)
        )
        effects.append(summary)
    entry = TurnLogEntry(
        id=_make_log_id(),
        timestamp=datetime.now(timezone.utc),
        turn=after.turn,
        player=player,
        action=_action_label(card, action),
        effects=tuple(effects),
        steps=steps,
    )
    _append_log_entry(store, entry)

def _log_new_game(store: SessionStore, *, starting_player: int) -> None:
    details: List[str] = []
    if store.seed is not None:
        details.append(f"Seed {store.seed}")
    details.append(f"Starting player P{starting_player}")
    details.append(f"You are P{store.human_player}")

    entry = TurnLogEntry(
        id=_make_log_id(),
        timestamp=datetime.now(timezone.utc),
        turn=0,
        player=None,
        action="New game started",
        effects=tuple(details),
        steps=(),
    )
    _append_log_entry(store, entry)


def _append_log_entry(store: SessionStore, entry: TurnLogEntry, *, max_entries: int = 100) -> None:
    store.log.append(entry)
    if len(store.log) > max_entries:
        store.log[:] = store.log[-max_entries:]


def _make_log_id() -> str:
    return uuid.uuid4().hex


def _turn_flow_summary(steps: Tuple[engine.TurnStep, ...]) -> List[str]:
    lines: List[str] = []
    for idx, step in enumerate(steps, start=1):
        title = step.name.title()
        if step.events:
            lines.append(f"Step {idx} – {title}: {step.events[0]}")
            for extra in step.events[1:]:
                lines.append(f"  ↳ {extra}")
        else:
            lines.append(f"Step {idx} – {title}: No effect.")
    return lines


def _describe_action_effects(
    store: SessionStore,
    before: state.State,
    after: state.State,
    player: int,
    *,
    include_draw: bool = True,
) -> List[str]:
    effects: List[str] = []

    scored = _zone_new_cards(before.zones.beasty_bar, after.zones.beasty_bar)
    if scored:
        owners = defaultdict(int)
        for card in scored:
            owners[card.owner] += card.points
        gains = ", ".join(f"P{owner} +{points}" for owner, points in sorted(owners.items()))
        effects.append(f"Heaven's Gate: {_format_card_list(scored)} ({gains})")

    bounced = _zone_new_cards(before.zones.thats_it, after.zones.thats_it)
    if bounced:
        effects.append(f"Sent to THAT'S IT: {_format_card_list(bounced)}")

    if include_draw:
        draw = _drawn_cards(before, after, player)
        if draw:
            if player == store.human_player:
                effects.append(f"Drew {_format_card_list(draw)}")
            else:
                count = len(draw)
                label = "card" if count == 1 else "cards"
                effects.append(f"Drew {count} {label}")

    return effects


def _drawn_cards(before: state.State, after: state.State, player: int) -> List[state.Card]:
    before_ids = {id(card) for card in before.players[player].hand}
    return [card for card in after.players[player].hand if id(card) not in before_ids]


def _zone_new_cards(before_cards: Tuple[state.Card, ...], after_cards: Tuple[state.Card, ...]) -> List[state.Card]:
    before_ids = {id(card) for card in before_cards}
    return [card for card in after_cards if id(card) not in before_ids]


def _format_card_list(cards: List[state.Card]) -> str:
    return ", ".join(f"{card.species} (P{card.owner})" for card in cards)


def _serialize_log_entry(entry: TurnLogEntry) -> dict:
    timestamp = entry.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    iso_ts = timestamp.isoformat(timespec="milliseconds")
    epoch_ms = int(timestamp.timestamp() * 1000)
    return {
        "id": entry.id,
        "turn": entry.turn,
        "player": entry.player,
        "action": entry.action,
        "effects": list(entry.effects),
        "timestamp": iso_ts,
        "timestampMs": epoch_ms,
        "steps": _serialize_turn_steps(entry.steps),
    }


def _serialize_turn_steps(steps: Tuple[engine.TurnStep, ...]) -> List[dict]:
    return [
        {
            "name": step.name,
            "events": list(step.events),
        }
        for step in steps
    ]


def _format_log_text(store: SessionStore, entries: List[TurnLogEntry]) -> str:
    if not entries:
        return ""

    lines: List[str] = []
    for entry in entries:
        if entry.player is None:
            header = f"Turn {entry.turn} – Setup: {entry.action}"
        else:
            label = _log_player_label(store, entry.player)
            header = f"Turn {entry.turn} – {label}: {entry.action}"
        lines.append(header)
        for effect in entry.effects:
            lines.append(f"  - {effect}")
    return "\n".join(lines)


def _log_player_label(store: SessionStore, player: int) -> str:
    if player == store.human_player:
        return f"P{player} (You)"
    return f"P{player}"


__all__ = ["create_app"]
