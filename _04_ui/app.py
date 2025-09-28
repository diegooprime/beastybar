"""FastAPI application exposing the Beasty Bar simulator."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from _01_simulator import actions, simulate, state
from _02_agents import DiegoAgent, FirstLegalAgent, GreedyAgent, RandomAgent
from _02_agents.base import Agent


@dataclass
class TurnLogEntry:
    """Structured record of a single turn action."""

    id: str
    timestamp: datetime
    turn: int
    player: Optional[int]
    action: str
    effects: Tuple[str, ...]


@dataclass
class SessionStore:
    """In-memory holder for the current game state."""

    state: Optional[state.State] = None
    seed: Optional[int] = None
    human_player: int = 0
    agent_player: Optional[int] = None
    opponent: Optional[str] = None
    agent: Optional[Agent] = None
    log: List[TurnLogEntry] = field(default_factory=list)

    def require_state(self) -> state.State:
        if self.state is None:
            raise HTTPException(status_code=404, detail="No active game")
        return self.state


class NewGameRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    seed: Optional[int] = None
    starting_player: int = Field(default=0, ge=0, le=1, alias="startingPlayer")
    human_player: int = Field(default=0, ge=0, le=1, alias="humanPlayer")
    opponent: Optional[str] = None


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

    @app.get("/api/agents")
    def api_agents() -> Dict[str, List[str]]:
        return {"agents": _available_agents()}

    @app.post("/api/new-game")
    def api_new_game(request: NewGameRequest) -> dict:
        seed = request.seed if request.seed is not None else secrets.randbits(32)
        store.seed = seed
        store.human_player = request.human_player
        store.opponent = request.opponent
        store.agent_player = None
        if store.agent:
            store.agent = None

        store.state = simulate.new_game(seed, starting_player=request.starting_player)
        store.log.clear()
        _log_new_game(store, starting_player=request.starting_player)

        if request.opponent:
            agent = _agent_from_name(request.opponent)
            store.agent = agent
            store.agent_player = 1 - store.human_player
            agent_view = state.mask_state_for_player(store.require_state(), store.agent_player)
            agent.start_game(agent_view)
            _auto_play(store)

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
        if store.agent is not None and player != store.human_player:
            raise HTTPException(status_code=400, detail="It is not the human player's turn")
        legal = simulate.legal_actions(game_state, player)
        params_tuple = tuple(payload.params)
        try:
            action = next(
                act for act in legal if act.hand_index == payload.hand_index and act.params == params_tuple
            )
        except StopIteration as exc:
            raise HTTPException(status_code=400, detail="Illegal action") from exc

        _apply_action(store, action)

        if store.agent is not None:
            current = store.require_state()
            if simulate.is_terminal(current):
                if store.agent_player is None:
                    raise RuntimeError("Agent player index is not set")
                view = state.mask_state_for_player(current, store.agent_player)
                store.agent.end_game(view)
            else:
                _auto_play(store)

        return _serialize(store.require_state(), store.seed, store)

    return app


def _serialize(game_state: state.State, seed: Optional[int], store: SessionStore) -> dict:
    active_player = game_state.active_player
    legal = simulate.legal_actions(game_state, active_player) if not simulate.is_terminal(game_state) else ()
    log_entries = list(store.log)
    return {
        "seed": seed,
        "turn": game_state.turn,
        "activePlayer": active_player,
        "isTerminal": simulate.is_terminal(game_state),
        "score": simulate.score(game_state) if simulate.is_terminal(game_state) else None,
        "humanPlayer": store.human_player,
        "agentPlayer": store.agent_player,
        "opponent": store.opponent,
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
        "log": [_serialize_log_entry(entry) for entry in log_entries],
        "logText": _format_log_text(store, log_entries),
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


def _agent_from_name(name: str) -> Agent:
    try:
        factory = _AGENT_FACTORIES[name]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown agent '{name}'") from exc
    return factory()


def _available_agents() -> List[str]:
    return sorted(_AGENT_FACTORIES.keys())


def _auto_play(store: SessionStore) -> None:
    if store.agent is None or store.agent_player is None:
        return
    if store.state is None:
        return

    while not simulate.is_terminal(store.state) and store.state.active_player == store.agent_player:
        legal = simulate.legal_actions(store.state, store.agent_player)
        if not legal:
            next_player = store.state.next_player()
            store.state = state.set_active_player(store.state, next_player, advance_turn=True)
            _append_log_entry(
                store,
                TurnLogEntry(
                    id=_make_log_id(),
                    timestamp=datetime.now(timezone.utc),
                    turn=store.state.turn,
                    player=store.agent_player,
                    action="No legal actions available",
                    effects=("Turn passes to next player",),
                ),
            )
            continue
        if store.agent_player is None:
            raise RuntimeError("Agent player index is not set")
        view = state.mask_state_for_player(store.require_state(), store.agent_player)
        action = store.agent.select_action(view, legal)
        _apply_action(store, action)

    if simulate.is_terminal(store.state):
        if store.agent_player is None:
            raise RuntimeError("Agent player index is not set")
        terminal_view = state.mask_state_for_player(store.require_state(), store.agent_player)
        store.agent.end_game(terminal_view)


def _apply_action(store: SessionStore, action: actions.Action) -> None:
    before = store.require_state()
    player = before.active_player
    try:
        card = before.players[player].hand[action.hand_index]
    except IndexError as exc:  # pragma: no cover - validation upstream should prevent this
        raise HTTPException(status_code=400, detail="Invalid hand index") from exc

    after = simulate.apply(before, action)
    store.state = after
    effects = _describe_action_effects(store, before, after, player)
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
    )
    _append_log_entry(store, entry)


def _log_new_game(store: SessionStore, *, starting_player: int) -> None:
    opponent = store.opponent
    details: List[str] = []
    if store.seed is not None:
        details.append(f"Seed {store.seed}")
    details.append(f"Starting player P{starting_player}")
    if opponent:
        human = store.human_player
        agent = store.agent_player if store.agent_player is not None else (1 - human)
        details.append(f"You are P{human}")
        details.append(f"Opponent {opponent} as P{agent}")

    entry = TurnLogEntry(
        id=_make_log_id(),
        timestamp=datetime.now(timezone.utc),
        turn=0,
        player=None,
        action="New game started",
        effects=tuple(details),
    )
    _append_log_entry(store, entry)


def _append_log_entry(store: SessionStore, entry: TurnLogEntry, *, max_entries: int = 100) -> None:
    store.log.append(entry)
    if len(store.log) > max_entries:
        store.log[:] = store.log[-max_entries:]


def _make_log_id() -> str:
    return uuid.uuid4().hex


def _describe_action_effects(store: SessionStore, before: state.State, after: state.State, player: int) -> List[str]:
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

    draw = _drawn_cards(before, after, player)
    if draw:
        is_visible = player == store.human_player or store.opponent is None
        if is_visible:
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
    }


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
    if store.opponent:
        if player == store.human_player:
            return f"P{player} (You)"
        if store.agent_player is not None and player == store.agent_player:
            return f"P{player} ({store.opponent})"
    return f"P{player}"


_AGENT_FACTORIES: Dict[str, type[Agent]] = {
    "first": FirstLegalAgent,
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "diego": DiegoAgent,
}


__all__ = ["create_app"]
