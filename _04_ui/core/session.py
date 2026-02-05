"""Session management for the Beasty Bar web UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime  # noqa: TC003
from threading import Lock

from fastapi import HTTPException

from _01_simulator import actions, engine, state  # noqa: TC001


@dataclass
class TurnLogEntry:
    """Structured record of a single turn action."""

    id: str
    timestamp: datetime
    turn: int
    player: int | None
    action: str
    effects: tuple[str, ...]
    steps: tuple[engine.TurnStep, ...] = field(default_factory=tuple)


@dataclass
class GameSession:
    """In-memory holder for a single game's state."""

    state: state.State | None = None
    seed: int | None = None
    human_player: int = 0
    log: list[TurnLogEntry] = field(default_factory=list)
    history: list[actions.Action] = field(default_factory=list)
    starting_player: int = 0
    ai_opponent: str | None = "heuristic"  # Default AI opponent

    def require_state(self) -> state.State:
        if self.state is None:
            raise HTTPException(status_code=404, detail="No active game")
        return self.state


class SessionStore:
    """Thread-safe session store with per-session isolation."""

    def __init__(self, max_sessions: int = 1000):
        self._sessions: dict[str, GameSession] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> GameSession:
        """Get existing session or create new one."""
        with self._lock:
            if session_id not in self._sessions:
                # Enforce max sessions limit
                if len(self._sessions) >= self._max_sessions:
                    # Remove oldest session (simple LRU approximation)
                    oldest_key = next(iter(self._sessions))
                    del self._sessions[oldest_key]
                self._sessions[session_id] = GameSession()
            return self._sessions[session_id]

    def get(self, session_id: str) -> GameSession | None:
        """Get session if it exists."""
        with self._lock:
            return self._sessions.get(session_id)
