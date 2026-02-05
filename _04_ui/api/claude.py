"""Claude Code integration API routes."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

from _01_simulator import simulate
from _04_ui.core.config import CLAUDE_BRIDGE_DIR, CLAUDE_MOVE_FILE, CLAUDE_STATE_FILE
from _04_ui.core.session import GameSession  # noqa: TC001
from _04_ui.services.game import apply_action
from _04_ui.services.serializer import (
    action_label,
    count_score,
    format_hand_for_claude,
    format_legal_actions_for_claude,
    format_queue_for_claude,
    serialize,
    serialize_action,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["claude"])

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


@router.get("/claude-state")
def api_claude_state() -> dict:
    """Get game state formatted for Claude Code to read and make a move."""
    store = get_store()
    game_state = store.require_state()
    ai_player = 1 - store.human_player
    legal = simulate.legal_actions(game_state, ai_player) if not simulate.is_terminal(game_state) else ()

    # Format state for Claude
    queue_str = format_queue_for_claude(game_state.zones.queue)
    hand_str = format_hand_for_claude(game_state.players[ai_player].hand, legal)
    legal_str = format_legal_actions_for_claude(game_state, legal)

    formatted = f"""## Beasty Bar - Claude's Turn (Player {ai_player})

**Queue** (left=Heaven's Gate, right=That's It):
{queue_str}

**Your Hand:**
{hand_str}

**Legal Actions:**
{legal_str}

**Scores:** You {count_score(game_state.zones.beasty_bar, ai_player)} - Opponent {count_score(game_state.zones.beasty_bar, store.human_player)}

Reply with the action number (e.g., "1" or "3")."""

    return {
        "formatted": formatted,
        "legalActions": [
            {"index": i + 1, **serialize_action(game_state, action)}
            for i, action in enumerate(legal)
        ],
        "isClaudeTurn": game_state.active_player == ai_player and not simulate.is_terminal(game_state),
    }


@router.post("/claude-move")
def api_claude_move(payload: dict) -> dict:
    """Apply Claude Code's move (entered by user)."""
    store = get_store()
    game_state = store.require_state()
    ai_player = 1 - store.human_player

    if simulate.is_terminal(game_state):
        raise HTTPException(status_code=400, detail="Game is already over")

    if game_state.active_player != ai_player:
        raise HTTPException(status_code=400, detail="It's not Claude's turn")

    action_index = payload.get("actionIndex")
    if action_index is None:
        raise HTTPException(status_code=400, detail="Missing actionIndex")

    legal = simulate.legal_actions(game_state, ai_player)
    if action_index < 1 or action_index > len(legal):
        raise HTTPException(status_code=400, detail=f"Invalid action index. Choose 1-{len(legal)}")

    action = legal[action_index - 1]
    apply_action(store, action)

    return serialize(store.require_state(), store.seed, store)


@router.post("/claude-bridge/write-state")
def api_claude_write_state() -> dict:
    """Write game state to file for Claude Code to read."""
    store = get_store()
    game_state = store.require_state()
    ai_player = 1 - store.human_player

    if game_state.active_player != ai_player:
        return {"written": False, "reason": "Not Claude's turn"}

    if simulate.is_terminal(game_state):
        return {"written": False, "reason": "Game is over"}

    legal = simulate.legal_actions(game_state, ai_player)

    # Ensure bridge directory exists
    CLAUDE_BRIDGE_DIR.mkdir(exist_ok=True)

    # Clear any old move file
    if CLAUDE_MOVE_FILE.exists():
        CLAUDE_MOVE_FILE.unlink()

    # Format state for Claude
    queue_str = format_queue_for_claude(game_state.zones.queue)
    hand_str = format_hand_for_claude(game_state.players[ai_player].hand, legal)
    legal_str = format_legal_actions_for_claude(game_state, legal)

    state_data = {
        "turn": game_state.turn,
        "player": ai_player,
        "formatted": f"""## Beasty Bar - Claude's Turn (Player {ai_player})

**Queue** (left=Heaven's Gate, right=That's It):
{queue_str}

**Your Hand:**
{hand_str}

**Legal Actions:**
{legal_str}

**Scores:** You {count_score(game_state.zones.beasty_bar, ai_player)} - Opponent {count_score(game_state.zones.beasty_bar, store.human_player)}

Reply with JUST the action number (e.g., "1" or "3").""",
        "legalActions": [
            {"index": i + 1, "label": action_label(game_state.players[ai_player].hand[a.hand_index], a)}
            for i, a in enumerate(legal)
        ],
    }

    CLAUDE_STATE_FILE.write_text(json.dumps(state_data, indent=2))
    return {"written": True, "path": str(CLAUDE_STATE_FILE)}


@router.get("/claude-bridge/check-move")
def api_claude_check_move() -> dict:
    """Check if Claude Code has written a move response."""
    if not CLAUDE_MOVE_FILE.exists():
        return {"hasMove": False}

    try:
        move_data = json.loads(CLAUDE_MOVE_FILE.read_text())
        return {"hasMove": True, "actionIndex": move_data.get("actionIndex")}
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in move file: {e}")
        return {"hasMove": False, "error": "Invalid JSON in move file"}
    except FileNotFoundError:
        # File was deleted between exists() check and read
        return {"hasMove": False}
    except OSError as e:
        logger.warning(f"Error reading move file: {e}")
        return {"hasMove": False, "error": "Error reading move file"}


@router.post("/claude-bridge/apply-move")
def api_claude_apply_move() -> dict:
    """Apply the move from Claude's response file."""
    store = get_store()

    if not CLAUDE_MOVE_FILE.exists():
        raise HTTPException(status_code=400, detail="No move file found")

    try:
        move_data = json.loads(CLAUDE_MOVE_FILE.read_text())
        action_index = move_data.get("actionIndex")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid move file") from exc

    # Clean up bridge files
    if CLAUDE_STATE_FILE.exists():
        CLAUDE_STATE_FILE.unlink()
    if CLAUDE_MOVE_FILE.exists():
        CLAUDE_MOVE_FILE.unlink()

    # Apply the move
    game_state = store.require_state()
    ai_player = 1 - store.human_player
    legal = simulate.legal_actions(game_state, ai_player)

    if action_index < 1 or action_index > len(legal):
        raise HTTPException(status_code=400, detail=f"Invalid action index: {action_index}")

    action = legal[action_index - 1]
    apply_action(store, action)

    return serialize(store.require_state(), store.seed, store)
