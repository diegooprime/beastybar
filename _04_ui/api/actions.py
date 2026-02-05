"""Action-related API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from _01_simulator import simulate, state
from _04_ui.core.session import GameSession  # noqa: TC001
from _04_ui.models.requests import ActionPayload  # noqa: TC001
from _04_ui.services.ai import AI_AGENTS, get_claude_move_func, get_visualizing_agent
from _04_ui.services.game import apply_action
from _04_ui.services.serializer import serialize

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["actions"])

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


@router.post("/action")
def api_action(payload: ActionPayload) -> dict:
    """Apply a human player action."""
    store = get_store()
    game_state = store.require_state()
    player = game_state.active_player
    legal = simulate.legal_actions(game_state, player)
    params_tuple = tuple(payload.params)
    try:
        action = next(act for act in legal if act.hand_index == payload.hand_index and act.params == params_tuple)
    except StopIteration as exc:
        raise HTTPException(status_code=400, detail="Illegal action") from exc

    apply_action(store, action)

    return serialize(store.require_state(), store.seed, store)


@router.post("/ai-move")
async def api_ai_move() -> dict:
    """Have the AI opponent make a move."""
    store = get_store()
    game_state = store.require_state()

    if simulate.is_terminal(game_state):
        raise HTTPException(status_code=400, detail="Game is already over")

    player = game_state.active_player
    if player == store.human_player:
        raise HTTPException(status_code=400, detail="It's the human player's turn")

    # Get legal actions
    legal = simulate.legal_actions(game_state, player)

    # Get AI agent or use Claude API
    ai_name = store.ai_opponent or "heuristic"
    if ai_name == "claude":
        # Use Anthropic API
        try:
            get_claude_move = get_claude_move_func()
            action = get_claude_move(game_state, player, legal)
        except Exception as e:
            logger.exception("Claude API error")
            raise HTTPException(
                status_code=500,
                detail="AI service temporarily unavailable. Please try again.",
            ) from e
    else:
        if ai_name not in AI_AGENTS:
            logger.warning("Unknown AI agent '%s', falling back to heuristic", ai_name)
            ai_name = "heuristic"
        agent = AI_AGENTS[ai_name]
        masked_state = state.mask_state_for_player(game_state, player)

        # Try to use visualizing wrapper for neural agents
        viz_agent = get_visualizing_agent(agent, ai_name)
        if viz_agent is not None:
            # Build game context for visualization
            game_context = {
                "queue_cards": [c.species for c in game_state.zones.queue],
                "hand_cards": [c.species for c in game_state.players[player].hand],
                "bar_cards": [c.species for c in game_state.zones.beasty_bar],
                "scores": [
                    sum(c.points for c in game_state.zones.beasty_bar if c.owner == p)
                    for p in range(2)
                ],
            }
            action, _ = await viz_agent.select_action_with_viz(
                masked_state, legal, game_state.turn, player, game_context
            )
        else:
            action = agent(masked_state, legal)

    # Apply the action
    apply_action(store, action)

    return serialize(store.require_state(), store.seed, store)
