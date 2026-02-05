"""Service layer for the Beasty Bar web UI."""

from _04_ui.services.ai import (
    AI_AGENTS,
    get_claude_move_func,
    get_visualizing_agent,
)
from _04_ui.services.game import (
    apply_action,
    get_session_id,
    log_new_game,
    set_session_cookie,
)
from _04_ui.services.serializer import (
    action_label,
    card_view,
    count_score,
    format_hand_for_claude,
    format_legal_actions_for_claude,
    format_queue_for_claude,
    serialize,
    serialize_action,
    serialize_battle_state,
)

__all__ = [
    "AI_AGENTS",
    "action_label",
    "apply_action",
    "card_view",
    "count_score",
    "format_hand_for_claude",
    "format_legal_actions_for_claude",
    "format_queue_for_claude",
    "get_claude_move_func",
    "get_session_id",
    "get_visualizing_agent",
    "log_new_game",
    "serialize",
    "serialize_action",
    "serialize_battle_state",
    "set_session_cookie",
]
