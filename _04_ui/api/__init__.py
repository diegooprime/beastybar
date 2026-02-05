"""API routers for the Beasty Bar web UI."""

from _04_ui.api.actions import router as actions_router
from _04_ui.api.agents import router as agents_router
from _04_ui.api.claude import router as claude_router
from _04_ui.api.game import router as game_router
from _04_ui.api.stats import router as stats_router
from _04_ui.api.visualization import router as visualization_router

__all__ = [
    "actions_router",
    "agents_router",
    "claude_router",
    "game_router",
    "stats_router",
    "visualization_router",
]
