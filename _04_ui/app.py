"""FastAPI application exposing the Beasty Bar simulator."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from _04_ui.api import (
    actions_router,
    agents_router,
    claude_router,
    game_router,
    stats_router,
    visualization_router,
)
from _04_ui.api.actions import set_store as set_actions_store
from _04_ui.api.claude import set_store as set_claude_store
from _04_ui.api.game import set_store as set_game_store
from _04_ui.api.visualization import set_static_dir
from _04_ui.api.visualization import set_store as set_viz_store
from _04_ui.core import RateLimiter, SessionStore


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    session_store = SessionStore()
    rate_limiter = RateLimiter()

    # Default session for backward compatibility (single-user local mode)
    default_session_id = "default"
    store = session_store.get_or_create(default_session_id)

    app = FastAPI(title="Beasty Bar Simulator", version="0.1.0")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )
        return await call_next(request)

    # Static files
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Set store references for routers that need them
    set_game_store(store)
    set_actions_store(store)
    set_claude_store(store)
    set_viz_store(store)
    set_static_dir(static_dir)

    # Include routers
    app.include_router(game_router)
    app.include_router(actions_router)
    app.include_router(agents_router)
    app.include_router(stats_router)
    app.include_router(claude_router)
    app.include_router(visualization_router)

    # Root route
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


__all__ = ["create_app"]

# Create app instance for uvicorn
app = create_app()
