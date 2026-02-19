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

# Expensive endpoints get stricter per-IP rate limits (requests per minute)
_EXPENSIVE_ENDPOINTS = {
    "/api/ai-battle/start": 10,
    "/api/benchmark": 1,
    "/api/claude-bridge/write-state": 5,
    "/api/claude-bridge/apply-move": 5,
}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    session_store = SessionStore()
    rate_limiter = RateLimiter()
    expensive_limiters = {
        path: RateLimiter(max_requests=limit, window_seconds=60)
        for path, limit in _EXPENSIVE_ENDPOINTS.items()
    }

    # Default session for backward compatibility (single-user local mode)
    default_session_id = "default"
    store = session_store.get_or_create(default_session_id)

    app = FastAPI(
        title="Beasty Bar Simulator",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Add CORS middleware — locked down for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://shiptoday101-beastybar.hf.space",
            "https://nuc.taildc84b3.ts.net",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Session-ID"],
    )

    # Security headers + rate limiting middleware
    @app.middleware("http")
    async def security_middleware(request: Request, call_next: Callable) -> Response:
        # Extract real client IP behind proxy (Tailscale Funnel, HF Spaces)
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )

        path = request.url.path

        # Skip rate limiting for static files (images, CSS, JS, card art)
        is_static = path.startswith("/static") or path == "/" or path == ""

        if not is_static:
            # Per-endpoint rate limiting for expensive operations
            if path in expensive_limiters and not expensive_limiters[path].is_allowed(client_ip):
                return Response(
                    content='{"detail": "Rate limit exceeded for this endpoint"}',
                    status_code=429,
                    media_type="application/json",
                )

            # Global rate limit (API endpoints only)
            if not rate_limiter.is_allowed(client_ip):
                return Response(
                    content='{"detail": "Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json",
                )

        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self' wss: ws:; "
            "frame-ancestors 'none'"
        )
        # Remove server header
        if "server" in response.headers:
            del response.headers["server"]

        return response

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
