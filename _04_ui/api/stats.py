"""Stats API routes."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

from _04_ui.core.config import STATS_FILE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats")
def api_get_stats() -> dict:
    """Get player stats from server."""
    try:
        if STATS_FILE.exists():
            return json.loads(STATS_FILE.read_text())
        return {}
    except Exception as e:
        logger.warning(f"Failed to load stats: {e}")
        return {}


@router.post("/stats")
def api_save_stats(stats: dict) -> dict:
    """Save player stats to server."""
    try:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(stats, indent=2))
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to save stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to save stats") from e


@router.delete("/stats")
def api_clear_stats() -> dict:
    """Clear all player stats."""
    try:
        if STATS_FILE.exists():
            STATS_FILE.unlink()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to clear stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear stats") from e
