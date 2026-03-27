"""Stats API routes."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request

from _04_ui.core.config import STATS_FILE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["stats"])

# Max stats payload size (64KB)
_MAX_STATS_SIZE = 65536


@router.get("/stats")
def api_get_stats() -> dict:
    """Get player stats from server."""
    try:
        if STATS_FILE.exists():
            return json.loads(STATS_FILE.read_text())
        return {}
    except Exception:
        logger.exception("Failed to load stats")
        return {}


@router.post("/stats")
async def api_save_stats(request: Request) -> dict:
    """Save player stats to server."""
    try:
        body = await request.body()
        if len(body) > _MAX_STATS_SIZE:
            raise HTTPException(status_code=413, detail="Payload too large")
        stats = json.loads(body)
        if not isinstance(stats, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object")
        if len(stats) > 100:
            raise HTTPException(status_code=400, detail="Too many entries")
        for key, val in stats.items():
            if not isinstance(key, str) or len(key) > 64:
                raise HTTPException(status_code=400, detail="Invalid opponent key")
            if not isinstance(val, dict):
                raise HTTPException(status_code=400, detail="Invalid stat entry")
            if not {"name", "wins", "losses", "draws"} <= val.keys():
                raise HTTPException(status_code=400, detail="Missing stat fields")
            if not isinstance(val.get("name"), str) or len(val["name"]) > 128:
                raise HTTPException(status_code=400, detail="Invalid name")
            for field in ("wins", "losses", "draws"):
                if not isinstance(val.get(field), int) or val[field] < 0:
                    raise HTTPException(status_code=400, detail=f"Invalid {field}")
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(stats, indent=2))
        return {"status": "ok"}
    except (HTTPException,):
        raise
    except Exception:
        logger.exception("Failed to save stats")
        raise HTTPException(status_code=500, detail="Failed to save stats") from None


@router.delete("/stats")
def api_clear_stats() -> dict:
    """Clear all player stats."""
    try:
        if STATS_FILE.exists():
            STATS_FILE.unlink()
        return {"status": "ok"}
    except Exception:
        logger.exception("Failed to clear stats")
        raise HTTPException(status_code=500, detail="Failed to clear stats") from None
