"""Pydantic models for the Beasty Bar web UI."""

from _04_ui.models.requests import (
    ActionPayload,
    AIBattleRequest,
    NewGameRequest,
)

__all__ = [
    "AIBattleRequest",
    "ActionPayload",
    "NewGameRequest",
]
