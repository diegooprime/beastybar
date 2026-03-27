"""Pydantic request models for the Beasty Bar API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Allowlist of valid AI opponents (excludes "claude" to prevent API credit burn)
VALID_AI_OPPONENTS = frozenset({
    "random", "heuristic", "aggressive", "defensive",
    "queue_control", "skunk_specialist", "noisy", "online",
    "neural", "ppo_iter949", "ppo_iter600", "ppo_iter949_tablebase",
})


class NewGameRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    seed: int | None = None
    starting_player: int = Field(default=0, ge=0, le=1, alias="startingPlayer")
    human_player: int = Field(default=0, ge=0, le=1, alias="humanPlayer")
    ai_opponent: str | None = Field(default="heuristic", alias="aiOpponent")

    @field_validator("ai_opponent")
    @classmethod
    def check_ai_opponent(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_AI_OPPONENTS:
            raise ValueError(f"Unknown AI opponent: {v}")
        return v


class ActionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    hand_index: int = Field(alias="handIndex", ge=0, le=11)
    params: list[int] = Field(default_factory=list, max_length=5)


class AIBattleRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    player1_agent: str = Field(alias="player1Agent")
    player2_agent: str = Field(alias="player2Agent")
    num_games: int = Field(default=10, ge=1, le=20, alias="numGames")
