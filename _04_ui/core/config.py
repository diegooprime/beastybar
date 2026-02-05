"""Configuration constants and paths for the Beasty Bar web UI."""

from __future__ import annotations

from pathlib import Path

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 60  # requests per window
RATE_LIMIT_WINDOW = 60  # window in seconds

# Claude Code special opponent identifier
CLAUDE_CODE_OPPONENT = "claude"

# File paths for Claude Code communication bridge
CLAUDE_BRIDGE_DIR = Path(__file__).resolve().parent.parent / "claude_bridge"
CLAUDE_STATE_FILE = CLAUDE_BRIDGE_DIR / "state.json"
CLAUDE_MOVE_FILE = CLAUDE_BRIDGE_DIR / "move.json"

# Stats file path
STATS_FILE = Path(__file__).resolve().parent.parent / "data" / "player_stats.json"
