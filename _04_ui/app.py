"""FastAPI application exposing the Beasty Bar simulator."""

from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from collections import defaultdict
from collections.abc import Callable  # noqa: TC003 - used at runtime for middleware
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

# Visualization imports (lazy to avoid circular imports)
_viz_manager = None
_viz_capture_agents: dict[str, "VisualizingNeuralAgent"] = {}  # type: ignore[name-defined]


def _get_viz_manager():
    """Lazy import of visualization WebSocket manager."""
    global _viz_manager
    if _viz_manager is None:
        from _04_ui.visualization.websocket_manager import visualizer_ws_manager
        _viz_manager = visualizer_ws_manager
    return _viz_manager


def _get_visualizing_agent(agent, agent_name: str):
    """Get or create a VisualizingNeuralAgent wrapper for an agent."""
    global _viz_capture_agents
    if agent_name not in _viz_capture_agents:
        try:
            from _02_agents.neural import NeuralAgent
            if isinstance(agent, NeuralAgent):
                from _04_ui.visualization.activation_capture import VisualizingNeuralAgent
                _viz_capture_agents[agent_name] = VisualizingNeuralAgent(
                    agent, _get_viz_manager()
                )
        except ImportError:
            pass
    return _viz_capture_agents.get(agent_name)

from _01_simulator import actions, engine, simulate, state
from _02_agents import HeuristicAgent, HeuristicConfig, RandomAgent
from _02_agents.heuristic import OnlineStrategies

# Import neural MCTS agent (requires network initialization)
try:
    from _02_agents.mcts import MCTSAgent
except ImportError:
    MCTSAgent = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
def _get_claude_move_func():
    """Lazy import of claude agent to avoid circular imports."""
    from _04_ui.claude_agent import get_claude_move
    return get_claude_move

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 60  # requests per window
RATE_LIMIT_WINDOW = 60  # window in seconds


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW):
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        with self._lock:
            # Clean old entries
            self._requests[client_id] = [
                t for t in self._requests[client_id] if now - t < self._window
            ]
            if len(self._requests[client_id]) >= self._max_requests:
                return False
            self._requests[client_id].append(now)
            return True

# AI opponent instances
AI_AGENTS = {
    "random": RandomAgent(seed=None),
    "heuristic": HeuristicAgent(seed=None),
    "aggressive": HeuristicAgent(config=HeuristicConfig(bar_weight=3.0, aggression=0.8)),
    "defensive": HeuristicAgent(config=HeuristicConfig(bar_weight=1.0, aggression=0.2)),
    "queue_control": HeuristicAgent(config=HeuristicConfig(queue_front_weight=2.0)),
    "skunk_specialist": HeuristicAgent(config=HeuristicConfig(species_weights={"skunk": 2.0})),
    "noisy": HeuristicAgent(config=HeuristicConfig(noise_epsilon=0.15)),
    "online": OnlineStrategies(),
}

# Add Neural agent if checkpoint exists
def _load_neural_agent(ckpt_path: str | Path | None = None) -> tuple:
    """Try to load neural agent from checkpoint.

    Returns (agent, name, iteration) or (None, None, None).
    """
    try:
        import torch
        import os
        from _02_agents.neural import NeuralAgent
        from _02_agents.neural.network import BeastyBarNetwork
        from _02_agents.neural.utils import NetworkConfig

        def load_from_checkpoint(path):
            """Load network from PPO or MCTS checkpoint."""
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            # Extract network config (nested in training config)
            config_dict = checkpoint.get("config", {})
            if "network_config" in config_dict:
                net_config = NetworkConfig.from_dict(config_dict["network_config"])
            else:
                net_config = NetworkConfig.from_dict(config_dict)

            # Create and load network
            network = BeastyBarNetwork(net_config)
            network.load_state_dict(checkpoint["model_state_dict"])
            network.eval()
            return network, checkpoint.get("iteration", 0)

        # Check for checkpoint path in environment or explicit path
        checkpoint_path = ckpt_path or os.environ.get("NEURAL_CHECKPOINT", None)
        if checkpoint_path and Path(checkpoint_path).exists():
            network, iteration = load_from_checkpoint(checkpoint_path)
            logger.info(f"Loaded neural agent from {checkpoint_path} (iter {iteration})")
            return NeuralAgent(network, mode="greedy"), f"ppo_iter{iteration}", iteration

        # Try to find latest checkpoint - prioritize v4 (best model)
        checkpoint_dirs = [
            Path("checkpoints/v4"),
            Path("checkpoints/v3"),
            Path("checkpoints/v2"),
            Path("checkpoints/v1"),
        ]
        for ckpt_dir in checkpoint_dirs:
            if ckpt_dir.exists():
                # Check for final.pt first, then iter_*.pt
                final_pt = ckpt_dir / "final.pt"
                if final_pt.exists():
                    network, iteration = load_from_checkpoint(final_pt)
                    logger.info(f"Loaded neural agent from {final_pt} (iter {iteration})")
                    return NeuralAgent(network, mode="greedy"), f"ppo_iter{iteration}", iteration
                checkpoints = sorted(ckpt_dir.glob("iter_*.pt"))
                if checkpoints:
                    network, iteration = load_from_checkpoint(checkpoints[-1])
                    logger.info(f"Loaded neural agent from {checkpoints[-1]} (iter {iteration})")
                    return NeuralAgent(network, mode="greedy"), f"ppo_iter{iteration}", iteration
        return None, None, None
    except Exception as e:
        logger.warning(f"Failed to load neural agent: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

_neural_agent, _neural_name, _neural_iter = _load_neural_agent()
if _neural_agent is not None:
    AI_AGENTS[_neural_name] = _neural_agent
    # Keep "neural" as alias for backwards compatibility
    AI_AGENTS["neural"] = _neural_agent

# Load additional neural agents
_neural_agents_extra: list[tuple] = []
_extra_checkpoints = [
    ("checkpoints/v4/iter_949.pt", 949),
    ("checkpoints/v4/iter_600_final.pt", 600),
]
for ckpt_path, expected_iter in _extra_checkpoints:
    if Path(ckpt_path).exists():
        _agent, _name, _iter = _load_neural_agent(ckpt_path)
        if _agent is not None and _name not in AI_AGENTS:
            AI_AGENTS[_name] = _agent
            _neural_agents_extra.append((_agent, _name, _iter))

# Add MCTS agent if available
# NOTE: MCTSAgent signature changed - temporarily disabled
# if MCTSAgent is not None:
#     AI_AGENTS["mcts"] = MCTSAgent(iterations=500, determinizations=8, seed=None)

# Claude Code is a special "opponent" - moves are entered manually by the user
CLAUDE_CODE_OPPONENT = "claude"

# File paths for Claude Code communication bridge
CLAUDE_BRIDGE_DIR = Path(__file__).resolve().parent / "claude_bridge"
CLAUDE_STATE_FILE = CLAUDE_BRIDGE_DIR / "state.json"
CLAUDE_MOVE_FILE = CLAUDE_BRIDGE_DIR / "move.json"


@dataclass
class TurnLogEntry:
    """Structured record of a single turn action."""

    id: str
    timestamp: datetime
    turn: int
    player: int | None
    action: str
    effects: tuple[str, ...]
    steps: tuple[engine.TurnStep, ...] = field(default_factory=tuple)


@dataclass
class GameSession:
    """In-memory holder for a single game's state."""

    state: state.State | None = None
    seed: int | None = None
    human_player: int = 0
    log: list[TurnLogEntry] = field(default_factory=list)
    history: list[actions.Action] = field(default_factory=list)
    starting_player: int = 0
    ai_opponent: str | None = "heuristic"  # Default AI opponent

    def require_state(self) -> state.State:
        if self.state is None:
            raise HTTPException(status_code=404, detail="No active game")
        return self.state


class SessionStore:
    """Thread-safe session store with per-session isolation."""

    def __init__(self, max_sessions: int = 1000):
        self._sessions: dict[str, GameSession] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> GameSession:
        """Get existing session or create new one."""
        with self._lock:
            if session_id not in self._sessions:
                # Enforce max sessions limit
                if len(self._sessions) >= self._max_sessions:
                    # Remove oldest session (simple LRU approximation)
                    oldest_key = next(iter(self._sessions))
                    del self._sessions[oldest_key]
                self._sessions[session_id] = GameSession()
            return self._sessions[session_id]

    def get(self, session_id: str) -> GameSession | None:
        """Get session if it exists."""
        with self._lock:
            return self._sessions.get(session_id)


class NewGameRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    seed: int | None = None
    starting_player: int = Field(default=0, ge=0, le=1, alias="startingPlayer")
    human_player: int = Field(default=0, ge=0, le=1, alias="humanPlayer")
    ai_opponent: str | None = Field(default="heuristic", alias="aiOpponent")


class ActionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    hand_index: int = Field(alias="handIndex")
    params: list[int] = Field(default_factory=list)


class AIBattleRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    player1_agent: str = Field(alias="player1Agent")
    player2_agent: str = Field(alias="player2Agent")
    num_games: int = Field(default=50, ge=1, le=200, alias="numGames")


def _get_session_id(request: Request) -> str:
    """Extract or generate session ID from request."""
    # Try to get from cookie first
    session_id = request.cookies.get("session_id")
    if session_id:
        return session_id
    # Fall back to header
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id
    # Generate new one (will be set in response)
    return uuid.uuid4().hex


def _set_session_cookie(response: Response, session_id: str) -> None:
    """Set session cookie on response."""
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,  # 7 days
    )


def create_app() -> FastAPI:
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

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.post("/api/new-game")
    def api_new_game(request: NewGameRequest) -> dict:
        seed = request.seed if request.seed is not None else secrets.randbits(32)
        store.seed = seed
        store.human_player = request.human_player
        store.starting_player = request.starting_player
        store.ai_opponent = request.ai_opponent
        store.state = simulate.new_game(seed, starting_player=request.starting_player)
        store.log.clear()
        store.history.clear()
        _log_new_game(store, starting_player=request.starting_player)

        return _serialize(store.require_state(), store.seed, store)

    @app.post("/api/ai-move")
    async def api_ai_move() -> dict:
        """Have the AI opponent make a move."""
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
                get_claude_move = _get_claude_move_func()
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
            viz_agent = _get_visualizing_agent(agent, ai_name)
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
        _apply_action(store, action)

        return _serialize(store.require_state(), store.seed, store)

    @app.get("/api/ai-agents")
    def api_ai_agents() -> list[dict]:
        """List available AI agents."""
        agents = []
        # Add neural agents sorted by iteration (highest first)
        all_neural = list(_neural_agents_extra)
        if _neural_name and _neural_name in AI_AGENTS:
            all_neural.append((_neural_agent, "neural", _neural_iter))
        all_neural.sort(key=lambda x: x[2], reverse=True)
        for i, (agent, name, iteration) in enumerate(all_neural):
            label = f"PPO iter {iteration}"
            if i == 0:
                label += " (Strongest)"
            agents.append({
                "id": name,
                "name": label,
                "description": f"Neural network trained for {iteration} iterations"
            })
        agents.extend([
            {"id": "heuristic", "name": "Heuristic (Default)", "description": "Balanced strategic AI"},
            {"id": "aggressive", "name": "Heuristic (Aggressive)", "description": "High aggression, bar-focused"},
            {"id": "defensive", "name": "Heuristic (Defensive)", "description": "Conservative, low aggression"},
            {"id": "queue_control", "name": "Heuristic (Queue Control)", "description": "Prioritizes queue front positioning"},
            {"id": "skunk_specialist", "name": "Heuristic (Skunk Specialist)", "description": "Values skunk plays higher"},
            {"id": "noisy", "name": "Heuristic (Noisy)", "description": "Human-like with random noise"},
            {"id": "online", "name": "Online Strategies", "description": "Reactive counter-play"},
            {"id": "random", "name": "Random", "description": "Plays random legal moves"},
        ])
        return agents

    @app.get("/api/ai-agents/battle")
    def api_ai_agents_for_battle() -> list[dict]:
        """List AI agents available for AI vs AI battles."""
        agents = []
        # Add neural agents sorted by iteration (highest first)
        all_neural = list(_neural_agents_extra)
        if _neural_name and _neural_name in AI_AGENTS:
            all_neural.append((_neural_agent, "neural", _neural_iter))
        all_neural.sort(key=lambda x: x[2], reverse=True)
        for i, (agent, name, iteration) in enumerate(all_neural):
            label = f"PPO iter {iteration}"
            if i == 0:
                label += " (Strongest)"
            agents.append({
                "id": name,
                "name": label,
                "description": f"Neural network trained for {iteration} iterations"
            })
        agents.extend([
            {"id": "heuristic", "name": "Heuristic (Default)", "description": "Balanced strategic AI"},
            {"id": "aggressive", "name": "Heuristic (Aggressive)", "description": "High aggression, bar-focused"},
            {"id": "defensive", "name": "Heuristic (Defensive)", "description": "Conservative, low aggression"},
            {"id": "queue_control", "name": "Heuristic (Queue Control)", "description": "Prioritizes queue front positioning"},
            {"id": "skunk_specialist", "name": "Heuristic (Skunk Specialist)", "description": "Values skunk plays higher"},
            {"id": "noisy", "name": "Heuristic (Noisy)", "description": "Human-like with random noise"},
            {"id": "online", "name": "Online Strategies", "description": "Reactive counter-play"},
            {"id": "random", "name": "Random", "description": "Plays random legal moves"},
        ])
        return agents

    @app.post("/api/ai-battle/start")
    def api_ai_battle_start(request: AIBattleRequest) -> dict:
        """Run multiple AI vs AI games and return full history for replay."""
        # Validate agents
        p1_agent_id = request.player1_agent
        p2_agent_id = request.player2_agent

        if p1_agent_id not in AI_AGENTS:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {p1_agent_id}")
        if p2_agent_id not in AI_AGENTS:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {p2_agent_id}")

        p1_agent = AI_AGENTS[p1_agent_id]
        p2_agent = AI_AGENTS[p2_agent_id]
        agents = [p1_agent, p2_agent]

        # Create visualizing wrappers for neural agents
        viz_wrappers: list = [None, None]
        is_neural = [False, False]
        try:
            from _02_agents.neural import NeuralAgent
            from _04_ui.visualization.activation_capture import VisualizingNeuralAgent

            for i, (agent, agent_id) in enumerate(zip(agents, [p1_agent_id, p2_agent_id])):
                if isinstance(agent, NeuralAgent):
                    viz_wrappers[i] = VisualizingNeuralAgent(agent, websocket_manager=None)
                    is_neural[i] = True
        except ImportError:
            pass

        games: list[dict] = []
        wins = [0, 0]

        for game_num in range(request.num_games):
            # Create new game with random seed
            seed = secrets.randbits(32)
            game_state = simulate.new_game(seed, starting_player=game_num % 2)

            turns: list[dict] = []
            # Record initial state
            turns.append({
                "turnNumber": 0,
                "activePlayer": game_state.active_player,
                "action": None,
                "state": _serialize_battle_state(game_state),
                "events": [],
                "activations": None,
            })

            # Play game to completion
            while not simulate.is_terminal(game_state):
                player = game_state.active_player
                agent = agents[player]
                viz_wrapper = viz_wrappers[player]
                legal = simulate.legal_actions(game_state, player)

                # Build game context for visualization
                game_context = {
                    "queue_cards": [c.species for c in game_state.zones.queue],
                    "bar_cards": [c.species for c in game_state.zones.beasty_bar],
                    "hand_cards": [c.species for c in game_state.players[player].hand],
                }

                # Get agent's move with optional activation capture
                masked_state = state.mask_state_for_player(game_state, player)
                activation_snapshot = None

                if viz_wrapper is not None:
                    action, activation_snapshot = viz_wrapper.select_action_with_capture_sync(
                        masked_state, legal, game_state.turn, player, game_context
                    )
                else:
                    action = agent(masked_state, legal)

                # Apply action and get trace for events
                new_state, steps = engine.step_with_trace(game_state, action)

                # Extract events from steps for highlights
                events = []
                for step in steps:
                    for event in step.events:
                        events.append({"phase": step.name, "description": event})

                # Get card info for the action
                card = game_state.players[player].hand[action.hand_index]

                turns.append({
                    "turnNumber": game_state.turn,
                    "activePlayer": player,
                    "action": {
                        "handIndex": action.hand_index,
                        "params": list(action.params),
                        "card": _card_view(card),
                        "label": _action_label(card, action),
                    },
                    "state": _serialize_battle_state(new_state),
                    "events": events,
                    "activations": activation_snapshot,
                })

                game_state = new_state

            # Game finished
            final_scores = simulate.score(game_state)
            winner = 0 if final_scores[0] > final_scores[1] else (1 if final_scores[1] > final_scores[0] else -1)
            if winner >= 0:
                wins[winner] += 1

            games.append({
                "gameNumber": game_num,
                "seed": seed,
                "winner": winner,
                "finalScores": list(final_scores),
                "turns": turns,
            })

        # Cleanup viz wrappers
        for wrapper in viz_wrappers:
            if wrapper is not None:
                wrapper.cleanup()

        return {
            "player1Agent": p1_agent_id,
            "player2Agent": p2_agent_id,
            "numGames": request.num_games,
            "wins": wins,
            "games": games,
            "isNeural": is_neural,
        }

    @app.get("/api/claude-state")
    def api_claude_state() -> dict:
        """Get game state formatted for Claude Code to read and make a move."""
        game_state = store.require_state()
        ai_player = 1 - store.human_player
        legal = simulate.legal_actions(game_state, ai_player) if not simulate.is_terminal(game_state) else ()

        # Format state for Claude
        queue_str = _format_queue_for_claude(game_state.zones.queue)
        hand_str = _format_hand_for_claude(game_state.players[ai_player].hand, legal)
        legal_str = _format_legal_actions_for_claude(game_state, legal)

        formatted = f"""## Beasty Bar - Claude's Turn (Player {ai_player})

**Queue** (left=Heaven's Gate, right=That's It):
{queue_str}

**Your Hand:**
{hand_str}

**Legal Actions:**
{legal_str}

**Scores:** You {_count_score(game_state.zones.beasty_bar, ai_player)} - Opponent {_count_score(game_state.zones.beasty_bar, store.human_player)}

Reply with the action number (e.g., "1" or "3")."""

        return {
            "formatted": formatted,
            "legalActions": [
                {"index": i + 1, **_serialize_action(game_state, action)}
                for i, action in enumerate(legal)
            ],
            "isClaudeTurn": game_state.active_player == ai_player and not simulate.is_terminal(game_state),
        }

    @app.post("/api/claude-move")
    def api_claude_move(payload: dict) -> dict:
        """Apply Claude Code's move (entered by user)."""
        game_state = store.require_state()
        ai_player = 1 - store.human_player

        if simulate.is_terminal(game_state):
            raise HTTPException(status_code=400, detail="Game is already over")

        if game_state.active_player != ai_player:
            raise HTTPException(status_code=400, detail="It's not Claude's turn")

        action_index = payload.get("actionIndex")
        if action_index is None:
            raise HTTPException(status_code=400, detail="Missing actionIndex")

        legal = simulate.legal_actions(game_state, ai_player)
        if action_index < 1 or action_index > len(legal):
            raise HTTPException(status_code=400, detail=f"Invalid action index. Choose 1-{len(legal)}")

        action = legal[action_index - 1]
        _apply_action(store, action)

        return _serialize(store.require_state(), store.seed, store)

    @app.post("/api/claude-bridge/write-state")
    def api_claude_write_state() -> dict:
        """Write game state to file for Claude Code to read."""
        game_state = store.require_state()
        ai_player = 1 - store.human_player

        if game_state.active_player != ai_player:
            return {"written": False, "reason": "Not Claude's turn"}

        if simulate.is_terminal(game_state):
            return {"written": False, "reason": "Game is over"}

        legal = simulate.legal_actions(game_state, ai_player)

        # Ensure bridge directory exists
        CLAUDE_BRIDGE_DIR.mkdir(exist_ok=True)

        # Clear any old move file
        if CLAUDE_MOVE_FILE.exists():
            CLAUDE_MOVE_FILE.unlink()

        # Format state for Claude
        queue_str = _format_queue_for_claude(game_state.zones.queue)
        hand_str = _format_hand_for_claude(game_state.players[ai_player].hand, legal)
        legal_str = _format_legal_actions_for_claude(game_state, legal)

        state_data = {
            "turn": game_state.turn,
            "player": ai_player,
            "formatted": f"""## Beasty Bar - Claude's Turn (Player {ai_player})

**Queue** (left=Heaven's Gate, right=That's It):
{queue_str}

**Your Hand:**
{hand_str}

**Legal Actions:**
{legal_str}

**Scores:** You {_count_score(game_state.zones.beasty_bar, ai_player)} - Opponent {_count_score(game_state.zones.beasty_bar, store.human_player)}

Reply with JUST the action number (e.g., "1" or "3").""",
            "legalActions": [
                {"index": i + 1, "label": _action_label(game_state.players[ai_player].hand[a.hand_index], a)}
                for i, a in enumerate(legal)
            ],
        }

        CLAUDE_STATE_FILE.write_text(json.dumps(state_data, indent=2))
        return {"written": True, "path": str(CLAUDE_STATE_FILE)}

    @app.get("/api/claude-bridge/check-move")
    def api_claude_check_move() -> dict:
        """Check if Claude Code has written a move response."""
        if not CLAUDE_MOVE_FILE.exists():
            return {"hasMove": False}

        try:
            move_data = json.loads(CLAUDE_MOVE_FILE.read_text())
            return {"hasMove": True, "actionIndex": move_data.get("actionIndex")}
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in move file: {e}")
            return {"hasMove": False, "error": "Invalid JSON in move file"}
        except FileNotFoundError:
            # File was deleted between exists() check and read
            return {"hasMove": False}
        except OSError as e:
            logger.warning(f"Error reading move file: {e}")
            return {"hasMove": False, "error": "Error reading move file"}

    @app.post("/api/claude-bridge/apply-move")
    def api_claude_apply_move() -> dict:
        """Apply the move from Claude's response file."""
        if not CLAUDE_MOVE_FILE.exists():
            raise HTTPException(status_code=400, detail="No move file found")

        try:
            move_data = json.loads(CLAUDE_MOVE_FILE.read_text())
            action_index = move_data.get("actionIndex")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid move file") from exc

        # Clean up bridge files
        if CLAUDE_STATE_FILE.exists():
            CLAUDE_STATE_FILE.unlink()
        if CLAUDE_MOVE_FILE.exists():
            CLAUDE_MOVE_FILE.unlink()

        # Apply the move
        game_state = store.require_state()
        ai_player = 1 - store.human_player
        legal = simulate.legal_actions(game_state, ai_player)

        if action_index < 1 or action_index > len(legal):
            raise HTTPException(status_code=400, detail=f"Invalid action index: {action_index}")

        action = legal[action_index - 1]
        _apply_action(store, action)

        return _serialize(store.require_state(), store.seed, store)

    @app.get("/api/state")
    def api_state() -> dict:
        return _serialize(store.require_state(), store.seed, store)

    @app.get("/api/legal-actions")
    def api_legal_actions() -> list[dict]:
        game_state = store.require_state()
        player = game_state.active_player
        legal = simulate.legal_actions(game_state, player)
        return [_serialize_action(game_state, action) for action in legal]

    @app.post("/api/action")
    def api_action(payload: ActionPayload) -> dict:
        game_state = store.require_state()
        player = game_state.active_player
        legal = simulate.legal_actions(game_state, player)
        params_tuple = tuple(payload.params)
        try:
            action = next(act for act in legal if act.hand_index == payload.hand_index and act.params == params_tuple)
        except StopIteration as exc:
            raise HTTPException(status_code=400, detail="Illegal action") from exc

        _apply_action(store, action)

        return _serialize(store.require_state(), store.seed, store)

    @app.post("/api/replay")
    def api_replay() -> dict:
        current = store.require_state()
        if store.seed is None:
            raise HTTPException(status_code=400, detail="No seed available to replay")

        candidate = simulate.new_game(store.seed, starting_player=store.starting_player)
        for index, action in enumerate(store.history, start=1):
            try:
                candidate = engine.step(candidate, action)
            except Exception as exc:  # pragma: no cover - defensive safeguard
                raise HTTPException(
                    status_code=400,
                    detail=f"Replay failed while applying action {index}",
                ) from exc

        if candidate != current:
            raise HTTPException(status_code=409, detail="Replay diverged from recorded state")

        return _serialize(current, store.seed, store)

    # =========================================================================
    # Neural Network Visualization Endpoints
    # =========================================================================

    @app.get("/visualizer")
    def visualizer_page() -> FileResponse:
        """Serve the neural network visualizer dashboard."""
        return FileResponse(static_dir / "visualizer.html")

    @app.get("/battle-visualizer")
    def battle_visualizer_page() -> FileResponse:
        """Serve the AI battle visualizer with dual network graphs."""
        return FileResponse(static_dir / "battle_visualizer.html")

    @app.websocket("/ws/visualizer")
    async def websocket_visualizer(websocket: WebSocket):
        """WebSocket endpoint for real-time activation streaming."""
        viz_manager = _get_viz_manager()
        await viz_manager.connect(websocket)

        try:
            while True:
                data = await websocket.receive_text()
                response = await viz_manager.handle_client_message(websocket, data)
                if response:
                    await viz_manager.send_to_client(websocket, response)
        except WebSocketDisconnect:
            await viz_manager.disconnect(websocket)

    @app.get("/api/viz/status")
    def api_viz_status() -> dict:
        """Get visualization system status."""
        viz_manager = _get_viz_manager()
        return {
            "connected_clients": viz_manager.connection_count,
            "neural_agents_available": list(_viz_capture_agents.keys()),
        }

    @app.get("/api/viz/history")
    def api_viz_history() -> list[dict]:
        """Get activation history for replay mode."""
        from _04_ui.visualization.data_compression import snapshot_to_dict

        # Find any visualizing agent with history
        for agent_name, viz_agent in _viz_capture_agents.items():
            if viz_agent.activation_history:
                return [
                    snapshot_to_dict(snapshot)
                    for snapshot in viz_agent.activation_history
                ]
        return []

    @app.post("/api/viz/clear-history")
    def api_viz_clear_history() -> dict:
        """Clear activation history (call on new game)."""
        for viz_agent in _viz_capture_agents.values():
            viz_agent.clear_history()
        return {"status": "cleared"}

    # =========================================================================
    # Move Explanation Endpoint (Phase 6)
    # =========================================================================

    @app.post("/api/explain-move")
    def api_explain_move(payload: dict) -> dict:
        """Explain why the AI chose a particular move.

        This endpoint provides insights into the neural network's decision-making
        process, including feature importance, alternative actions considered,
        and confidence levels.

        Request body:
            - action_index (int): Index of the action to explain (1-based)
            - agent_name (str, optional): Which AI agent to use for explanation

        Returns:
            - chosen_action: Details about the selected action
            - alternatives: Other actions that were considered
            - confidence: How confident the AI is in this choice
            - value_prediction: Expected game outcome
            - top_factors: Most important factors in the decision
            - reasoning: Human-readable explanation
        """
        game_state = store.require_state()

        if simulate.is_terminal(game_state):
            raise HTTPException(status_code=400, detail="Game is already over")

        player = game_state.active_player
        legal = simulate.legal_actions(game_state, player)

        if not legal:
            raise HTTPException(status_code=400, detail="No legal actions available")

        action_index = payload.get("actionIndex", 1)
        agent_name = payload.get("agentName", "neural")

        if action_index < 1 or action_index > len(legal):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action index. Choose 1-{len(legal)}"
            )

        # Get the neural agent
        if agent_name not in AI_AGENTS:
            agent_name = "neural" if "neural" in AI_AGENTS else "heuristic"

        agent = AI_AGENTS.get(agent_name)

        # Check if agent is a neural agent
        try:
            from _02_agents.neural import NeuralAgent
            if not isinstance(agent, NeuralAgent):
                return {
                    "error": "Explanation only available for neural agents",
                    "agent_type": type(agent).__name__,
                }
        except ImportError:
            return {"error": "Neural agent module not available"}

        # Get observation and explain
        try:
            from _01_simulator.observations import state_to_tensor
            from _02_agents.neural.explain import MoveExplainer, format_explanation_for_api

            # Convert state to tensor for the neural network
            masked_state = state.mask_state_for_player(game_state, player)
            observation = state_to_tensor(masked_state, player)

            # Get action indices
            chosen_action = legal[action_index - 1]
            legal_indices = list(range(len(legal)))

            # Get card species for labels
            card_species = [
                game_state.players[player].hand[a.hand_index].species
                for a in legal
            ]

            # Create explainer and get explanation
            explainer = MoveExplainer(agent.model)
            explanation = explainer.explain(
                observation=observation,
                chosen_action_idx=action_index - 1,
                legal_action_indices=legal_indices,
                card_species=card_species,
            )

            # Format for API response
            result = format_explanation_for_api(explanation)
            result["action_labels"] = [
                _action_label(game_state.players[player].hand[a.hand_index], a)
                for a in legal
            ]
            return result

        except Exception as e:
            logger.exception("Error generating explanation")
            return {
                "error": f"Failed to generate explanation: {str(e)}",
                "action_index": action_index,
            }

    # =========================================================================
    # Benchmark Endpoint (Phase 6)
    # =========================================================================

    @app.get("/api/benchmark")
    def api_benchmark() -> dict:
        """Get inference benchmark results for the current neural model.

        Returns performance metrics including latency, throughput,
        and memory usage for the loaded neural network.
        """
        if "neural" not in AI_AGENTS:
            return {"error": "No neural agent loaded"}

        try:
            from _02_agents.neural import NeuralAgent
            from _03_training.benchmark import benchmark_model, BenchmarkConfig

            agent = AI_AGENTS["neural"]
            if not isinstance(agent, NeuralAgent):
                return {"error": "Neural agent not available"}

            config = BenchmarkConfig(
                num_iterations=100,
                warmup_iterations=10,
                batch_sizes=[1, 8],
                include_memory=True,
                include_games=False,
            )

            result = benchmark_model(
                agent.model,
                config=config,
                model_name="neural",
            )

            return {
                "model_name": result.model_name,
                "device": result.device,
                "latency": [
                    {
                        "batch_size": lat.batch_size,
                        "avg_ms": round(lat.avg_ms, 3),
                        "p95_ms": round(lat.p95_ms, 3),
                    }
                    for lat in result.latency
                ],
                "memory_mb": round(result.memory.model_size_mb, 1) if result.memory else None,
                "throughput_per_sec": round(result.throughput.inferences_per_second, 0) if result.throughput else None,
            }
        except Exception as e:
            logger.exception("Benchmark failed")
            return {"error": str(e)}

    return app


def _serialize_battle_state(game_state: state.State) -> dict:
    """Serialize game state for AI battle replay (both hands visible)."""
    return {
        "turn": game_state.turn,
        "activePlayer": game_state.active_player,
        "isTerminal": simulate.is_terminal(game_state),
        "score": simulate.score(game_state) if simulate.is_terminal(game_state) else None,
        "queue": [_card_view(card) for card in game_state.zones.queue],
        "zones": {
            "beastyBar": [_card_view(card) for card in game_state.zones.beasty_bar],
            "thatsIt": [_card_view(card) for card in game_state.zones.thats_it],
        },
        "hands": [[_card_view(card) for card in player_state.hand] for player_state in game_state.players],
    }


def _serialize(game_state: state.State, seed: int | None, store: SessionStore) -> dict:
    active_player = game_state.active_player
    legal = simulate.legal_actions(game_state, active_player) if not simulate.is_terminal(game_state) else ()
    log_entries = list(store.log)
    visible_state = game_state
    is_ai_turn = active_player != store.human_player and not simulate.is_terminal(game_state)
    return {
        "seed": seed,
        "turn": game_state.turn,
        "activePlayer": active_player,
        "isTerminal": simulate.is_terminal(game_state),
        "score": simulate.score(game_state) if simulate.is_terminal(game_state) else None,
        "humanPlayer": store.human_player,
        "aiOpponent": store.ai_opponent,
        "isAiTurn": is_ai_turn,
        "queue": [_card_view(card) for card in visible_state.zones.queue],
        "zones": {
            "beastyBar": [_card_view(card) for card in visible_state.zones.beasty_bar],
            "thatsIt": [_card_view(card) for card in visible_state.zones.thats_it],
        },
        "hands": [[_card_view(card) for card in player_state.hand] for player_state in visible_state.players],
        "legalActions": [_serialize_action(game_state, action) for action in legal],
        "log": [_serialize_log_entry(entry) for entry in log_entries],
        "logText": _format_log_text(store, log_entries),
        "turnFlow": _serialize_turn_steps(log_entries[-1].steps) if log_entries else [],
    }


def _serialize_action(game_state: state.State, action: actions.Action) -> dict:
    player = game_state.active_player
    card = game_state.players[player].hand[action.hand_index]
    return {
        "handIndex": action.hand_index,
        "params": list(action.params),
        "card": _card_view(card),
        "label": _action_label(card, action),
    }


def _card_view(card: state.Card) -> dict:
    return {
        "owner": card.owner,
        "species": card.species,
        "strength": card.strength,
        "points": card.points,
    }


def _action_label(card: state.Card, action: actions.Action) -> str:
    species = card.species
    if species == "kangaroo":
        if action.params:
            hop = action.params[0]
            return f"Play {species} (hop {hop})"
        return f"Play {species}"
    if action.params:
        params = ",".join(str(p) for p in action.params)
        return f"Play {species} ({params})"
    return f"Play {species}"


def _apply_action(
    store: SessionStore,
    action: actions.Action,
    *,
    record_history: bool = True,
) -> None:
    before = store.require_state()
    player = before.active_player
    try:
        card = before.players[player].hand[action.hand_index]
    except IndexError as exc:  # pragma: no cover - validation upstream should prevent this
        raise HTTPException(status_code=400, detail="Invalid hand index") from exc

    after, steps = engine.step_with_trace(before, action)
    store.state = after
    if record_history:
        store.history.append(action)

    effects = _turn_flow_summary(steps)
    effects.extend(_describe_action_effects(store, before, after, player, include_draw=False))
    if simulate.is_terminal(after):
        final_score = simulate.score(after)
        summary = "Game over: " + " - ".join(f"P{idx} {score}" for idx, score in enumerate(final_score))
        effects.append(summary)
    entry = TurnLogEntry(
        id=_make_log_id(),
        timestamp=datetime.now(timezone.utc),
        turn=after.turn,
        player=player,
        action=_action_label(card, action),
        effects=tuple(effects),
        steps=steps,
    )
    _append_log_entry(store, entry)


def _log_new_game(store: SessionStore, *, starting_player: int) -> None:
    details: list[str] = []
    if store.seed is not None:
        details.append(f"Seed {store.seed}")
    details.append(f"Starting player P{starting_player}")
    details.append(f"You are P{store.human_player}")

    entry = TurnLogEntry(
        id=_make_log_id(),
        timestamp=datetime.now(timezone.utc),
        turn=0,
        player=None,
        action="New game started",
        effects=tuple(details),
        steps=(),
    )
    _append_log_entry(store, entry)


def _append_log_entry(store: SessionStore, entry: TurnLogEntry, *, max_entries: int = 100) -> None:
    store.log.append(entry)
    if len(store.log) > max_entries:
        store.log[:] = store.log[-max_entries:]


def _make_log_id() -> str:
    return uuid.uuid4().hex


def _turn_flow_summary(steps: tuple[engine.TurnStep, ...]) -> list[str]:
    lines: list[str] = []
    for idx, step in enumerate(steps, start=1):
        title = step.name.title()
        if step.events:
            lines.append(f"Step {idx} - {title}: {step.events[0]}")
            for extra in step.events[1:]:
                lines.append(f"  ↳ {extra}")
        else:
            lines.append(f"Step {idx} - {title}: No effect.")
    return lines


def _describe_action_effects(
    store: SessionStore,
    before: state.State,
    after: state.State,
    player: int,
    *,
    include_draw: bool = True,
) -> list[str]:
    effects: list[str] = []

    scored = _zone_new_cards(before.zones.beasty_bar, after.zones.beasty_bar)
    if scored:
        owners: dict[int, int] = defaultdict(int)
        for card in scored:
            owners[card.owner] += card.points
        gains = ", ".join(f"P{owner} +{points}" for owner, points in sorted(owners.items()))
        effects.append(f"Heaven's Gate: {_format_card_list(scored)} ({gains})")

    bounced = _zone_new_cards(before.zones.thats_it, after.zones.thats_it)
    if bounced:
        effects.append(f"Sent to THAT'S IT: {_format_card_list(bounced)}")

    if include_draw:
        draw = _drawn_cards(before, after, player)
        if draw:
            if player == store.human_player:
                effects.append(f"Drew {_format_card_list(draw)}")
            else:
                count = len(draw)
                label = "card" if count == 1 else "cards"
                effects.append(f"Drew {count} {label}")

    return effects


def _card_key(card: state.Card) -> tuple[int, str, int, int]:
    """Create a value-based key for card comparison."""
    return (card.owner, card.species, card.strength, card.points)


def _drawn_cards(before: state.State, after: state.State, player: int) -> list[state.Card]:
    """Find cards in after hand that weren't in before hand."""
    before_keys = [_card_key(card) for card in before.players[player].hand]
    new_cards = []
    for card in after.players[player].hand:
        key = _card_key(card)
        if key in before_keys:
            before_keys.remove(key)  # Handle duplicates
        else:
            new_cards.append(card)
    return new_cards


def _zone_new_cards(before_cards: tuple[state.Card, ...], after_cards: tuple[state.Card, ...]) -> list[state.Card]:
    """Find cards in after zone that weren't in before zone."""
    before_keys = [_card_key(card) for card in before_cards]
    new_cards = []
    for card in after_cards:
        key = _card_key(card)
        if key in before_keys:
            before_keys.remove(key)  # Handle duplicates
        else:
            new_cards.append(card)
    return new_cards


def _format_card_list(cards: list[state.Card]) -> str:
    return ", ".join(f"{card.species} (P{card.owner})" for card in cards)


def _serialize_log_entry(entry: TurnLogEntry) -> dict:
    timestamp = entry.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    iso_ts = timestamp.isoformat(timespec="milliseconds")
    epoch_ms = int(timestamp.timestamp() * 1000)
    return {
        "id": entry.id,
        "turn": entry.turn,
        "player": entry.player,
        "action": entry.action,
        "effects": list(entry.effects),
        "timestamp": iso_ts,
        "timestampMs": epoch_ms,
        "steps": _serialize_turn_steps(entry.steps),
    }


def _serialize_turn_steps(steps: tuple[engine.TurnStep, ...]) -> list[dict]:
    return [
        {
            "name": step.name,
            "events": list(step.events),
        }
        for step in steps
    ]


def _format_log_text(store: SessionStore, entries: list[TurnLogEntry]) -> str:
    if not entries:
        return ""

    lines: list[str] = []
    for entry in entries:
        if entry.player is None:
            header = f"Turn {entry.turn} - Setup: {entry.action}"
        else:
            label = _log_player_label(store, entry.player)
            header = f"Turn {entry.turn} - {label}: {entry.action}"
        lines.append(header)
        for effect in entry.effects:
            lines.append(f"  - {effect}")
    return "\n".join(lines)


def _log_player_label(store: SessionStore, player: int) -> str:
    if player == store.human_player:
        return f"P{player} (You)"
    return f"P{player}"


def _format_queue_for_claude(queue: tuple[state.Card, ...]) -> str:
    if not queue:
        return "[empty]"
    cards = []
    for i, card in enumerate(queue):
        owner = "You" if card.owner == 1 else "Opp"
        cards.append(f"{i}: {card.species}({card.strength}) [{owner}]")
    return " -> ".join(cards)


def _format_hand_for_claude(hand: tuple[state.Card, ...], legal: tuple[actions.Action, ...]) -> str:
    if not hand:
        return "[empty]"
    legal_indices = {a.hand_index for a in legal}
    cards = []
    for i, card in enumerate(hand):
        playable = "*" if i in legal_indices else " "
        cards.append(f"[{i}]{playable} {card.species}({card.strength}, {card.points}pts)")
    return "\n".join(cards)


def _format_legal_actions_for_claude(game_state: state.State, legal: tuple[actions.Action, ...]) -> str:
    if not legal:
        return "[none]"
    player = game_state.active_player
    lines = []
    for i, action in enumerate(legal, 1):
        card = game_state.players[player].hand[action.hand_index]
        label = _action_label(card, action)
        lines.append(f"{i}. {label}")
    return "\n".join(lines)


def _count_score(beasty_bar: tuple[state.Card, ...], player: int) -> int:
    return sum(c.points for c in beasty_bar if c.owner == player)


__all__ = ["create_app"]

# Create app instance for uvicorn
app = create_app()
