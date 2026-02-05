"""Neural network visualization API routes."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from _01_simulator import simulate, state
from _04_ui.core.session import GameSession  # noqa: TC001
from _04_ui.services.ai import AI_AGENTS, get_viz_capture_agents
from _04_ui.services.serializer import action_label

logger = logging.getLogger(__name__)

router = APIRouter(tags=["visualization"])

# Store reference - will be set by app factory
_store: GameSession | None = None
_static_dir: Path | None = None

# Lazy viz manager
_viz_manager = None


def _get_viz_manager():
    """Lazy import of visualization WebSocket manager."""
    global _viz_manager
    if _viz_manager is None:
        from _04_ui.visualization.websocket_manager import visualizer_ws_manager
        _viz_manager = visualizer_ws_manager
    return _viz_manager


def set_store(store: GameSession) -> None:
    """Set the session store for this router."""
    global _store
    _store = store


def set_static_dir(static_dir: Path) -> None:
    """Set the static directory path."""
    global _static_dir
    _static_dir = static_dir


def get_store() -> GameSession:
    """Get the session store."""
    if _store is None:
        raise RuntimeError("Store not initialized")
    return _store


@router.get("/visualizer")
def visualizer_page() -> FileResponse:
    """Serve the neural network visualizer dashboard."""
    if _static_dir is None:
        raise HTTPException(status_code=500, detail="Static directory not configured")
    return FileResponse(_static_dir / "visualizer.html")


@router.get("/battle-visualizer")
def battle_visualizer_page() -> FileResponse:
    """Serve the AI battle visualizer with dual network graphs."""
    if _static_dir is None:
        raise HTTPException(status_code=500, detail="Static directory not configured")
    return FileResponse(_static_dir / "battle_visualizer.html")


@router.websocket("/ws/visualizer")
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


@router.get("/api/viz/status")
def api_viz_status() -> dict:
    """Get visualization system status."""
    viz_manager = _get_viz_manager()
    viz_capture_agents = get_viz_capture_agents()
    return {
        "connected_clients": viz_manager.connection_count,
        "neural_agents_available": list(viz_capture_agents.keys()),
    }


@router.get("/api/viz/history")
def api_viz_history() -> list[dict]:
    """Get activation history for replay mode."""
    from _04_ui.visualization.data_compression import snapshot_to_dict

    viz_capture_agents = get_viz_capture_agents()
    # Find any visualizing agent with history
    for _agent_name, viz_agent in viz_capture_agents.items():
        if viz_agent.activation_history:
            return [
                snapshot_to_dict(snapshot)
                for snapshot in viz_agent.activation_history
            ]
    return []


@router.post("/api/viz/clear-history")
def api_viz_clear_history() -> dict:
    """Clear activation history (call on new game)."""
    viz_capture_agents = get_viz_capture_agents()
    for viz_agent in viz_capture_agents.values():
        viz_agent.clear_history()
    return {"status": "cleared"}


@router.post("/api/explain-move")
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
    store = get_store()
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
        legal[action_index - 1]
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
            action_label(game_state.players[player].hand[a.hand_index], a)
            for a in legal
        ]
        return result

    except Exception as e:
        logger.exception("Error generating explanation")
        return {
            "error": f"Failed to generate explanation: {e!s}",
            "action_index": action_index,
        }


@router.get("/api/benchmark")
def api_benchmark() -> dict:
    """Get inference benchmark results for the current neural model.

    Returns performance metrics including latency, throughput,
    and memory usage for the loaded neural network.
    """
    if "neural" not in AI_AGENTS:
        return {"error": "No neural agent loaded"}

    try:
        from _02_agents.neural import NeuralAgent
        from _03_training.benchmark import BenchmarkConfig, benchmark_model

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
