"""AI agent listing and battle API routes."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException

from _01_simulator import simulate, state
from _04_ui.models.requests import AIBattleRequest  # noqa: TC001
from _04_ui.services.ai import (
    AI_AGENTS,
    get_neural_agent_info,
    get_neural_agents_extra,
)
from _04_ui.services.serializer import action_label, card_view, serialize_battle_state

router = APIRouter(prefix="/api", tags=["agents"])


def _build_agent_list() -> list[dict]:
    """Build the list of available AI agents."""
    _neural_agent, _neural_name, _neural_iter = get_neural_agent_info()
    _neural_agents_extra = get_neural_agents_extra()

    agents = []
    # Add tablebase-enhanced agent if available (strongest option)
    if "ppo_iter949_tablebase" in AI_AGENTS:
        agents.append({
            "id": "ppo_iter949_tablebase",
            "name": "Neural + Tablebase",
            "description": "Trained neural net with perfect endgame lookup — the strongest agent"
        })
    # Add neural agents sorted by iteration (highest first)
    all_neural = list(_neural_agents_extra)
    if _neural_name and _neural_name in AI_AGENTS:
        all_neural.append((_neural_agent, "neural", _neural_iter))
    all_neural.sort(key=lambda x: x[2], reverse=True)
    seen_neural = False
    for _agent, name, iteration in all_neural:
        if not seen_neural:
            label = "Neural PPO"
            desc = "Transformer trained via self-play PPO (949 iters, 15M games) — 75.7% win rate"
            seen_neural = True
        else:
            label = f"Neural (iter {iteration})"
            desc = f"Earlier training snapshot at iteration {iteration}"
        agents.append({
            "id": name,
            "name": label,
            "description": desc,
        })
    agents.extend([
        {"id": "heuristic", "name": "Heuristic", "description": "Hand-coded balanced strategy"},
        {"id": "aggressive", "name": "Aggressive", "description": "Prioritizes scoring over defense"},
        {"id": "defensive", "name": "Defensive", "description": "Conservative, avoids risk"},
        {"id": "queue_control", "name": "Queue Control", "description": "Prioritizes queue front positioning"},
        {"id": "skunk_specialist", "name": "Skunk Expert", "description": "Specializes in skunk card plays"},
        {"id": "noisy", "name": "Human-like", "description": "Adds random noise to simulate human play"},
        {"id": "online", "name": "Counter-Play", "description": "Adapts strategy based on opponent moves"},
        {"id": "random", "name": "Random", "description": "Plays random legal moves — the baseline"},
    ])
    return agents


@router.get("/ai-agents")
def api_ai_agents() -> list[dict]:
    """List available AI agents."""
    return _build_agent_list()


@router.get("/ai-agents/battle")
def api_ai_agents_for_battle() -> list[dict]:
    """List AI agents available for AI vs AI battles."""
    return _build_agent_list()


@router.post("/ai-battle/start")
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

        for i, (agent, _agent_id) in enumerate(zip(agents, [p1_agent_id, p2_agent_id], strict=False)):
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
            "state": serialize_battle_state(game_state),
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
            from _01_simulator import engine
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
                    "card": card_view(card),
                    "label": action_label(card, action),
                },
                "state": serialize_battle_state(new_state),
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
