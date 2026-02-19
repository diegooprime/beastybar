"""AI agent management for the Beasty Bar web UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from _02_agents import HeuristicAgent, HeuristicConfig, RandomAgent
from _02_agents.heuristic import OnlineStrategies

if TYPE_CHECKING:
    from _04_ui.visualization.visualizing_agent import VisualizingNeuralAgent

logger = logging.getLogger(__name__)

# Visualization capture agents cache
_viz_capture_agents: dict[str, VisualizingNeuralAgent] = {}

# Lazy viz manager reference
_viz_manager = None


def _get_viz_manager():
    """Lazy import of visualization WebSocket manager."""
    global _viz_manager
    if _viz_manager is None:
        from _04_ui.visualization.websocket_manager import visualizer_ws_manager

        _viz_manager = visualizer_ws_manager
    return _viz_manager


def get_visualizing_agent(agent, agent_name: str):
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


def get_viz_capture_agents() -> dict:
    """Get the viz capture agents dict for external access."""
    return _viz_capture_agents


def get_claude_move_func():
    """Lazy import of claude agent to avoid circular imports."""
    from _04_ui.claude_agent import get_claude_move

    return get_claude_move


def _load_neural_agent(ckpt_path: str | Path | None = None) -> tuple:
    """Try to load neural agent from checkpoint.

    Returns (agent, name, iteration) or (None, None, None).
    """
    try:
        import os

        import torch

        from _02_agents.neural import NeuralAgent
        from _02_agents.neural.network import BeastyBarNetwork
        from _02_agents.neural.utils import NetworkConfig

        def load_from_checkpoint(path):
            """Load network from PPO or MCTS checkpoint."""
            # weights_only=False required (pickle protocol 4 checkpoint)
            # Safe: checkpoint path is validated to be under allowed directories
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state_dict"]

            # Extract network config (nested in training config)
            config_dict = checkpoint.get("config", {})
            if "network_config" in config_dict:
                config_dict = config_dict["network_config"]

            # Auto-detect deep_value_head from state_dict shape
            hidden_dim = config_dict.get("hidden_dim", 256)
            value_head_0_shape = state_dict.get("value_head.0.weight", torch.zeros(1)).shape
            deep_value_head = len(value_head_0_shape) >= 2 and value_head_0_shape[0] == hidden_dim

            net_config = NetworkConfig(
                observation_dim=config_dict.get("observation_dim", 988),
                action_dim=config_dict.get("action_dim", 124),
                hidden_dim=hidden_dim,
                num_heads=config_dict.get("num_heads", 8),
                num_layers=config_dict.get("num_layers", 4),
                dropout=config_dict.get("dropout", 0.1),
                species_embedding_dim=config_dict.get("species_embedding_dim", 64),
                card_feature_dim=config_dict.get("card_feature_dim", 17),
                num_species=config_dict.get("num_species", 12),
                max_queue_length=config_dict.get("max_queue_length", 5),
                max_bar_length=config_dict.get("max_bar_length", 24),
                hand_size=config_dict.get("hand_size", 4),
                deep_value_head=deep_value_head,
            )

            # Create and load network
            network = BeastyBarNetwork(net_config)
            network.load_state_dict(state_dict)
            network.eval()
            return network, checkpoint.get("iteration", 0)

        # Check for checkpoint path in environment or explicit path
        checkpoint_path = ckpt_path or os.environ.get("NEURAL_CHECKPOINT", None)
        if checkpoint_path:
            # Validate path is under allowed directories
            resolved = Path(checkpoint_path).resolve()
            allowed_dirs = [Path("checkpoints").resolve(), Path(".").resolve()]
            if not any(str(resolved).startswith(str(d)) for d in allowed_dirs):
                logger.warning("Rejected checkpoint path outside allowed dirs: %s", resolved)
                return None, None, None
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
    except Exception:
        logger.exception("Failed to load neural agent")
        return None, None, None


# AI opponent instances
AI_AGENTS: dict = {
    "random": RandomAgent(seed=None),
    "heuristic": HeuristicAgent(seed=None),
    "aggressive": HeuristicAgent(config=HeuristicConfig(bar_weight=3.0, aggression=0.8)),
    "defensive": HeuristicAgent(config=HeuristicConfig(bar_weight=1.0, aggression=0.2)),
    "queue_control": HeuristicAgent(config=HeuristicConfig(queue_front_weight=2.0)),
    "skunk_specialist": HeuristicAgent(config=HeuristicConfig(species_weights={"skunk": 2.0})),
    "noisy": HeuristicAgent(config=HeuristicConfig(noise_epsilon=0.15)),
    "online": OnlineStrategies(),
}

# Track neural agents for listing
_neural_agent = None
_neural_name = None
_neural_iter = None
_neural_agents_extra: list[tuple] = []


def _initialize_neural_agents():
    """Initialize neural agents on module load."""
    global _neural_agent, _neural_name, _neural_iter, _neural_agents_extra

    # Load main neural agent
    _neural_agent, _neural_name, _neural_iter = _load_neural_agent()
    if _neural_agent is not None:
        AI_AGENTS[_neural_name] = _neural_agent
        # Keep "neural" as alias for backwards compatibility
        AI_AGENTS["neural"] = _neural_agent

    # Load additional neural agents
    _extra_checkpoints = [
        ("checkpoints/v4/iter_949.pt", 949),
        ("checkpoints/v4/iter_600_final.pt", 600),
    ]
    for ckpt_path, _expected_iter in _extra_checkpoints:
        if Path(ckpt_path).exists():
            agent, name, iteration = _load_neural_agent(ckpt_path)
            if agent is not None and name not in AI_AGENTS:
                AI_AGENTS[name] = agent
                _neural_agents_extra.append((agent, name, iteration))

    # Load tablebase-enhanced agent
    _load_tablebase_agent()


def _load_tablebase_agent():
    """Load tablebase-enhanced neural agent if available."""
    try:
        from _02_agents.tablebase import EndgameTablebase, TablebaseAgent
    except ImportError:
        return

    tablebase_path = Path("data/endgame_4card_final.tb")
    if not tablebase_path.exists():
        return

    try:
        tablebase = EndgameTablebase.load(tablebase_path)
        logger.info(f"Loaded tablebase with {len(tablebase.positions)} positions")
        # Find the iter_949 neural agent to wrap with tablebase
        iter949_agent = AI_AGENTS.get("ppo_iter949")
        if iter949_agent is not None:
            tablebase_agent = TablebaseAgent(
                tablebase=tablebase,
                fallback_agent=iter949_agent,
                solve_on_miss=True,
            )
            AI_AGENTS["ppo_iter949_tablebase"] = tablebase_agent
            logger.info("Created tablebase-enhanced iter_949 agent: ppo_iter949_tablebase")
    except Exception as e:
        logger.warning(f"Failed to load tablebase: {e}")


def get_neural_agent_info() -> tuple:
    """Get info about the primary neural agent."""
    return _neural_agent, _neural_name, _neural_iter


def get_neural_agents_extra() -> list[tuple]:
    """Get info about additional neural agents."""
    return _neural_agents_extra


# Initialize on module import
_initialize_neural_agents()
