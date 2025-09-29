"""Agent implementations and utilities."""

from .base import Agent, AgentFn, ensure_legal
from .diego import DiegoAgent
from .evaluation import HeuristicFn, best_action, evaluate_action, material_advantage
from .first import FirstLegalAgent
from .greedy import GreedyAgent
from .random_agent import RandomAgent
from .self_play_rl import ExplorationConfig, SelfPlayRLAgent

__all__ = [
    "Agent",
    "AgentFn",
    "ensure_legal",
    "FirstLegalAgent",
    "RandomAgent",
    "GreedyAgent",
    "DiegoAgent",
    "ExplorationConfig",
    "SelfPlayRLAgent",
    "HeuristicFn",
    "best_action",
    "evaluate_action",
    "material_advantage",
]
