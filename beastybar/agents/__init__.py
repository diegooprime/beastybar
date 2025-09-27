"""Agent implementations and utilities."""

from .base import Agent, AgentFn, ensure_legal
from .baselines import FirstLegalAgent, RandomAgent
from .evaluation import HeuristicFn, best_action, evaluate_action, material_advantage
from .greedy import GreedyAgent

__all__ = [
    "Agent",
    "AgentFn",
    "ensure_legal",
    "FirstLegalAgent",
    "RandomAgent",
    "GreedyAgent",
    "HeuristicFn",
    "best_action",
    "evaluate_action",
    "material_advantage",
]
