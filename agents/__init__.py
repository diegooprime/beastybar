"""Agent implementations and utilities."""

from .base import Agent, AgentFn, ensure_legal
from .baselines import FirstLegalAgent, RandomAgent
from .diego import DiegoAgent
from .frontrunner import FrontRunnerAgent
from .evaluation import HeuristicFn, best_action, evaluate_action, material_advantage
from .greedy import GreedyAgent
from .killer import KillerAgent

__all__ = [
    "Agent",
    "AgentFn",
    "ensure_legal",
    "FirstLegalAgent",
    "RandomAgent",
    "GreedyAgent",
    "DiegoAgent",
    "FrontRunnerAgent",
    "KillerAgent",
    "HeuristicFn",
    "best_action",
    "evaluate_action",
    "material_advantage",
]
