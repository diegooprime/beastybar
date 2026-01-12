"""Agent implementations for Beasty Bar."""

from __future__ import annotations

from .base import Agent, AgentFn
from .heuristic import HeuristicAgent, HeuristicConfig, MaterialEvaluator, create_heuristic_variants
from .mcts import MCTSAgent, MCTSNode, SimpleMCTSNode
from .openings import OpeningBook, OpeningBookAgent, OpeningBookGenerator, OpeningEntry
from .outcome_heuristic import (
    DISTILLED_WEIGHTS,
    DistilledOutcomeHeuristic,
    OutcomeHeuristic,
    OutcomeHeuristicV2,
    OutcomeWeights,
    extract_weights_from_ppo,
)
from .random_agent import RandomAgent

__all__ = [
    "DISTILLED_WEIGHTS",
    "Agent",
    "AgentFn",
    "DistilledOutcomeHeuristic",
    "HeuristicAgent",
    "HeuristicConfig",
    "MCTSAgent",
    "MCTSNode",
    "MaterialEvaluator",
    "OpeningBook",
    "OpeningBookAgent",
    "OpeningBookGenerator",
    "OpeningEntry",
    "OutcomeHeuristic",
    "OutcomeHeuristicV2",
    "OutcomeWeights",
    "RandomAgent",
    "SimpleMCTSNode",
    "create_heuristic_variants",
    "extract_weights_from_ppo",
]
