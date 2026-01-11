"""Agent implementations for Beasty Bar."""

from __future__ import annotations

from .base import Agent, AgentFn
from .heuristic import HeuristicAgent, HeuristicConfig, MaterialEvaluator, create_heuristic_variants
from .outcome_heuristic import (
    DISTILLED_WEIGHTS,
    DistilledOutcomeHeuristic,
    OutcomeHeuristic,
    OutcomeHeuristicV2,
    OutcomeWeights,
    extract_weights_from_ppo,
)
from .mcts import MCTSAgent, MCTSNode, SimpleMCTSNode
from .random_agent import RandomAgent

__all__ = [
    "Agent",
    "AgentFn",
    "DISTILLED_WEIGHTS",
    "DistilledOutcomeHeuristic",
    "HeuristicAgent",
    "HeuristicConfig",
    "MaterialEvaluator",
    "MCTSAgent",
    "MCTSNode",
    "OutcomeHeuristic",
    "OutcomeHeuristicV2",
    "OutcomeWeights",
    "RandomAgent",
    "SimpleMCTSNode",
    "create_heuristic_variants",
    "extract_weights_from_ppo",
]
