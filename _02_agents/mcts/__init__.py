"""Neural Monte Carlo Tree Search implementation for Beasty Bar.

This package provides a neural network-guided MCTS agent that uses
a policy-value network to guide tree search and evaluate positions.

Classes:
    MCTSNode: Tree node with UCB scoring and backpropagation (from search.py).
    SimpleMCTSNode: Simpler node implementation for Action-based MCTS (from simple_node.py).
    MCTS: AlphaZero-style tree search with neural network integration.
    BatchMCTS: Optimized batched MCTS with virtual loss for parallel search.
    MCTSAgent: Agent using neural MCTS for action selection.

Example (Neural MCTS):
    ```python
    from _02_agents.mcts import MCTSAgent
    from _02_agents.neural.network import BeastyBarNetwork
    from _01_simulator import state

    # Create network and agent
    network = BeastyBarNetwork()
    agent = MCTSAgent(network, num_simulations=400)

    # Use agent to select actions
    game_state = state.initial_state(seed=42)
    legal = list(engine.legal_actions(game_state, 0))
    action = agent.select_action(game_state, legal)
    ```

Example (Batched MCTS):
    ```python
    from _02_agents.mcts import BatchMCTS
    from _02_agents.neural.network import BeastyBarNetwork
    from _01_simulator import state

    # Create network and batched MCTS
    network = BeastyBarNetwork()
    batch_mcts = BatchMCTS(
        network,
        num_simulations=100,
        virtual_loss=3.0,
        batch_size=8,
    )

    # Search multiple states in parallel
    states = [state.initial_state(seed=i) for i in range(8)]
    distributions = batch_mcts.search_batch(states, perspective=0)
    ```

Example (Simple MCTS):
    ```python
    from _02_agents.mcts import SimpleMCTSNode
    from _01_simulator import state, engine

    # Create root node
    game_state = state.initial_state(seed=42)
    root = SimpleMCTSNode(state=game_state)

    # Expand with policy priors
    legal = list(engine.legal_actions(game_state, 0))
    priors = {action: 1.0 / len(legal) for action in legal}
    root.expand(priors)

    # Select best child
    action, child = root.select_child(c_puct=1.414)
    ```
"""

from __future__ import annotations

from .adaptive_search import (
    STRENGTH_CONFIGS,
    AdaptiveMCTS,
    AdaptiveMCTSAgent,
    SearchStats,
    StrengthConfig,
    StrengthLevel,
)
from .agent import MCTSAgent
from .batch_mcts import BatchMCTS
from .search import MCTS, MCTSNode
from .simple_node import SimpleMCTSNode

__all__ = [
    "MCTS",
    "STRENGTH_CONFIGS",
    "AdaptiveMCTS",
    "AdaptiveMCTSAgent",
    "BatchMCTS",
    "MCTSAgent",
    "MCTSNode",
    "SearchStats",
    "SimpleMCTSNode",
    "StrengthConfig",
    "StrengthLevel",
]
