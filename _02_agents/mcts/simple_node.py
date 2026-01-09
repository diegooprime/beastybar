"""Simple MCTS Node implementation for AlphaZero-style tree search.

This is a simpler, standalone MCTSNode implementation that works with
explicit policy priors (dict of Action -> float) rather than neural network
tensor outputs. It's useful for understanding MCTS mechanics and for use
with hand-coded policy functions.

For neural network-guided MCTS, use the MCTSNode from search.py instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from _01_simulator import actions, engine, state

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class SimpleMCTSNode:
    """Node in the Monte Carlo Tree Search tree.

    This implementation follows AlphaZero-style MCTS with neural network priors.
    Each node represents a game state and tracks visit counts, value estimates,
    and prior probabilities from the policy network.

    Attributes:
        state: The game state this node represents.
        parent: Parent node in the tree (None for root).
        children: Dict mapping actions to child nodes.
        visit_count: Number of times this node has been visited.
        value_sum: Sum of all values backed up through this node.
        prior: Prior probability from policy network (0.0 for root).
    """

    state: state.State
    parent: SimpleMCTSNode | None = None
    children: dict[actions.Action, SimpleMCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    @property
    def is_expanded(self) -> bool:
        """Check if this node has been expanded with children.

        Returns:
            True if children have been created for this node.
        """
        return len(self.children) > 0

    @property
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state.

        Returns:
            True if the game has ended at this state.
        """
        return engine.is_terminal(self.state)

    @property
    def value(self) -> float:
        """Average value of this node.

        Returns:
            Mean value over all visits, or 0 if never visited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float = 1.414) -> float:
        """Calculate Upper Confidence Bound score with prior.

        The UCB score balances exploitation (current value estimate) with
        exploration (visit count and prior probability). Higher c_puct
        encourages more exploration.

        Formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
            Q(s,a) = value estimate for action a in state s
            P(s,a) = prior probability from policy network
            N(s) = visit count of parent node
            N(s,a) = visit count of this child node

        Args:
            parent_visits: Visit count of the parent node.
            c_puct: Exploration constant controlling exploration vs exploitation.

        Returns:
            UCB score for node selection.
        """
        if self.visit_count == 0:
            # Unvisited nodes get maximum exploration bonus
            exploration = c_puct * self.prior * math.sqrt(parent_visits)
            return self.value + exploration

        # Balance exploitation and exploration
        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return exploitation + exploration

    def expand(self, policy_priors: Mapping[actions.Action, float]) -> None:
        """Create child nodes for all legal actions with policy priors.

        This method should be called once per node to create children for all
        legal actions in the current state. Each child is initialized with a
        prior probability from the policy network.

        Args:
            policy_priors: Dict mapping legal actions to their prior probabilities
                from the policy network. Should sum to approximately 1.0.

        Raises:
            ValueError: If node is already expanded or if actions don't match state.
        """
        if self.is_expanded:
            raise ValueError("Node is already expanded")

        if self.is_terminal:
            # Terminal nodes have no children
            return

        # Get legal actions for the current state
        legal = list(engine.legal_actions(self.state, self.state.active_player))

        # Validate that policy priors match legal actions
        if set(policy_priors.keys()) != set(legal):
            raise ValueError(
                f"Policy priors actions {set(policy_priors.keys())} don't match legal actions {set(legal)}"
            )

        # Create child nodes with priors
        for action, prior in policy_priors.items():
            # Compute next state by applying action
            next_state = engine.step(self.state, action)

            # Create child node
            child = SimpleMCTSNode(
                state=next_state,
                parent=self,
                prior=prior,
            )
            self.children[action] = child

    def select_child(self, c_puct: float = 1.414) -> tuple[actions.Action, SimpleMCTSNode]:
        """Select the best child node using UCB scores.

        Args:
            c_puct: Exploration constant for UCB calculation.

        Returns:
            Tuple of (action, child_node) with highest UCB score.

        Raises:
            ValueError: If node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select child from unexpanded node")

        best_score = float("-inf")
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_action is None or best_child is None:
            raise RuntimeError("Failed to select child node")

        return best_action, best_child

    def backup(self, value: float) -> None:
        """Propagate value estimate up the tree to the root.

        This method updates visit counts and value sums for this node and all
        ancestors. The value is negated at each level to account for alternating
        players (zero-sum game).

        Args:
            value: Value estimate to back up (from perspective of current player).
        """
        node: SimpleMCTSNode | None = self
        current_value = value

        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value

            # Move to parent and negate value for opponent
            node = node.parent
            if node is not None:
                # Flip value for two-player zero-sum game
                current_value = -current_value

    def best_child_by_visits(self) -> tuple[actions.Action, SimpleMCTSNode] | None:
        """Get the child with the most visits (for final move selection).

        Returns:
            Tuple of (action, child_node) with most visits, or None if no children.
        """
        if not self.children:
            return None

        best_action = None
        best_child = None
        max_visits = -1

        for action, child in self.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_action = action
                best_child = child

        if best_action is None or best_child is None:
            return None

        return best_action, best_child


__all__ = ["SimpleMCTSNode"]
