"""Neural Monte Carlo Tree Search with UCB selection and Dirichlet noise exploration.

This module implements MCTS with neural network guidance following the AlphaZero approach:
- Policy network provides action priors for PUCT exploration
- Value network provides position evaluation for backup
- Dirichlet noise at root for exploration
- Action masking for legal move enforcement
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from _01_simulator import action_space, engine, observations, state

if TYPE_CHECKING:
    from _02_agents.neural.network import BeastyBarNetwork


@dataclass
class MCTSNode:
    """Node in the MCTS tree.

    Each node represents a game state and tracks:
    - Visit statistics for UCB calculation
    - Action priors from the policy network
    - Child nodes for explored actions
    """

    state: state.State
    parent: MCTSNode | None = None
    action_taken: int | None = None  # Action index that led to this node

    # Statistics
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 1.0  # Policy prior probability

    # Children: mapping from action index to child node
    children: dict[int, MCTSNode] = field(default_factory=dict)

    # Cached policy and value from network evaluation
    _policy_priors: dict[int, float] | None = None
    _value: float | None = None

    @property
    def is_expanded(self) -> bool:
        """Check if node has been evaluated by the network."""
        return self._policy_priors is not None

    @property
    def mean_value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, child_action: int, c_puct: float, parent_visits: int) -> float:
        """Calculate PUCT score for a child action.

        PUCT (Predictor + Upper Confidence Bound) formula:
            Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
            Q(s,a) = mean value of child node
            P(s,a) = policy prior probability
            N(s) = parent visit count
            N(s,a) = child visit count
            c_puct = exploration constant

        Args:
            child_action: Action index to score
            c_puct: Exploration constant
            parent_visits: Visit count of parent node

        Returns:
            PUCT score for selecting this action
        """
        if self._policy_priors is None:
            return float("inf")  # Prioritize unexpanded nodes

        prior = self._policy_priors.get(child_action, 0.0)

        # Get child statistics
        child = self.children.get(child_action)
        if child is None:
            # Unexpanded action
            q_value = 0.0
            n_visits = 0
        else:
            q_value = child.mean_value
            n_visits = child.visit_count

        # PUCT formula
        u_value = c_puct * prior * math.sqrt(parent_visits) / (1 + n_visits)
        return q_value + u_value


class MCTS:
    """Neural Monte Carlo Tree Search implementation.

    Uses a neural network to guide tree search with:
    - Policy priors for action selection (PUCT)
    - Value estimates for position evaluation
    - Dirichlet noise at root for exploration
    - Temperature-based action sampling

    Key hyperparameters:
    - num_simulations: Number of MCTS iterations per move
    - c_puct: Exploration constant for PUCT formula
    - dirichlet_alpha: Alpha parameter for Dirichlet noise
    - dirichlet_epsilon: Weight of Dirichlet noise vs policy
    """

    def __init__(
        self,
        network: BeastyBarNetwork,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize MCTS with neural network.

        Args:
            network: Policy-value network for evaluation
            num_simulations: Number of MCTS iterations per search
            c_puct: Exploration constant (higher = more exploration)
            dirichlet_alpha: Dirichlet noise concentration parameter
            dirichlet_epsilon: Mixing weight for Dirichlet noise at root
            device: Device to run network on (auto-detected if None)
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        # Set device
        if device is None:
            try:
                self.device = next(network.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Ensure network is on correct device
        self.network = self.network.to(self.device)
        self.network.eval()  # Set to evaluation mode

    def search(
        self,
        game_state: state.State,
        perspective: int,
        *,
        temperature: float = 1.0,
        add_root_noise: bool = True,
    ) -> dict[int, float]:
        """Run MCTS and return visit count distribution over actions.

        Args:
            game_state: Current game state to search from
            perspective: Player index for observation encoding
            temperature: Temperature for visit count sampling (used by caller)
            add_root_noise: Whether to add Dirichlet noise to root policy

        Returns:
            Dictionary mapping action indices to visit count probabilities
        """
        # Create root node
        root = MCTSNode(state=game_state)

        # Expand root with network evaluation
        self._expand(root, perspective)

        # Add Dirichlet noise to root for exploration
        if add_root_noise and root._policy_priors:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, perspective)

        # Convert visit counts to probability distribution
        return self._visit_count_distribution(root)

    def _simulate(self, root: MCTSNode, perspective: int) -> None:
        """Run one MCTS iteration: select, expand, backup.

        Args:
            root: Root node to search from
            perspective: Player index for observation encoding
        """
        # SELECT: Walk down tree using PUCT until unexpanded node or terminal
        node = root
        search_path = [node]

        while node.is_expanded and not engine.is_terminal(node.state) and node._policy_priors:
            # Select best action using PUCT
            action_idx = self._select_action(node)

            # Create child if doesn't exist
            if action_idx not in node.children:
                # Execute action and create child node
                action = action_space.index_to_action(action_idx)
                next_state = engine.step(node.state, action)
                child = MCTSNode(
                    state=next_state,
                    parent=node,
                    action_taken=action_idx,
                )
                node.children[action_idx] = child

            node = node.children[action_idx]
            search_path.append(node)

        # EXPAND: Evaluate position with network (unless terminal)
        if not engine.is_terminal(node.state):
            self._expand(node, perspective)
            value = node._value if node._value is not None else 0.0
        else:
            # Terminal state: use game outcome
            value = self._evaluate_terminal(node.state, perspective)

        # BACKUP: Propagate value up the search path
        self._backup(search_path, value, perspective)

    def _select_action(self, node: MCTSNode) -> int:
        """Select best action from node using PUCT.

        Args:
            node: Node to select action from

        Returns:
            Action index with highest PUCT score
        """
        if not node._policy_priors:
            raise RuntimeError("Cannot select action from unexpanded node")

        best_action = -1
        best_score = float("-inf")

        for action_idx in node._policy_priors:
            score = node.ucb_score(action_idx, self.c_puct, node.visit_count)
            if score > best_score:
                best_score = score
                best_action = action_idx

        return best_action

    def _expand(self, node: MCTSNode, perspective: int) -> None:
        """Expand node by evaluating with neural network.

        Args:
            node: Node to expand
            perspective: Player index for observation encoding
        """
        # Convert state to observation tensor
        obs_tensor = observations.state_to_tensor(node.state, perspective)
        obs_torch = torch.from_numpy(obs_tensor).unsqueeze(0).to(self.device)

        # Get legal action mask
        action_mask = action_space.legal_action_mask_tensor(node.state, perspective)
        mask_torch = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)

        # Evaluate with network
        with torch.no_grad():
            policy_logits, value = self.network(obs_torch, mask_torch)

        # Apply mask to policy and compute priors
        policy_priors = self._compute_policy_priors(policy_logits.squeeze(0), mask_torch.squeeze(0))

        # Store results
        node._policy_priors = policy_priors
        node._value = float(value.squeeze().item())

    def _compute_policy_priors(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[int, float]:
        """Compute policy prior probabilities from logits with masking.

        Args:
            logits: Raw policy logits (ACTION_DIM,)
            mask: Legal action mask (ACTION_DIM,)

        Returns:
            Dictionary mapping legal action indices to prior probabilities.
            Empty dict if no legal actions (terminal-like state).
        """
        # Get legal action indices
        legal_indices = torch.where(mask > 0)[0]

        # Handle edge case: no legal actions
        if len(legal_indices) == 0:
            return {}

        # Apply mask: set illegal actions to very negative value
        masked_logits = logits.masked_fill(mask <= 0, float("-inf"))

        # Compute softmax probabilities
        probs = torch.softmax(masked_logits, dim=-1)

        # Handle potential NaN from softmax (defensive)
        if torch.isnan(probs).any():
            # Fallback: uniform distribution over legal actions
            uniform_prob = 1.0 / len(legal_indices)
            return {int(idx.item()): uniform_prob for idx in legal_indices}

        # Convert to dictionary (only legal actions)
        policy_priors = {int(idx.item()): float(probs[idx].item()) for idx in legal_indices}

        return policy_priors

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """Add Dirichlet noise to root policy for exploration.

        Mixes network policy with Dirichlet noise:
            P'(a) = (1 - ε) * P(a) + ε * η_a
        where η ~ Dir(alpha)

        Args:
            node: Root node to add noise to
        """
        if not node._policy_priors:
            return

        # Generate Dirichlet noise for all legal actions
        legal_actions = list(node._policy_priors.keys())
        num_actions = len(legal_actions)

        if num_actions == 0:
            return

        # Sample Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)

        # Mix policy with noise
        epsilon = self.dirichlet_epsilon
        for i, action_idx in enumerate(legal_actions):
            prior = node._policy_priors[action_idx]
            node._policy_priors[action_idx] = (1 - epsilon) * prior + epsilon * noise[i]

    def _backup(
        self,
        search_path: list[MCTSNode],
        value: float,
        perspective: int,
    ) -> None:
        """Backpropagate value up the search path.

        Values are stored from the perspective of the player to move at each node.
        A positive value means good for the perspective player.

        Args:
            search_path: List of nodes from root to leaf
            value: Leaf value from perspective player's viewpoint
            perspective: Player index whose perspective the value represents
        """
        for node in reversed(search_path):
            # Determine value from this node's player perspective
            # If node's active player matches perspective, use value as-is
            # Otherwise, negate (opponent's gain is our loss)
            node_value = value if node.state.active_player == perspective else -value

            node.visit_count += 1
            node.total_value += node_value

    def _evaluate_terminal(self, game_state: state.State, perspective: int) -> float:
        """Evaluate terminal state from perspective.

        Args:
            game_state: Terminal game state
            perspective: Player index

        Returns:
            Value in [-1, 1] where 1 = win, -1 = loss, 0 = draw.
            Margin bonus scales within the win/loss range to preserve [-1, 1] bounds.
        """
        scores = engine.score(game_state)
        my_score = scores[perspective]
        opp_score = scores[1 - perspective]

        if my_score > opp_score:
            # Win with margin bonus (scaled to stay within [0.8, 1.0])
            margin = my_score - opp_score
            max_margin = 36  # Theoretical max
            margin_bonus = 0.2 * min(1.0, margin / max_margin)
            return 0.8 + margin_bonus  # Range: [0.8, 1.0]
        elif my_score < opp_score:
            # Loss with margin penalty (scaled to stay within [-1.0, -0.8])
            margin = opp_score - my_score
            max_margin = 36
            margin_penalty = 0.2 * min(1.0, margin / max_margin)
            return -0.8 - margin_penalty  # Range: [-1.0, -0.8]
        else:
            # Draw
            return 0.0

    def _visit_count_distribution(self, root: MCTSNode) -> dict[int, float]:
        """Convert visit counts to probability distribution.

        Args:
            root: Root node with visit statistics

        Returns:
            Dictionary mapping action indices to normalized visit probabilities
        """
        if root.visit_count == 0:
            # No visits, return uniform over legal actions
            if root._policy_priors:
                num_legal = len(root._policy_priors)
                return dict.fromkeys(root._policy_priors, 1.0 / num_legal)
            return {}

        # Collect visit counts
        visit_counts = {}
        for action_idx, child in root.children.items():
            visit_counts[action_idx] = child.visit_count

        # Normalize
        total_visits = sum(visit_counts.values())
        if total_visits == 0:
            return {}

        return {action_idx: count / total_visits for action_idx, count in visit_counts.items()}


__all__ = ["MCTS", "MCTSNode"]
