"""Adaptive MCTS with configurable simulation budgets and early stopping.

This module implements AdaptiveMCTS for inference-time compute scaling:
- Configurable minimum and maximum simulation budgets
- Adaptive early stopping based on action confidence
- Time budget support for real-time play
- Strength level presets (100/400/1600/6400 simulations)

Key Features:
- Action confidence-based stopping: terminates early when top action is clear
- Time-aware search: respects wall-clock time budgets
- Uncertainty threshold: continues searching uncertain positions
- Strength scaling: more simulations = stronger play (+ELO per 4x compute)

Typical Usage:
    ```python
    from _02_agents.mcts import AdaptiveMCTS
    from _02_agents.neural.network import BeastyBarNetwork

    network = BeastyBarNetwork()

    # Time-constrained play (e.g., 500ms per move)
    mcts = AdaptiveMCTS(network, strength="tournament")
    policy = mcts.search(state, perspective=0, time_budget_ms=500)

    # Fixed simulation budget
    mcts = AdaptiveMCTS(network, strength="superhuman")
    policy = mcts.search(state, perspective=0)
    ```

Strength Levels:
    - "fast": 100 simulations (baseline, quick inference)
    - "tournament": 400 simulations (+50 ELO vs fast)
    - "analysis": 1600 simulations (+100 ELO vs fast)
    - "superhuman": 6400 simulations (+150 ELO vs fast)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch

from _01_simulator import action_space, actions, engine, observations, state
from _02_agents.mcts.search import MCTSNode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _02_agents.neural.network import BeastyBarNetwork


class StrengthLevel(Enum):
    """Predefined strength levels with simulation budgets.

    Each level represents a different trade-off between speed and strength.
    Higher simulation counts yield stronger play at the cost of more compute.
    """

    FAST = "fast"  # 100 simulations - baseline
    TOURNAMENT = "tournament"  # 400 simulations - +50 ELO
    ANALYSIS = "analysis"  # 1600 simulations - +100 ELO
    SUPERHUMAN = "superhuman"  # 6400 simulations - +150 ELO


@dataclass(frozen=True)
class StrengthConfig:
    """Configuration for a strength level.

    Attributes:
        min_simulations: Minimum simulations before early stopping check
        max_simulations: Maximum simulations (hard cap)
        confidence_threshold: Minimum top action probability for early stop
        uncertainty_threshold: Maximum policy entropy for early stop
    """

    min_simulations: int
    max_simulations: int
    confidence_threshold: float
    uncertainty_threshold: float


# Predefined strength configurations
STRENGTH_CONFIGS: dict[StrengthLevel, StrengthConfig] = {
    StrengthLevel.FAST: StrengthConfig(
        min_simulations=50,
        max_simulations=100,
        confidence_threshold=0.8,  # Stop if top action has 80%+ visits
        uncertainty_threshold=0.5,  # Stop if normalized entropy < 0.5
    ),
    StrengthLevel.TOURNAMENT: StrengthConfig(
        min_simulations=100,
        max_simulations=400,
        confidence_threshold=0.75,
        uncertainty_threshold=0.4,
    ),
    StrengthLevel.ANALYSIS: StrengthConfig(
        min_simulations=400,
        max_simulations=1600,
        confidence_threshold=0.7,
        uncertainty_threshold=0.3,
    ),
    StrengthLevel.SUPERHUMAN: StrengthConfig(
        min_simulations=1600,
        max_simulations=6400,
        confidence_threshold=0.65,
        uncertainty_threshold=0.2,
    ),
}


@dataclass
class SearchStats:
    """Statistics from an adaptive MCTS search.

    Useful for analysis and debugging search behavior.

    Attributes:
        simulations_used: Total simulations performed
        time_elapsed_ms: Wall-clock time used
        stopped_by: Reason for stopping ("budget", "confidence", "uncertainty", "time")
        top_action_confidence: Final confidence in top action
        policy_entropy: Final normalized entropy of policy
    """

    simulations_used: int
    time_elapsed_ms: float
    stopped_by: str
    top_action_confidence: float
    policy_entropy: float


class AdaptiveMCTS:
    """MCTS with adaptive simulation budget and early stopping.

    Implements inference-time compute scaling by dynamically adjusting
    the number of MCTS simulations based on:
    1. Action confidence: stop early when top action is clearly best
    2. Time budget: respect wall-clock time constraints
    3. Uncertainty: continue searching when position is unclear

    This enables stronger play by investing more compute in difficult
    positions while saving time on clear decisions.

    Key features:
    - Strength level presets for easy configuration
    - Time budget support for real-time play
    - Early stopping based on visit distribution confidence
    - Statistics tracking for analysis

    Typical ELO scaling:
    - 100 sims: baseline
    - 400 sims: +50 ELO
    - 1600 sims: +100 ELO
    - 6400 sims: +150 ELO
    """

    def __init__(
        self,
        network: BeastyBarNetwork,
        *,
        strength: str | StrengthLevel = StrengthLevel.TOURNAMENT,
        min_simulations: int | None = None,
        max_simulations: int | None = None,
        confidence_threshold: float | None = None,
        uncertainty_threshold: float | None = None,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize AdaptiveMCTS with neural network.

        Args:
            network: Policy-value network for evaluation
            strength: Strength level preset ("fast", "tournament", "analysis", "superhuman")
            min_simulations: Override minimum simulations (from strength if None)
            max_simulations: Override maximum simulations (from strength if None)
            confidence_threshold: Override confidence threshold for early stop
            uncertainty_threshold: Override uncertainty threshold for early stop
            c_puct: Exploration constant for PUCT formula (higher = more exploration)
            dirichlet_alpha: Dirichlet noise concentration parameter
            dirichlet_epsilon: Mixing weight for Dirichlet noise at root
            device: Device to run network on (auto-detected if None)
        """
        self.network = network
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        # Resolve strength level
        if isinstance(strength, str):
            strength = StrengthLevel(strength.lower())
        config = STRENGTH_CONFIGS[strength]

        # Apply configuration with overrides
        self.min_simulations = min_simulations if min_simulations is not None else config.min_simulations
        self.max_simulations = max_simulations if max_simulations is not None else config.max_simulations
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else config.confidence_threshold
        )
        self.uncertainty_threshold = (
            uncertainty_threshold if uncertainty_threshold is not None else config.uncertainty_threshold
        )

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
        self.network.eval()

        # Search statistics (updated after each search)
        self._last_stats: SearchStats | None = None

    @property
    def last_stats(self) -> SearchStats | None:
        """Get statistics from the most recent search."""
        return self._last_stats

    def search(
        self,
        game_state: state.State,
        perspective: int,
        *,
        temperature: float = 1.0,
        add_root_noise: bool = True,
        time_budget_ms: float | None = None,
        min_simulations: int | None = None,
        max_simulations: int | None = None,
    ) -> dict[int, float]:
        """Run adaptive MCTS and return visit count distribution.

        The search continues until one of the following conditions is met:
        1. Maximum simulations reached
        2. Time budget exhausted (if specified)
        3. Top action confidence exceeds threshold (after min_simulations)
        4. Policy uncertainty falls below threshold (after min_simulations)

        Args:
            game_state: Current game state to search from
            perspective: Player index for observation encoding
            temperature: Temperature for visit count sampling (used by caller)
            add_root_noise: Whether to add Dirichlet noise to root policy
            time_budget_ms: Wall-clock time budget in milliseconds (None = unlimited)
            min_simulations: Override instance min_simulations for this search
            max_simulations: Override instance max_simulations for this search

        Returns:
            Dictionary mapping action indices to visit count probabilities
        """
        # Apply per-search overrides
        min_sims = min_simulations if min_simulations is not None else self.min_simulations
        max_sims = max_simulations if max_simulations is not None else self.max_simulations

        # Start timing
        start_time = time.perf_counter()

        # Create root node
        root = MCTSNode(state=game_state)

        # Expand root with network evaluation
        self._expand(root, perspective)

        # Add Dirichlet noise to root for exploration
        if add_root_noise and root._policy_priors:
            self._add_dirichlet_noise(root)

        # Track stopping reason
        stopped_by = "budget"
        simulations_used = 0

        # Run simulations with adaptive stopping
        while simulations_used < max_sims:
            # Check time budget
            if time_budget_ms is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if elapsed_ms >= time_budget_ms:
                    stopped_by = "time"
                    break

            # Run one simulation
            self._simulate(root, perspective)
            simulations_used += 1

            # After minimum simulations, check for early stopping
            if simulations_used >= min_sims:
                confidence = self._compute_action_confidence(root)
                entropy = self._compute_policy_entropy(root)

                # Stop if top action is clear winner
                if confidence >= self.confidence_threshold:
                    stopped_by = "confidence"
                    break

                # Stop if policy is sufficiently certain
                if entropy <= self.uncertainty_threshold:
                    stopped_by = "uncertainty"
                    break

        # Compute final statistics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        final_confidence = self._compute_action_confidence(root)
        final_entropy = self._compute_policy_entropy(root)

        self._last_stats = SearchStats(
            simulations_used=simulations_used,
            time_elapsed_ms=elapsed_ms,
            stopped_by=stopped_by,
            top_action_confidence=final_confidence,
            policy_entropy=final_entropy,
        )

        # Convert visit counts to probability distribution
        return self._visit_count_distribution(root)

    def search_with_stats(
        self,
        game_state: state.State,
        perspective: int,
        **kwargs: object,
    ) -> tuple[dict[int, float], SearchStats]:
        """Run search and return both policy and statistics.

        Convenience method that returns statistics alongside the policy.

        Args:
            game_state: Current game state to search from
            perspective: Player index for observation encoding
            **kwargs: Additional arguments passed to search()

        Returns:
            Tuple of (visit_distribution, search_stats)
        """
        policy = self.search(game_state, perspective, **kwargs)
        assert self._last_stats is not None  # Set by search()
        return policy, self._last_stats

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
            uniform_prob = 1.0 / len(legal_indices)
            return {int(idx.item()): uniform_prob for idx in legal_indices}

        # Convert to dictionary (only legal actions)
        return {int(idx.item()): float(probs[idx].item()) for idx in legal_indices}

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """Add Dirichlet noise to root policy for exploration.

        Mixes network policy with Dirichlet noise:
            P'(a) = (1 - epsilon) * P(a) + epsilon * noise_a
        where noise ~ Dir(alpha)

        Args:
            node: Root node to add noise to
        """
        if not node._policy_priors:
            return

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

        Values are stored from the perspective of the player to move.

        Args:
            search_path: List of nodes from root to leaf
            value: Leaf value from perspective player's viewpoint
            perspective: Player index whose perspective the value represents
        """
        for node in reversed(search_path):
            # Determine value from this node's player perspective
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
        """
        scores = engine.score(game_state)
        my_score = scores[perspective]
        opp_score = scores[1 - perspective]

        if my_score > opp_score:
            margin = my_score - opp_score
            max_margin = 36
            margin_bonus = 0.2 * min(1.0, margin / max_margin)
            return 0.8 + margin_bonus
        elif my_score < opp_score:
            margin = opp_score - my_score
            max_margin = 36
            margin_penalty = 0.2 * min(1.0, margin / max_margin)
            return -0.8 - margin_penalty
        else:
            return 0.0

    def _visit_count_distribution(self, root: MCTSNode) -> dict[int, float]:
        """Convert visit counts to probability distribution.

        Args:
            root: Root node with visit statistics

        Returns:
            Dictionary mapping action indices to normalized visit probabilities
        """
        if root.visit_count == 0:
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

    def _compute_action_confidence(self, root: MCTSNode) -> float:
        """Compute confidence in the top action.

        Confidence is defined as the visit share of the most-visited action.
        Higher confidence indicates a clearer best move.

        Args:
            root: Root node with visit statistics

        Returns:
            Confidence in [0, 1] where 1 = all visits to one action
        """
        if root.visit_count == 0 or not root.children:
            return 0.0

        # Get visit counts
        visit_counts = [child.visit_count for child in root.children.values()]
        total_visits = sum(visit_counts)

        if total_visits == 0:
            return 0.0

        # Confidence = max visit share
        max_visits = max(visit_counts)
        return max_visits / total_visits

    def _compute_policy_entropy(self, root: MCTSNode) -> float:
        """Compute normalized entropy of the visit distribution.

        Low entropy indicates confident policy (few actions with high visits).
        High entropy indicates uncertain policy (many actions with similar visits).

        The entropy is normalized to [0, 1] by dividing by max entropy.

        Args:
            root: Root node with visit statistics

        Returns:
            Normalized entropy in [0, 1] where 0 = deterministic, 1 = uniform
        """
        if root.visit_count == 0 or not root.children:
            return 1.0  # Maximum uncertainty

        # Get visit distribution
        visit_counts = np.array([child.visit_count for child in root.children.values()], dtype=np.float64)
        total_visits = visit_counts.sum()

        if total_visits == 0:
            return 1.0

        # Compute probabilities
        probs = visit_counts / total_visits

        # Compute entropy: H = -sum(p * log(p))
        # Handle zeros by masking
        nonzero_mask = probs > 0
        entropy = -np.sum(probs[nonzero_mask] * np.log(probs[nonzero_mask]))

        # Normalize by maximum entropy (uniform distribution)
        num_actions = len(probs)
        if num_actions <= 1:
            return 0.0

        max_entropy = np.log(num_actions)
        normalized_entropy = entropy / max_entropy

        return float(normalized_entropy)

    def set_strength(self, strength: str | StrengthLevel) -> None:
        """Update strength level configuration.

        Args:
            strength: New strength level preset
        """
        if isinstance(strength, str):
            strength = StrengthLevel(strength.lower())

        config = STRENGTH_CONFIGS[strength]
        self.min_simulations = config.min_simulations
        self.max_simulations = config.max_simulations
        self.confidence_threshold = config.confidence_threshold
        self.uncertainty_threshold = config.uncertainty_threshold


class AdaptiveMCTSAgent:
    """Agent using AdaptiveMCTS for action selection.

    Wrapper around AdaptiveMCTS that provides the standard Agent interface.
    Supports strength level configuration and time-budgeted play.
    """

    def __init__(
        self,
        network: BeastyBarNetwork,
        *,
        strength: str | StrengthLevel = StrengthLevel.TOURNAMENT,
        time_budget_ms: float | None = None,
        temperature: float = 1.0,
        device: torch.device | str | None = None,
        name: str | None = None,
        **mcts_kwargs: object,
    ) -> None:
        """Initialize AdaptiveMCTS agent.

        Args:
            network: Policy-value network for evaluation
            strength: Strength level preset
            time_budget_ms: Time budget per move (None = use simulation budget)
            temperature: Temperature for action sampling
            device: Device to run network on
            name: Custom agent name
            **mcts_kwargs: Additional arguments for AdaptiveMCTS
        """
        self._network = network
        self._time_budget_ms = time_budget_ms
        self._temperature = temperature
        self._name = name
        self._strength = strength if isinstance(strength, StrengthLevel) else StrengthLevel(strength.lower())

        self.mcts = AdaptiveMCTS(
            network=network,
            strength=strength,
            device=device,
            **mcts_kwargs,
        )

    @property
    def name(self) -> str:
        """Return agent name for display."""
        if self._name:
            return self._name
        time_str = f", {self._time_budget_ms}ms" if self._time_budget_ms else ""
        return f"AdaptiveMCTS({self._strength.value}{time_str})"

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select action using adaptive MCTS.

        Args:
            game_state: Current game state
            legal_actions: Available legal actions

        Returns:
            Selected action from MCTS search
        """
        # Early exit for single legal action
        if len(legal_actions) == 1:
            return legal_actions[0]

        perspective = game_state.active_player

        # Run adaptive MCTS search
        visit_distribution = self.mcts.search(
            game_state,
            perspective,
            temperature=self._temperature,
            add_root_noise=True,
            time_budget_ms=self._time_budget_ms,
        )

        # Sample action based on visit counts and temperature
        action_idx = self._sample_action(visit_distribution, self._temperature)
        return action_space.index_to_action(action_idx)

    def select_action_deterministic(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select best action deterministically.

        Args:
            game_state: Current game state
            legal_actions: Available legal actions

        Returns:
            Action with highest visit count
        """
        if len(legal_actions) == 1:
            return legal_actions[0]

        perspective = game_state.active_player

        visit_distribution = self.mcts.search(
            game_state,
            perspective,
            temperature=0.0,
            add_root_noise=False,
            time_budget_ms=self._time_budget_ms,
        )

        if not visit_distribution:
            return legal_actions[0]

        best_action_idx = max(visit_distribution.keys(), key=lambda a: visit_distribution[a])
        return action_space.index_to_action(best_action_idx)

    def get_policy_with_stats(
        self,
        game_state: state.State,
    ) -> tuple[dict[int, float], SearchStats]:
        """Get MCTS policy distribution and search statistics.

        Args:
            game_state: Current game state

        Returns:
            Tuple of (policy_distribution, search_stats)
        """
        perspective = game_state.active_player
        return self.mcts.search_with_stats(
            game_state,
            perspective,
            temperature=self._temperature,
            add_root_noise=True,
            time_budget_ms=self._time_budget_ms,
        )

    def _sample_action(
        self,
        visit_distribution: dict[int, float],
        temperature: float,
    ) -> int:
        """Sample action from visit distribution with temperature.

        Args:
            visit_distribution: Mapping from action indices to visit probabilities
            temperature: Temperature for sampling

        Returns:
            Sampled action index
        """
        if not visit_distribution:
            raise ValueError("No actions in visit distribution")

        actions_list = list(visit_distribution.keys())
        probs = np.array([visit_distribution[a] for a in actions_list])

        if temperature == 0:
            best_idx = int(np.argmax(probs))
            return actions_list[best_idx]
        elif temperature == 1.0:
            return int(np.random.choice(actions_list, p=probs))
        else:
            log_probs = np.log(probs + 1e-10)
            scaled_logits = log_probs / temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            scaled_probs = exp_logits / np.sum(exp_logits)
            return int(np.random.choice(actions_list, p=scaled_probs))

    def set_strength(self, strength: str | StrengthLevel) -> None:
        """Update strength level.

        Args:
            strength: New strength level
        """
        self._strength = strength if isinstance(strength, StrengthLevel) else StrengthLevel(strength.lower())
        self.mcts.set_strength(self._strength)

    def set_time_budget(self, time_budget_ms: float | None) -> None:
        """Update time budget per move.

        Args:
            time_budget_ms: New time budget in milliseconds (None = unlimited)
        """
        self._time_budget_ms = time_budget_ms

    def set_temperature(self, temperature: float) -> None:
        """Update action selection temperature.

        Args:
            temperature: New temperature value
        """
        if temperature < 0:
            raise ValueError(f"Temperature must be non-negative, got {temperature}")
        self._temperature = temperature


__all__ = [
    "STRENGTH_CONFIGS",
    "AdaptiveMCTS",
    "AdaptiveMCTSAgent",
    "SearchStats",
    "StrengthConfig",
    "StrengthLevel",
]
