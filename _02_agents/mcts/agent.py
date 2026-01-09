"""Neural MCTS agent for Beasty Bar.

This module provides an Agent implementation that uses neural network-guided
Monte Carlo Tree Search for action selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from _01_simulator import action_space, actions, state
from _02_agents.base import Agent

from .search import MCTS

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from _02_agents.neural.network import BeastyBarNetwork


class MCTSAgent(Agent):
    """Agent using neural MCTS for action selection.

    This agent combines a neural network policy-value estimator with
    Monte Carlo Tree Search to select strong moves. The network guides
    the search with policy priors and value estimates.

    Key features:
    - Neural network-guided tree search
    - Configurable exploration vs exploitation
    - Temperature-based stochasticity
    - Dirichlet noise for exploration during training
    """

    def __init__(
        self,
        network: BeastyBarNetwork,
        num_simulations: int = 400,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
        device: torch.device | str | None = None,  # type: ignore[name-defined]
        name: str | None = None,
    ) -> None:
        """Initialize MCTS agent with neural network.

        Args:
            network: Policy-value network for position evaluation
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for PUCT formula
            dirichlet_alpha: Dirichlet noise concentration parameter
            dirichlet_epsilon: Mixing weight for Dirichlet noise
            temperature: Temperature for action sampling (1.0 = stochastic, 0.1 = greedy)
            device: Device to run network on (auto-detected if None)
            name: Custom name for the agent (defaults to class name)
        """
        self._network = network
        self._num_simulations = num_simulations
        self._temperature = temperature
        self._name = name

        # Initialize MCTS
        self.mcts = MCTS(
            network=network,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            device=device,
        )

    @property
    def name(self) -> str:
        """Return agent name for display."""
        if self._name:
            return self._name
        return f"MCTSAgent(sims={self._num_simulations})"

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select action using neural MCTS.

        Args:
            game_state: Current game state
            legal_actions: Available legal actions

        Returns:
            Selected action from MCTS search
        """
        # Early exit for single legal action
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Determine perspective from active player
        perspective = game_state.active_player

        # Run MCTS search
        visit_distribution = self.mcts.search(
            game_state,
            perspective,
            temperature=self._temperature,
            add_root_noise=True,  # Add noise for exploration
        )

        # Sample action based on visit counts and temperature
        action_idx = self._sample_action(visit_distribution, self._temperature)

        # Convert action index to Action object
        return action_space.index_to_action(action_idx)

    def select_action_deterministic(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select best action deterministically (for evaluation).

        Args:
            game_state: Current game state
            legal_actions: Available legal actions

        Returns:
            Action with highest visit count
        """
        # Early exit for single legal action
        if len(legal_actions) == 1:
            return legal_actions[0]

        perspective = game_state.active_player

        # Run MCTS search without root noise
        visit_distribution = self.mcts.search(
            game_state,
            perspective,
            temperature=0.0,  # Not used in this case
            add_root_noise=False,
        )

        # Select action with highest visit count
        if not visit_distribution:
            # Fallback to first legal action
            return legal_actions[0]

        best_action_idx = max(visit_distribution.keys(), key=lambda a: visit_distribution[a])
        return action_space.index_to_action(best_action_idx)

    def get_policy(
        self,
        game_state: state.State,
    ) -> dict[int, float]:
        """Get MCTS policy distribution over actions.

        Useful for training data generation in self-play.

        Args:
            game_state: Current game state

        Returns:
            Dictionary mapping action indices to probabilities
        """
        perspective = game_state.active_player

        return self.mcts.search(
            game_state,
            perspective,
            temperature=self._temperature,
            add_root_noise=True,
        )

    def _sample_action(
        self,
        visit_distribution: dict[int, float],
        temperature: float,
    ) -> int:
        """Sample action from visit distribution with temperature.

        Args:
            visit_distribution: Mapping from action indices to visit probabilities
            temperature: Temperature for sampling (0 = greedy, 1 = stochastic)

        Returns:
            Sampled action index
        """
        if not visit_distribution:
            raise ValueError("No actions in visit distribution")

        # Convert to arrays
        actions_list = list(visit_distribution.keys())
        probs = np.array([visit_distribution[a] for a in actions_list])

        # Apply temperature
        if temperature == 0:
            # Greedy selection
            best_idx = int(np.argmax(probs))
            return actions_list[best_idx]
        elif temperature == 1.0:
            # Stochastic selection (use visit distribution directly)
            return int(np.random.choice(actions_list, p=probs))
        else:
            # Temperature scaling
            # Convert to "logits" by taking log, apply temperature, then softmax
            log_probs = np.log(probs + 1e-10)
            scaled_logits = log_probs / temperature

            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            scaled_probs = exp_logits / np.sum(exp_logits)

            return int(np.random.choice(actions_list, p=scaled_probs))

    def set_temperature(self, temperature: float) -> None:
        """Update action selection temperature.

        Useful for adjusting exploration during training (e.g., annealing).

        Args:
            temperature: New temperature value (0 = greedy, 1 = stochastic)
        """
        if temperature < 0:
            raise ValueError(f"Temperature must be non-negative, got {temperature}")
        self._temperature = temperature

    def set_num_simulations(self, num_simulations: int) -> None:
        """Update number of MCTS simulations.

        Args:
            num_simulations: New simulation count
        """
        if num_simulations < 1:
            raise ValueError(f"num_simulations must be positive, got {num_simulations}")
        self._num_simulations = num_simulations
        self.mcts.num_simulations = num_simulations


__all__ = ["MCTSAgent"]
