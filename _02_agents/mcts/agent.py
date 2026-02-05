"""Lightweight MCTS agent placeholder.

This module provides a minimal MCTSAgent implementation to satisfy
training and evaluation integration. It does NOT implement full tree
search; instead, it uses the provided network to select actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from _01_simulator import actions, state
from _01_simulator.action_space import greedy_masked_action, legal_action_mask_tensor, index_to_action
from _01_simulator.observations import state_to_tensor
from _02_agents.base import Agent


@dataclass
class MCTSAgent(Agent):
    """Minimal MCTS agent shim using network policy for action selection.

    Args:
        network: Policy-value network used for action selection.
        num_simulations: Placeholder field for compatibility.
        c_puct: Placeholder field for compatibility.
        dirichlet_alpha: Placeholder field for compatibility.
        dirichlet_epsilon: Placeholder field for compatibility.
        temperature: Sampling temperature (unused in greedy selection).
        name: Optional display name.
    """

    network: Any
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    name: str = "MCTSAgent"

    def select_action(
        self,
        game_state: state.State,
        legal_actions: list[actions.Action] | tuple[actions.Action, ...],
    ) -> actions.Action:
        """Select an action using the network policy (greedy over masked logits)."""
        if not legal_actions:
            raise ValueError("No legal actions provided to MCTSAgent")

        if self.network is None:
            return legal_actions[0]

        # Build observation and mask from the active player's perspective.
        player = game_state.active_player
        obs = state_to_tensor(game_state, player)
        mask = legal_action_mask_tensor(game_state, player)

        obs_tensor = torch.from_numpy(obs).to(next(self.network.parameters()).device)
        mask_tensor = torch.from_numpy(mask).to(obs_tensor.device)

        with torch.no_grad():
            policy_logits, _value = self.network(obs_tensor, mask_tensor)

        action_idx = greedy_masked_action(
            logits=policy_logits.squeeze(0) if policy_logits.dim() > 1 else policy_logits,
            mask=mask_tensor.squeeze(0) if mask_tensor.dim() > 1 else mask_tensor,
        )
        return index_to_action(action_idx)

