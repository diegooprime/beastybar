"""Activation capture system using PyTorch forward hooks.

This module provides a wrapper around the neural network that captures
intermediate activations during forward passes for visualization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from _02_agents.neural.network import BeastyBarNetwork


@dataclass
class ActivationSnapshot:
    """Captured activations from a single forward pass."""

    turn: int
    player: int
    timestamp_ms: int

    # Zone encoder outputs (each: 1 x hidden_dim after squeeze)
    queue_rep: torch.Tensor | None = None
    bar_rep: torch.Tensor | None = None
    thats_it_rep: torch.Tensor | None = None
    hand_rep: torch.Tensor | None = None
    opponent_hand_rep: torch.Tensor | None = None

    # Fusion block outputs (each: 1 x hidden_dim)
    fusion1_out: torch.Tensor | None = None
    fusion2_out: torch.Tensor | None = None
    fusion3_out: torch.Tensor | None = None

    # Final outputs
    policy_logits: torch.Tensor | None = None
    value: torch.Tensor | None = None

    # Context
    legal_mask: torch.Tensor | None = None
    action_taken: int | None = None
    action_label: str | None = None

    # All action labels (indexed by action catalog index)
    action_labels: dict[int, dict[str, Any]] = field(default_factory=dict)

    # Game context for UI
    game_context: dict[str, Any] = field(default_factory=dict)


class ActivationCaptureWrapper:
    """Wrapper that captures activations during neural network forward passes.

    Usage:
        wrapper = ActivationCaptureWrapper(model)
        policy, value, snapshot = wrapper.forward_with_capture(obs, mask, turn, player)
        wrapper.cleanup()  # Remove hooks when done
    """

    # Hook target module names
    ZONE_ENCODERS = [
        "queue_encoder",
        "bar_encoder",
        "thats_it_encoder",
        "hand_encoder",
        "opponent_hand_encoder",
    ]
    FUSION_BLOCKS = ["fusion1", "fusion2", "fusion3"]

    def __init__(self, model: BeastyBarNetwork) -> None:
        """Initialize wrapper with hooks on target modules.

        Args:
            model: The BeastyBarNetwork to capture activations from.
        """
        self.model = model
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Register forward hooks on all target modules."""
        # Zone encoder hooks
        for name in self.ZONE_ENCODERS:
            module = getattr(self.model, name, None)
            if module is not None:
                self._register_hook(module, name)

        # Fusion block hooks
        for name in self.FUSION_BLOCKS:
            module = getattr(self.model, name, None)
            if module is not None:
                self._register_hook(module, name)

    def _register_hook(self, module: nn.Module, name: str) -> None:
        """Register a forward hook on a module to capture its output."""

        def hook(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            # Detach and move to CPU to avoid GPU memory retention
            self._activations[name] = output.detach().cpu()

        handle = module.register_forward_hook(hook)
        self._hooks.append(handle)

    def forward_with_capture(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        turn: int,
        player: int,
        game_context: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ActivationSnapshot]:
        """Forward pass that captures and returns activations.

        Args:
            obs: Observation tensor (988-dim or batch x 988)
            mask: Legal action mask (124-dim or batch x 124)
            turn: Current game turn number
            player: Current player index (0 or 1)
            game_context: Optional dict with game state for UI display

        Returns:
            Tuple of (policy_logits, value, activation_snapshot)
        """
        # Clear previous activations
        self._activations.clear()

        # Run forward pass (hooks capture activations automatically)
        with torch.no_grad():
            policy_logits, value = self.model(obs, mask)

        # Build snapshot from captured activations
        snapshot = ActivationSnapshot(
            turn=turn,
            player=player,
            timestamp_ms=int(time.time() * 1000),
            # Zone representations
            queue_rep=self._activations.get("queue_encoder"),
            bar_rep=self._activations.get("bar_encoder"),
            thats_it_rep=self._activations.get("thats_it_encoder"),
            hand_rep=self._activations.get("hand_encoder"),
            opponent_hand_rep=self._activations.get("opponent_hand_encoder"),
            # Fusion outputs
            fusion1_out=self._activations.get("fusion1"),
            fusion2_out=self._activations.get("fusion2"),
            fusion3_out=self._activations.get("fusion3"),
            # Final outputs
            policy_logits=policy_logits.detach().cpu(),
            value=value.detach().cpu(),
            legal_mask=mask.detach().cpu() if isinstance(mask, torch.Tensor) else None,
            game_context=game_context or {},
        )

        return policy_logits, value, snapshot

    def set_action_taken(
        self, snapshot: ActivationSnapshot, action_idx: int, action_label: str
    ) -> None:
        """Update snapshot with the action that was actually taken.

        Args:
            snapshot: The snapshot to update
            action_idx: Index of action taken (0-123)
            action_label: Human-readable action description
        """
        snapshot.action_taken = action_idx
        snapshot.action_label = action_label

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._activations.clear()

    def __del__(self) -> None:
        """Ensure hooks are cleaned up on deletion."""
        self.cleanup()


class VisualizingNeuralAgent:
    """Neural agent wrapper that captures activations for visualization.

    This wraps the standard NeuralAgent to provide activation capture
    without modifying the original agent implementation.
    """

    def __init__(
        self,
        agent: Any,  # NeuralAgent
        websocket_manager: Any | None = None,  # VisualizerWebSocketManager
    ) -> None:
        """Initialize visualizing agent.

        Args:
            agent: The NeuralAgent to wrap
            websocket_manager: Optional WebSocket manager for broadcasting
        """
        self.agent = agent
        self.websocket_manager = websocket_manager
        self._capture_wrapper: ActivationCaptureWrapper | None = None
        self._activation_history: list[ActivationSnapshot] = []
        self._max_history = 100  # Keep last 100 snapshots

        # Initialize capture wrapper
        # NeuralAgent stores model as _model
        model = getattr(agent, "_model", None) or getattr(agent, "network", None)
        if model is not None:
            self._capture_wrapper = ActivationCaptureWrapper(model)

        # Get device from agent or model
        self._device = getattr(agent, "_device", None)
        if self._device is None and model is not None:
            self._device = next(model.parameters()).device

    @property
    def activation_history(self) -> list[ActivationSnapshot]:
        """Get activation history for replay mode."""
        return self._activation_history

    def clear_history(self) -> None:
        """Clear activation history (call on new game)."""
        self._activation_history.clear()

    async def select_action_with_viz(
        self,
        game_state: Any,  # state.State
        legal_actions: list[Any],  # list[Action]
        turn: int,
        player: int,
        game_context: dict[str, Any] | None = None,
    ) -> tuple[Any, ActivationSnapshot | None]:  # (Action, snapshot)
        """Select action while capturing activations for visualization.

        Args:
            game_state: Current game state
            legal_actions: List of legal actions
            turn: Current turn number
            player: Current player index
            game_context: Optional game context for UI

        Returns:
            Tuple of (selected_action, activation_snapshot)
        """
        from _01_simulator.action_space import action_index, legal_action_mask_tensor
        from _01_simulator.observations import state_to_tensor

        snapshot = None

        if self._capture_wrapper is not None:
            # Prepare tensors and move to correct device
            obs = torch.from_numpy(state_to_tensor(game_state, player)).float()
            mask = torch.from_numpy(
                legal_action_mask_tensor(game_state, player)
            ).float()

            # Move to model device
            if self._device is not None:
                obs = obs.to(self._device)
                mask = mask.to(self._device)

            # Build action labels for all legal actions
            action_labels = self._build_action_labels(game_state, legal_actions, player)

            # Forward with capture
            _policy_logits, _value, snapshot = self._capture_wrapper.forward_with_capture(
                obs.unsqueeze(0),
                mask.unsqueeze(0),
                turn,
                player,
                game_context,
            )

            # Add action labels to snapshot
            snapshot.action_labels = action_labels

            # Select action using agent's method
            action = self.agent.select_action(game_state, legal_actions)

            # Record which action was taken
            action_idx = action_index(action)
            action_label = self._get_action_label(action, game_state, player)
            self._capture_wrapper.set_action_taken(snapshot, action_idx, action_label)

            # Add to history
            self._activation_history.append(snapshot)
            if len(self._activation_history) > self._max_history:
                self._activation_history.pop(0)

            # Broadcast if websocket manager is available
            if self.websocket_manager is not None:
                from _04_ui.visualization.data_compression import snapshot_to_dict

                await self.websocket_manager.broadcast_activation(
                    snapshot_to_dict(snapshot)
                )
        else:
            # Fallback: just select action without capture
            action = self.agent.select_action(game_state, legal_actions)

        return action, snapshot

    def select_action_with_capture_sync(
        self,
        game_state: Any,  # state.State
        legal_actions: list[Any],  # list[Action]
        turn: int,
        player: int,
        game_context: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:  # (Action, snapshot_dict)
        """Synchronous version for batch processing (e.g., AI battles).

        Returns snapshot as dict instead of ActivationSnapshot object,
        and doesn't broadcast via WebSocket.
        """
        from _01_simulator.action_space import action_index, legal_action_mask_tensor
        from _01_simulator.observations import state_to_tensor
        from _04_ui.visualization.data_compression import snapshot_to_dict

        snapshot_dict = None

        if self._capture_wrapper is not None:
            # Prepare tensors
            obs = torch.from_numpy(state_to_tensor(game_state, player)).float()
            mask = torch.from_numpy(
                legal_action_mask_tensor(game_state, player)
            ).float()

            if self._device is not None:
                obs = obs.to(self._device)
                mask = mask.to(self._device)

            action_labels = self._build_action_labels(game_state, legal_actions, player)

            _policy_logits, _value, snapshot = self._capture_wrapper.forward_with_capture(
                obs.unsqueeze(0),
                mask.unsqueeze(0),
                turn,
                player,
                game_context,
            )

            snapshot.action_labels = action_labels

            # Select action
            action = self.agent.select_action(game_state, legal_actions)

            # Record action taken
            action_idx = action_index(action)
            action_label = self._get_action_label(action, game_state, player)
            self._capture_wrapper.set_action_taken(snapshot, action_idx, action_label)

            # Convert to dict for JSON serialization
            snapshot_dict = snapshot_to_dict(snapshot)
        else:
            action = self.agent.select_action(game_state, legal_actions)

        return action, snapshot_dict

    def _build_action_labels(
        self, game_state: Any, legal_actions: list[Any], player: int
    ) -> dict[int, dict[str, Any]]:
        """Build labels and metadata for all legal actions."""
        from _01_simulator.action_space import action_index

        labels = {}
        hand = game_state.players[player].hand

        for action in legal_actions:
            idx = action_index(action)
            if action.hand_index < len(hand):
                card = hand[action.hand_index]
                # Build descriptive label
                if action.params:
                    params_str = ", ".join(str(p) for p in action.params)
                    label = f"Play {card.species.title()} ({params_str})"
                else:
                    label = f"Play {card.species.title()}"

                labels[idx] = {
                    "label": label,
                    "species": card.species,
                    "strength": card.strength,
                    "points": card.points,
                    "hand_index": action.hand_index,
                    "params": list(action.params) if action.params else [],
                }
            else:
                labels[idx] = {
                    "label": f"Action {idx}",
                    "species": "unknown",
                    "strength": 0,
                    "points": 0,
                    "hand_index": action.hand_index,
                    "params": [],
                }

        return labels

    def _get_action_label(
        self, action: Any, game_state: Any, player: int
    ) -> str:
        """Generate human-readable action label."""
        try:
            hand = game_state.players[player].hand
            if action.hand_index < len(hand):
                card = hand[action.hand_index]
                if action.params:
                    params_str = ", ".join(str(p) for p in action.params)
                    return f"Play {card.species.title()} ({params_str})"
                return f"Play {card.species.title()}"
        except Exception:
            pass
        return f"Action {action.hand_index}"

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._capture_wrapper is not None:
            self._capture_wrapper.cleanup()
