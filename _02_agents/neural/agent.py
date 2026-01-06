"""Neural agent implementation for Beasty Bar.

This module provides NeuralAgent, a reinforcement learning agent that uses
a neural network policy for action selection. The agent supports multiple
inference modes (greedy, stochastic, temperature-scaled) and batch inference
for parallel game simulation.

Example:
    >>> from _02_agents.neural import BeastyBarNetwork, NeuralAgent
    >>> model = BeastyBarNetwork()
    >>> agent = NeuralAgent(model, mode="greedy")
    >>> action = agent.select_action(game_state, legal_actions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from _01_simulator.action_space import (
    ACTION_DIM,
    index_to_action,
    legal_action_mask_tensor,
)
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor
from _02_agents.base import Agent

from .network import BeastyBarNetwork
from .utils import (
    NetworkConfig,
    batch_greedy_actions,
    batch_sample_actions,
    get_device,
    greedy_action,
    load_network_from_checkpoint,
    sample_action,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _01_simulator.actions import Action
    from _01_simulator.state import State

logger = logging.getLogger(__name__)

InferenceMode = Literal["greedy", "stochastic", "temperature"]


class NeuralAgent(Agent):
    """Agent that uses a neural network for action selection.

    The NeuralAgent wraps a BeastyBarNetwork and provides an interface
    compatible with the Agent base class. It supports multiple inference
    modes for different use cases (evaluation vs training).

    Attributes:
        model: The neural network used for policy and value estimation.
        device: The device tensors are placed on (CPU, CUDA, or MPS).
        mode: Action selection mode ("greedy", "stochastic", or "temperature").
        temperature: Temperature parameter for temperature-scaled sampling.
    """

    def __init__(
        self,
        model: BeastyBarNetwork,
        device: torch.device | None = None,
        mode: InferenceMode = "stochastic",
        temperature: float = 1.0,
    ) -> None:
        """Initialize the neural agent.

        Args:
            model: Trained BeastyBarNetwork for policy inference.
            device: Device for tensor operations. If None, auto-detects
                the best available device (CUDA > MPS > CPU).
            mode: Action selection mode:
                - "greedy": Always select highest probability action.
                - "stochastic": Sample from policy distribution (temp=1.0).
                - "temperature": Sample with custom temperature scaling.
            temperature: Temperature for "temperature" mode. Higher values
                increase exploration, lower values approach greedy selection.
                Only used when mode="temperature". Must be positive.

        Raises:
            ValueError: If temperature is not positive.
            ValueError: If mode is not one of the valid options.
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if mode not in ("greedy", "stochastic", "temperature"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'greedy', 'stochastic', or 'temperature'")

        self._model = model
        self._mode = mode
        self._temperature = temperature

        # Set up device
        if device is None:
            self._device = get_device()
        else:
            self._device = device

        # Move model to device and set to eval mode
        self._model = self._model.to(self._device)
        self._model.eval()

        logger.debug(
            f"NeuralAgent initialized: device={self._device}, mode={self._mode}, "
            f"temperature={self._temperature}"
        )

    @property
    def model(self) -> BeastyBarNetwork:
        """The underlying neural network model."""
        return self._model

    @property
    def device(self) -> torch.device:
        """The device tensors are placed on."""
        return self._device

    @property
    def mode(self) -> InferenceMode:
        """Current action selection mode."""
        return self._mode

    @mode.setter
    def mode(self, value: InferenceMode) -> None:
        """Set the action selection mode."""
        if value not in ("greedy", "stochastic", "temperature"):
            raise ValueError(f"Invalid mode: {value}")
        self._mode = value

    @property
    def temperature(self) -> float:
        """Temperature parameter for temperature-scaled sampling."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the temperature parameter."""
        if value <= 0:
            raise ValueError(f"Temperature must be positive, got {value}")
        self._temperature = value

    @property
    def name(self) -> str:
        """Return descriptive agent name for display and benchmarking."""
        if self._mode == "greedy":
            return "NeuralAgent(greedy)"
        elif self._mode == "stochastic":
            return "NeuralAgent(stochastic)"
        else:
            return f"NeuralAgent(temp={self._temperature:.2f})"

    def select_action(
        self,
        game_state: State,
        legal_actions: Sequence[Action],
    ) -> Action:
        """Select an action using the neural network policy.

        Converts the game state to a tensor observation, generates an
        action mask from legal actions, runs the network forward pass,
        and selects an action based on the current inference mode.

        Args:
            game_state: Current game state (automatically masked for
                opponent info via state_to_tensor).
            legal_actions: Available legal actions to choose from.

        Returns:
            Selected action from the legal actions.

        Raises:
            ValueError: If no legal actions are available.
            RuntimeError: If network forward pass fails.
        """
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Determine perspective from active player
        perspective = game_state.active_player

        # Convert state to observation tensor
        obs_np = state_to_tensor(game_state, perspective)
        obs_tensor = torch.from_numpy(obs_np).to(self._device)

        # Generate action mask
        mask_np = legal_action_mask_tensor(game_state, perspective)
        mask_tensor = torch.from_numpy(mask_np).to(self._device)

        # Run network forward pass (no gradients needed for inference)
        with torch.no_grad():
            policy_logits, _value = self._model(obs_tensor, mask_tensor)

        # Select action based on mode
        if self._mode == "greedy":
            action_idx = greedy_action(policy_logits, mask_tensor)
        elif self._mode == "stochastic":
            action_idx = sample_action(policy_logits, mask_tensor, temperature=1.0)
        else:  # temperature mode
            action_idx = sample_action(policy_logits, mask_tensor, temperature=self._temperature)

        # Convert index to action
        return index_to_action(action_idx)

    def select_actions_batch(
        self,
        game_states: list[State],
        legal_actions_list: list[Sequence[Action]],
    ) -> list[Action]:
        """Process multiple states in a single forward pass.

        Efficient batch inference for parallel game simulation. All states
        are encoded and processed together, reducing the overhead of
        individual forward passes.

        Args:
            game_states: List of game states to select actions for.
            legal_actions_list: List of legal action sequences, one per state.

        Returns:
            List of selected actions, one per state.

        Raises:
            ValueError: If lists have different lengths.
            ValueError: If any state has no legal actions.
        """
        if len(game_states) != len(legal_actions_list):
            raise ValueError(
                f"Mismatch: {len(game_states)} states vs {len(legal_actions_list)} action lists"
            )

        if not game_states:
            return []

        # Check all states have legal actions
        for i, legal_actions in enumerate(legal_actions_list):
            if not legal_actions:
                raise ValueError(f"State {i} has no legal actions")

        batch_size = len(game_states)

        # Encode all states and masks
        import numpy as np

        obs_batch = np.zeros((batch_size, OBSERVATION_DIM), dtype=np.float32)
        mask_batch = np.zeros((batch_size, ACTION_DIM), dtype=np.float32)

        for i, (game_state, _) in enumerate(zip(game_states, legal_actions_list, strict=True)):
            perspective = game_state.active_player
            obs_batch[i] = state_to_tensor(game_state, perspective)
            mask_batch[i] = legal_action_mask_tensor(game_state, perspective)

        # Convert to tensors
        obs_tensor = torch.from_numpy(obs_batch).to(self._device)
        mask_tensor = torch.from_numpy(mask_batch).to(self._device)

        # Run batch forward pass
        with torch.no_grad():
            policy_logits, _values = self._model(obs_tensor, mask_tensor)

        # Select actions based on mode
        if self._mode == "greedy":
            action_indices = batch_greedy_actions(policy_logits, mask_tensor)
        elif self._mode == "stochastic":
            action_indices = batch_sample_actions(policy_logits, mask_tensor, temperature=1.0)
        else:  # temperature mode
            action_indices = batch_sample_actions(
                policy_logits, mask_tensor, temperature=self._temperature
            )

        # Convert indices to actions
        return [index_to_action(int(idx.item())) for idx in action_indices]

    def get_policy_and_value(
        self,
        game_state: State,
        perspective: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Get full policy distribution and value estimate for a state.

        Useful for training data collection where both policy probabilities
        and value estimates are needed.

        Args:
            game_state: Current game state.
            perspective: Player perspective for encoding. If None, uses
                the active player.

        Returns:
            Tuple of:
                - policy_probs: Tensor of shape (ACTION_DIM,) with action probabilities.
                - mask: Tensor of shape (ACTION_DIM,) with action mask.
                - value: Scalar value estimate in [-1, 1].
        """
        if perspective is None:
            perspective = game_state.active_player

        # Convert state to observation tensor
        obs_np = state_to_tensor(game_state, perspective)
        obs_tensor = torch.from_numpy(obs_np).to(self._device)

        # Generate action mask
        mask_np = legal_action_mask_tensor(game_state, perspective)
        mask_tensor = torch.from_numpy(mask_np).to(self._device)

        # Run network forward pass
        with torch.no_grad():
            policy_logits, value = self._model(obs_tensor, mask_tensor)

        # Apply mask and softmax to get probabilities
        masked_logits = torch.where(
            mask_tensor > 0,
            policy_logits,
            torch.tensor(float("-inf"), device=self._device),
        )
        policy_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

        return policy_probs, mask_tensor, float(value.item())

    def to(self, device: torch.device | str) -> NeuralAgent:
        """Move the agent to a different device.

        Args:
            device: Target device.

        Returns:
            Self for method chaining.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self._model = self._model.to(device)
        return self

    def eval(self) -> NeuralAgent:
        """Set the model to evaluation mode.

        Returns:
            Self for method chaining.
        """
        self._model.eval()
        return self

    def train(self) -> NeuralAgent:
        """Set the model to training mode.

        Returns:
            Self for method chaining.
        """
        self._model.train()
        return self


def load_neural_agent(
    checkpoint_path: str | Path,
    mode: InferenceMode = "greedy",
    temperature: float = 1.0,
    device: str | torch.device | None = None,
) -> NeuralAgent:
    """Load a NeuralAgent from a checkpoint file.

    Factory function that handles loading the network from a checkpoint
    and wrapping it in a NeuralAgent for inference.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt).
        mode: Action selection mode for the agent.
        temperature: Temperature for temperature-scaled sampling.
        device: Device to place the model on. If None, auto-detects.

    Returns:
        NeuralAgent loaded from checkpoint, ready for inference.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint loading fails.

    Example:
        >>> agent = load_neural_agent("checkpoints/model_100000.pt", mode="greedy")
        >>> action = agent.select_action(state, legal_actions)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Convert device string to torch.device
    if isinstance(device, str):
        torch_device: torch.device | None = torch.device(device)
    else:
        torch_device = device

    # Load network from checkpoint
    model, _config, step = load_network_from_checkpoint(
        checkpoint_path,
        network_cls=BeastyBarNetwork,
        device=torch_device,
    )

    logger.info(
        f"Loaded NeuralAgent from checkpoint at step {step}, "
        f"device={torch_device or 'auto'}, mode={mode}"
    )

    # Create and return agent
    # The device will be auto-detected in NeuralAgent if None
    return NeuralAgent(
        model=model,  # type: ignore[arg-type]  # model is BeastyBarNetwork
        device=torch_device,
        mode=mode,
        temperature=temperature,
    )


def create_neural_agent(
    config: NetworkConfig | None = None,
    mode: InferenceMode = "stochastic",
    temperature: float = 1.0,
    device: str | torch.device | None = None,
) -> NeuralAgent:
    """Create a new NeuralAgent with randomly initialized weights.

    Factory function for creating a fresh agent for training or testing.

    Args:
        config: Network configuration. If None, uses defaults.
        mode: Action selection mode.
        temperature: Temperature for temperature-scaled sampling.
        device: Device to place the model on. If None, auto-detects.

    Returns:
        NeuralAgent with randomly initialized network.
    """
    # Create network
    model = BeastyBarNetwork(config)

    # Convert device
    if isinstance(device, str):
        torch_device: torch.device | None = torch.device(device)
    else:
        torch_device = device

    return NeuralAgent(
        model=model,
        device=torch_device,
        mode=mode,
        temperature=temperature,
    )


__all__ = [
    "InferenceMode",
    "NeuralAgent",
    "create_neural_agent",
    "load_neural_agent",
]
