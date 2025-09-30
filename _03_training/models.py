"""Torch policy/value networks for self-play training."""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, Sequence

import torch
from torch import nn


@dataclass
class PolicyConfig:
    """Configuration for the shared actor-critic network."""

    observation_size: int
    action_size: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PolicyConfig":
        return cls(
            observation_size=int(payload["observation_size"]),
            action_size=int(payload["action_size"]),
            hidden_sizes=tuple(int(value) for value in payload.get("hidden_sizes", (256, 256))),
            activation=str(payload.get("activation", "relu")),
            dropout=float(payload.get("dropout", 0.0)),
        )


class PolicyValueNet(nn.Module):
    """Simple MLP with shared trunk and policy/value heads."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        input_size = config.observation_size
        activation = _activation_factory(config.activation)

        for hidden in config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(activation())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            input_size = hidden

        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.policy_head = nn.Linear(input_size, config.action_size)
        self.value_head = nn.Linear(input_size, 1)

        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        features = self.body(observation)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value.squeeze(-1)


def save_checkpoint(
    path: str | PathLike[str],
    *,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    metadata: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": model.config.to_dict(),
        "step": int(step),
    }
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_checkpoint(
    path: str | PathLike[str],
    *,
    device: torch.device | str = "cpu",
    include_optimizer: bool = False,
) -> tuple[PolicyValueNet, Dict[str, Any] | None, Dict[str, Any]]:
    payload: Dict[str, Any] = torch.load(path, map_location=device)
    config = PolicyConfig.from_dict(payload["config"])
    model = PolicyValueNet(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    optimizer_state = payload.get("optimizer_state") if include_optimizer else None
    return model, optimizer_state, payload


def _activation_factory(name: str) -> type[nn.Module]:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU
    if normalized == "tanh":
        return nn.Tanh
    if normalized == "elu":
        return nn.ELU
    raise ValueError(f"Unsupported activation: {name}")


__all__ = ["PolicyConfig", "PolicyValueNet", "save_checkpoint", "load_checkpoint"]
