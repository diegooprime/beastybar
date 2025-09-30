"""Utilities to convert simulator observations into model-ready tensors."""
from __future__ import annotations

from typing import Iterable, List

import torch

from _01_simulator import observations


def _flatten_zone(zone: Iterable[observations.CardEncoding]) -> List[float]:
    features: List[float] = []
    for card in zone:
        features.extend(float(component) for component in card)
    return features


def encode_observation(obs: observations.Observation) -> torch.Tensor:
    """Flatten the observation dataclass into a 1-D float tensor."""

    features: List[float] = []
    features.extend(_flatten_zone(obs.queue))
    features.extend(_flatten_zone(obs.beasty_bar))
    features.extend(_flatten_zone(obs.thats_it))
    features.extend(_flatten_zone(obs.hand))

    features.extend(float(count) for count in obs.deck_counts)
    features.extend(float(count) for count in obs.hand_counts)
    features.append(float(obs.active_player))
    features.append(float(obs.perspective))
    features.append(float(obs.turn))

    return torch.tensor(features, dtype=torch.float32)


def observation_size() -> int:
    """Return the length of the flattened observation vector."""

    dummy = observations.build_observation(
        game_state=_build_dummy_state(),
        perspective=0,
        mask_hidden=False,
    )
    return encode_observation(dummy).numel()


def _build_dummy_state():
    from _01_simulator import state

    base = state.initial_state(seed=0)
    return base


__all__ = ["encode_observation", "observation_size"]
