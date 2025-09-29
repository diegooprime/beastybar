"""Learnable agent wrapper for self-play RL checkpoints."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from _01_simulator import action_space, actions, observations, state
from .base import Agent, ensure_legal

ModelFn = Callable[[observations.Observation], Sequence[float]]


@dataclass
class ExplorationConfig:
    """Toggle stochastic exploration during inference."""

    temperature: float = 0.0
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")

    @property
    def deterministic(self) -> bool:
        return self.temperature == 0.0 and self.epsilon == 0.0


class SelfPlayRLAgent(Agent):
    """Agent driven by a learned policy checkpoint.

    The agent relies on a caller-supplied ``model_factory`` that produces a
    callable mapping an :class:`observations.Observation` to a sequence of logits
    aligned with the canonical action catalog provided by
    :mod:`_01_simulator.action_space`. Tests can inject lightweight mocks without
    requiring heavy ML frameworks. Exploration is deterministic whenever
    ``temperature`` and ``epsilon`` are both zero.
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], ModelFn],
        exploration: Optional[ExplorationConfig] = None,
    ) -> None:
        self._model_factory = model_factory
        self._exploration = exploration or ExplorationConfig()
        self._model: Optional[ModelFn] = None

    def start_game(self, game_state: state.State) -> None:  # pragma: no cover - trivial
        if self._model is None:
            self._model = self._load_model()

    def select_action(
        self,
        game_state: state.State,
        legal: Sequence[actions.Action],
    ) -> actions.Action:
        if not legal:
            raise RuntimeError("SelfPlayRLAgent received no legal actions")

        if self._model is None:
            self._model = self._load_model()

        obs = observations.build_observation(game_state, game_state.active_player)
        logits = list(self._model(obs))
        catalog = action_space.canonical_actions()
        if len(logits) != len(catalog):
            raise ValueError("Model returned logits of unexpected length")

        action_space_view = action_space.legal_action_space(game_state, game_state.active_player)
        chosen_catalog_index = self._select_index(game_state, logits, action_space_view)
        chosen_action = catalog[chosen_catalog_index]
        return ensure_legal(chosen_action, legal)

    def _load_model(self) -> ModelFn:
        model = self._model_factory()
        if not callable(model):
            raise TypeError("model_factory must return a callable")
        return model

    def _select_index(
        self,
        game_state: state.State,
        logits: Sequence[float],
        action_space_view: action_space.ActionSpace,
    ) -> int:
        legal_entries = [(index, logits[index]) for index in action_space_view.legal_indices]
        if not legal_entries:
            raise ValueError("No legal actions available in catalog")

        if self._exploration.deterministic:
            return max(legal_entries, key=lambda item: item[1])[0]

        rng = random.Random(self._rng_seed(game_state))

        if self._exploration.epsilon > 0.0 and rng.random() < self._exploration.epsilon:
            return rng.choice([index for index, _ in legal_entries])

        if self._exploration.temperature > 0.0:
            adjusted = [logit / self._exploration.temperature for _, logit in legal_entries]
            max_adjusted = max(adjusted)
            weights = [math.exp(value - max_adjusted) for value in adjusted]
            total = sum(weights)
            if total <= 0:
                return max(legal_entries, key=lambda item: item[1])[0]
            draw = rng.random() * total
            cumulative = 0.0
            for (index, _), weight in zip(legal_entries, weights):
                cumulative += weight
                if draw <= cumulative:
                    return index

        return max(legal_entries, key=lambda item: item[1])[0]

    @staticmethod
    def _rng_seed(game_state: state.State) -> int:
        return game_state.seed * 1_000_003 + game_state.turn


__all__ = ["ExplorationConfig", "SelfPlayRLAgent"]
