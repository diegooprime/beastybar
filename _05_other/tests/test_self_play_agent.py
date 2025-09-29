"""Smoke tests for the SelfPlayRLAgent."""
from __future__ import annotations

from typing import Iterable, Sequence

import pytest

from _01_simulator import action_space, engine, state
from _02_agents.self_play_rl import ExplorationConfig, SelfPlayRLAgent


def _constant_state(seed: int = 101) -> state.State:
    base = state.initial_state(seed=seed)
    return base


def _logits_targeting(action_indices: Iterable[int], total: int, high: float = 5.0) -> list[float]:
    logits = [-1.0] * total
    for idx, index in enumerate(action_indices):
        logits[index] = high - idx
    return logits


class _IterModel:
    def __init__(self, sequences: Sequence[Sequence[float]]) -> None:
        self._sequences = list(sequences)
        self.calls = 0

    def __call__(self, obs):  # noqa: ANN001 - match ModelFn signature
        if self.calls >= len(self._sequences):
            raise AssertionError("Model called more times than expected")
        result = self._sequences[self.calls]
        self.calls += 1
        return result


def test_deterministic_argmax_selection() -> None:
    game_state = _constant_state()
    legal = tuple(engine.legal_actions(game_state, game_state.active_player))
    catalog = action_space.canonical_actions()
    target_action = legal[-1]
    target_index = action_space.action_index(target_action)

    logits = _logits_targeting([target_index], len(catalog))
    model = _IterModel([logits])
    agent = SelfPlayRLAgent(model_factory=lambda: model)

    chosen = agent.select_action(game_state, legal)
    assert chosen == target_action
    assert model.calls == 1


@pytest.mark.parametrize("config", [ExplorationConfig(temperature=0.5), ExplorationConfig(epsilon=0.8)])
def test_exploration_remains_seeded(config: ExplorationConfig) -> None:
    game_state = _constant_state(seed=2025)
    legal = tuple(engine.legal_actions(game_state, game_state.active_player))

    catalog = action_space.canonical_actions()
    indices = [action_space.action_index(action) for action in legal]
    logits = _logits_targeting(indices, len(catalog), high=2.0)
    model = _IterModel([logits, logits])
    agent = SelfPlayRLAgent(model_factory=lambda: model, exploration=config)

    first = agent.select_action(game_state, legal)
    second = agent.select_action(game_state, legal)
    assert first == second


def test_model_factory_must_return_callable() -> None:
    game_state = _constant_state()
    legal = tuple(engine.legal_actions(game_state, game_state.active_player))
    agent = SelfPlayRLAgent(model_factory=lambda: 123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        agent.select_action(game_state, legal)


def test_invalid_logit_length_raises() -> None:
    game_state = _constant_state()
    legal = tuple(engine.legal_actions(game_state, game_state.active_player))
    catalog = action_space.canonical_actions()
    bad_logits = [0.0] * (len(catalog) - 1)
    model = _IterModel([bad_logits])
    agent = SelfPlayRLAgent(model_factory=lambda: model)

    with pytest.raises(ValueError):
        agent.select_action(game_state, legal)
