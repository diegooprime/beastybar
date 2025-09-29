"""Tests for reward helper utilities."""
from __future__ import annotations

from _01_simulator import rewards, rules, state


def _state_with_scores(p0_points: int, p1_points: int) -> state.State:
    base = state.initial_state(seed=1)
    bar_cards = []
    if p0_points:
        species = next(s for s in rules.BASE_DECK if rules.SPECIES[s].points == p0_points)
        bar_cards.append(state.Card(owner=0, species=species))
    if p1_points:
        species = next(s for s in rules.BASE_DECK if rules.SPECIES[s].points == p1_points)
        bar_cards.append(state.Card(owner=1, species=species))

    return state.State(
        seed=base.seed,
        turn=base.turn,
        active_player=base.active_player,
        players=base.players,
        zones=state.Zones(beasty_bar=tuple(bar_cards)),
    )


def test_win_loss_rewards() -> None:
    winning_state = _state_with_scores(p0_points=4, p1_points=0)
    losing_state = _state_with_scores(p0_points=0, p1_points=4)
    draw_state = _state_with_scores(p0_points=0, p1_points=0)

    assert rewards.win_loss(winning_state) == (1, -1)
    assert rewards.win_loss(losing_state) == (-1, 1)
    assert rewards.win_loss(draw_state) == (0, 0)


def test_normalized_margin() -> None:
    state_a = _state_with_scores(p0_points=4, p1_points=0)
    margin = rewards.normalized_margin(state_a)
    expected = 4 / sum(rules.SPECIES[s].points for s in rules.BASE_DECK)
    assert margin == (expected, -expected)


def test_shaped_reward_is_deterministic_with_seed() -> None:
    terminal = _state_with_scores(p0_points=4, p1_points=0)
    reward_a = rewards.shaped_reward(terminal, seed=123)
    reward_b = rewards.shaped_reward(terminal, seed=123)
    reward_c = rewards.shaped_reward(terminal, seed=321)

    assert reward_a == reward_b
    assert reward_a != reward_c
