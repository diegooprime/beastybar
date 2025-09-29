"""Tests for observation encoding utilities."""
from __future__ import annotations

import pytest

from _01_simulator import observations, rules, state


def test_observation_has_fixed_lengths() -> None:
    game_state = state.initial_state(seed=7)
    obs = observations.build_observation(game_state, perspective=0)

    assert len(obs.queue) == rules.MAX_QUEUE_LENGTH
    assert len(obs.hand) == rules.HAND_SIZE
    assert len(obs.beasty_bar) == rules.DECK_SIZE * rules.PLAYER_COUNT
    assert len(obs.thats_it) == rules.DECK_SIZE * rules.PLAYER_COUNT

    for slot in obs.queue + obs.hand:
        assert len(slot) == 5  # present, owner, species id, strength, points


@pytest.mark.parametrize("mask_hidden", [True, False])
def test_hand_encoding_respects_mask(mask_hidden: bool) -> None:
    game_state = state.initial_state(seed=42)
    # Remove one card from the perspective hand to exercise padding behaviour.
    game_state, _ = state.remove_hand_card(game_state, player=0, index=0)

    obs = observations.build_observation(game_state, perspective=0, mask_hidden=mask_hidden)
    present_flags = [slot[0] for slot in obs.hand]

    assert present_flags.count(1) == len(game_state.players[0].hand)
    assert present_flags.count(0) == rules.HAND_SIZE - len(game_state.players[0].hand)


def test_queue_encoding_matches_card_features() -> None:
    base = state.initial_state(seed=99)
    lion = state.Card(owner=0, species="lion")
    updated = state.append_queue(base, lion)

    obs = observations.build_observation(updated, perspective=0)

    first_slot = obs.queue[0]
    assert first_slot[0] == 1  # present flag
    assert first_slot[1] == lion.owner
    assert first_slot[2] == observations.species_index("lion")
    assert first_slot[3] == lion.strength
    assert first_slot[4] == lion.points


def test_species_round_trip() -> None:
    for species in rules.SPECIES:
        index = observations.species_index(species)
        assert observations.species_name(index) == species
