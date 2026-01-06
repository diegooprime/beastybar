"""Tests for legal action catalog and masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from _01_simulator import action_space, engine, rules, state

if TYPE_CHECKING:
    from collections.abc import Iterable


def _build_state(
    *,
    hand_species: Iterable[str],
    queue_species: Iterable[str],
    seed: int = 0,
    active_player: int = 0,
) -> state.State:
    base = state.initial_state(seed=seed, starting_player=active_player)
    players = list(base.players)

    hand_cards = tuple(state.Card(owner=active_player, species=s) for s in hand_species)
    players[active_player] = state.PlayerState(deck=players[active_player].deck, hand=hand_cards)

    queue_cards = []
    owner = active_player
    for species in queue_species:
        queue_cards.append(state.Card(owner=owner, species=species))
        owner = (owner + 1) % rules.PLAYER_COUNT

    return state.State(
        seed=base.seed,
        turn=base.turn,
        active_player=active_player,
        players=tuple(players),
        zones=state.Zones(queue=tuple(queue_cards), beasty_bar=(), thats_it=()),
    )


def test_action_catalog_size() -> None:
    catalog = action_space.canonical_actions()
    expected = rules.HAND_SIZE * (1 + rules.MAX_QUEUE_LENGTH + rules.MAX_QUEUE_LENGTH**2)
    assert len(catalog) == expected


def test_mask_matches_engine_legal_actions() -> None:
    base = state.initial_state(seed=123)
    mask = action_space.legal_action_space(base, perspective=0)
    legal = tuple(engine.legal_actions(base, 0))

    assert sum(mask.mask) == len(legal)
    for action in legal:
        index = action_space.action_index(action)
        assert mask.mask[index] == 1


def test_chameleon_parrot_actions_are_present() -> None:
    # Queue of parrots ensures chameleon requires two-parameter actions.
    game_state = _build_state(
        hand_species=["chameleon"],
        queue_species=["parrot", "parrot", "parrot", "parrot", "parrot"],
    )
    mask = action_space.legal_action_space(game_state, perspective=0)
    legal = tuple(engine.legal_actions(game_state, 0))

    two_param_actions = [a for a in legal if len(a.params) == 2]
    assert two_param_actions, "Expected chameleon to generate two-parameter actions"

    action_space.canonical_actions()
    for action in two_param_actions:
        index = action_space.action_index(action)
        assert mask.mask[index] == 1


def test_inactive_player_has_empty_mask() -> None:
    base = state.initial_state(seed=9)
    mask = action_space.legal_action_space(base, perspective=1)
    assert sum(mask.mask) == 0
