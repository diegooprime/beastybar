"""Observation encoding utilities for the Beasty Bar simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import rules, state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

CardEncoding = tuple[int, int, int, int, int]

# Card encoding order: present flag, owner, species id, strength, points.

_SPECIES_INDEX = {species: index for index, species in enumerate(sorted(rules.SPECIES.keys()))}
_INDEX_TO_SPECIES = tuple(sorted(rules.SPECIES.keys()))
_UNKNOWN_ID = _SPECIES_INDEX["unknown"]
_TOTAL_CARD_SLOTS = rules.DECK_SIZE * rules.PLAYER_COUNT

_PAD_CARD: CardEncoding = (0, -1, _UNKNOWN_ID, 0, 0)


@dataclass(frozen=True)
class Observation:
    """Fixed-size encoding of a game state from one player's perspective."""

    queue: tuple[CardEncoding, ...]
    beasty_bar: tuple[CardEncoding, ...]
    thats_it: tuple[CardEncoding, ...]
    hand: tuple[CardEncoding, ...]
    deck_counts: tuple[int, ...]
    hand_counts: tuple[int, ...]
    active_player: int
    perspective: int
    turn: int

    def as_dict(self) -> dict[str, object]:
        """Return the observation as a dict for JSON/np serialization."""

        return {
            "queue": self.queue,
            "beasty_bar": self.beasty_bar,
            "thats_it": self.thats_it,
            "hand": self.hand,
            "deck_counts": self.deck_counts,
            "hand_counts": self.hand_counts,
            "active_player": self.active_player,
            "perspective": self.perspective,
            "turn": self.turn,
        }


def build_observation(
    game_state: state.State,
    perspective: int,
    *,
    mask_hidden: bool = True,
) -> Observation:
    """Encode public zones and the perspective hand into fixed-size tensors.

    Args:
        game_state: Source immutable state snapshot.
        perspective: Player index the observation is built for.
        mask_hidden: When true, opponent hands/decks are masked via
            :func:`state.mask_state_for_player` to prevent leaking hidden cards.

    Returns:
        Observation dataclass with card feature tuples and turn context.
    """

    if not (0 <= perspective < rules.PLAYER_COUNT):
        raise ValueError("Perspective player index out of range")

    visible_state = state.mask_state_for_player(game_state, perspective) if mask_hidden else game_state

    queue = _encode_zone(visible_state.zones.queue, rules.MAX_QUEUE_LENGTH)
    beasty_bar = _encode_zone(visible_state.zones.beasty_bar, _TOTAL_CARD_SLOTS)
    thats_it = _encode_zone(visible_state.zones.thats_it, _TOTAL_CARD_SLOTS)
    hand = _encode_zone(visible_state.players[perspective].hand, rules.HAND_SIZE)

    deck_counts = tuple(len(player.deck) for player in visible_state.players)
    hand_counts = tuple(len(player.hand) for player in visible_state.players)

    return Observation(
        queue=queue,
        beasty_bar=beasty_bar,
        thats_it=thats_it,
        hand=hand,
        deck_counts=deck_counts,
        hand_counts=hand_counts,
        active_player=visible_state.active_player,
        perspective=perspective,
        turn=visible_state.turn,
    )


def species_index(species: str) -> int:
    """Return the stable integer index for a species name."""

    try:
        return _SPECIES_INDEX[species]
    except KeyError as exc:
        raise ValueError(f"Unknown species: {species}") from exc


def species_name(index: int) -> str:
    """Reverse lookup the species string for an encoded id."""

    if not (0 <= index < len(_INDEX_TO_SPECIES)):
        raise ValueError("Species index out of range")
    return _INDEX_TO_SPECIES[index]


def _encode_zone(cards: Sequence[state.Card], max_len: int) -> tuple[CardEncoding, ...]:
    if max_len <= 0:
        raise ValueError("Zone length must be positive")

    encoded: list[CardEncoding] = []
    for card in cards:
        encoded.append(
            (
                1,
                card.owner,
                species_index(card.species),
                card.strength,
                card.points,
            )
        )

    if len(encoded) > max_len:
        raise ValueError("Zone contains more cards than the maximum length")

    encoded.extend(_PAD_CARD for _ in range(max_len - len(encoded)))
    return tuple(encoded)


# ============================================================================
# Tensor Observation Encoding for Neural Networks
# ============================================================================

# Tensor schema dimensions
# Species count (excluding 'unknown' for card encoding)
_NUM_SPECIES = len(rules.SPECIES) - 1  # 12 species (unknown not used in one-hot)
_TOTAL_CARD_SLOTS_IN_ZONES = rules.DECK_SIZE * rules.PLAYER_COUNT  # 24 cards

# Card feature vector dimensions
_CARD_PRESENCE_DIM = 1  # 0/1 flag
_CARD_OWNER_DIM = 1  # 0=opponent, 1=self, 0.5=neutral
_CARD_SPECIES_DIM = _NUM_SPECIES  # 12-dimensional one-hot
_CARD_STRENGTH_DIM = 1  # normalized [0, 1]
_CARD_POINTS_DIM = 1  # normalized [0, 1]
_CARD_POSITION_DIM = 1  # normalized [0, 1]
_CARD_FEATURE_DIM = (
    _CARD_PRESENCE_DIM
    + _CARD_OWNER_DIM
    + _CARD_SPECIES_DIM
    + _CARD_STRENGTH_DIM
    + _CARD_POINTS_DIM
    + _CARD_POSITION_DIM
)  # 17 dims per card

# Masked opponent hand (presence only, no species info)
_MASKED_CARD_FEATURE_DIM = _CARD_PRESENCE_DIM + _CARD_POSITION_DIM + 1  # 3 dims (presence, position, padding)

# Zone dimensions
_QUEUE_DIM = rules.MAX_QUEUE_LENGTH * _CARD_FEATURE_DIM  # 5 x 17 = 85
_BEASTY_BAR_DIM = _TOTAL_CARD_SLOTS_IN_ZONES * _CARD_FEATURE_DIM  # 24 x 17 = 408
_THATS_IT_DIM = _TOTAL_CARD_SLOTS_IN_ZONES * _CARD_FEATURE_DIM  # 24 x 17 = 408
_OWN_HAND_DIM = rules.HAND_SIZE * _CARD_FEATURE_DIM  # 4 x 17 = 68
_OPPONENT_HAND_DIM = rules.HAND_SIZE * _MASKED_CARD_FEATURE_DIM  # 4 x 3 = 12

# Scalar features
_SCALARS_DIM = 7  # deck_counts(2), hand_counts(2), current_player(1), turn_normalized(1), queue_length(1)

# Total observation dimension
OBSERVATION_DIM = (
    _QUEUE_DIM + _BEASTY_BAR_DIM + _THATS_IT_DIM + _OWN_HAND_DIM + _OPPONENT_HAND_DIM + _SCALARS_DIM
)  # 988 dims

# Max values for normalization
_MAX_STRENGTH = 12
_MAX_POINTS = 4
_MAX_TURN = 100  # Reasonable upper bound for normalization


def _encode_card_features(
    card_encoding: CardEncoding, position_in_zone: int, zone_length: int, perspective: int
) -> NDArray[np.float32]:
    """Encode a single card into a 17-dimensional feature vector.

    Args:
        card_encoding: Tuple (present, owner, species_id, strength, points)
        position_in_zone: 0-indexed position in the zone
        zone_length: Total length of the zone for normalization
        perspective: Player index for owner encoding

    Returns:
        17-dimensional float32 array with values in [0, 1] range
    """
    present, owner, species_id, strength, points = card_encoding

    features = []

    # Presence flag (1 dim)
    features.append(float(present))

    # Owner encoding (1 dim): 0=opponent, 1=self, 0.5=empty
    if present == 0:
        owner_val = 0.5
    elif owner == perspective:
        owner_val = 1.0
    else:
        owner_val = 0.0
    features.append(owner_val)

    # Species one-hot encoding (12 dims) - exclude 'unknown'
    species_onehot = [0.0] * _NUM_SPECIES
    if present == 1 and species_id != _UNKNOWN_ID:
        # Map species_id to one-hot, accounting for 'unknown' being in the index
        adjusted_id = species_id if species_id < _UNKNOWN_ID else species_id - 1
        if 0 <= adjusted_id < _NUM_SPECIES:
            species_onehot[adjusted_id] = 1.0
    features.extend(species_onehot)

    # Strength normalized (1 dim) - clamp to valid range
    strength_norm = min(strength, _MAX_STRENGTH) / _MAX_STRENGTH if present == 1 else 0.0
    features.append(strength_norm)

    # Points normalized (1 dim) - clamp to valid range
    points_norm = min(points, _MAX_POINTS) / _MAX_POINTS if present == 1 else 0.0
    features.append(points_norm)

    # Position in zone normalized (1 dim)
    position_norm = position_in_zone / max(zone_length - 1, 1) if zone_length > 1 else 0.0
    features.append(position_norm)

    return np.array(features, dtype=np.float32)


def _encode_masked_card_features(present: int, position_in_zone: int, zone_length: int) -> NDArray[np.float32]:
    """Encode opponent hand card with minimal information (presence only).

    Args:
        present: 0 or 1 indicating card presence
        position_in_zone: 0-indexed position in hand
        zone_length: Total length of hand for normalization

    Returns:
        3-dimensional float32 array
    """
    features = []

    # Presence flag (1 dim)
    features.append(float(present))

    # Position normalized (1 dim)
    position_norm = position_in_zone / max(zone_length - 1, 1) if zone_length > 1 else 0.0
    features.append(position_norm)

    # Padding (1 dim)
    features.append(0.0)

    return np.array(features, dtype=np.float32)


def observation_to_tensor(obs: Observation, perspective: int) -> NDArray[np.float32]:
    """Convert an Observation to a flattened tensor for neural network input.

    Args:
        obs: Observation dataclass with encoded zones
        perspective: Player index (for validation, already encoded in obs)

    Returns:
        Flattened float32 array of dimension OBSERVATION_DIM (988)
        All values in [0, 1] range
    """
    if perspective != obs.perspective:
        raise ValueError(f"Perspective mismatch: {perspective} != {obs.perspective}")

    tensor_parts = []

    # Encode queue zone (5 x 17 = 85)
    for i, card_enc in enumerate(obs.queue):
        card_features = _encode_card_features(card_enc, i, len(obs.queue), perspective)
        tensor_parts.append(card_features)

    # Encode Beasty Bar zone (24 x 17 = 408)
    for i, card_enc in enumerate(obs.beasty_bar):
        card_features = _encode_card_features(card_enc, i, len(obs.beasty_bar), perspective)
        tensor_parts.append(card_features)

    # Encode THAT'S IT zone (24 x 17 = 408)
    for i, card_enc in enumerate(obs.thats_it):
        card_features = _encode_card_features(card_enc, i, len(obs.thats_it), perspective)
        tensor_parts.append(card_features)

    # Encode own hand (4 x 17 = 68)
    for i, card_enc in enumerate(obs.hand):
        card_features = _encode_card_features(card_enc, i, len(obs.hand), perspective)
        tensor_parts.append(card_features)

    # Encode opponent hand (masked, 4 x 3 = 12)
    opponent_idx = 1 - perspective
    opponent_hand_count = obs.hand_counts[opponent_idx]
    for i in range(rules.HAND_SIZE):
        present = 1 if i < opponent_hand_count else 0
        masked_features = _encode_masked_card_features(present, i, rules.HAND_SIZE)
        tensor_parts.append(masked_features)

    # Encode scalar features (7)
    scalars = []

    # Deck counts (2): own deck, opponent deck, normalized
    scalars.append(obs.deck_counts[perspective] / rules.DECK_SIZE)
    scalars.append(obs.deck_counts[opponent_idx] / rules.DECK_SIZE)

    # Hand counts (2): own hand, opponent hand, normalized
    scalars.append(obs.hand_counts[perspective] / rules.HAND_SIZE)
    scalars.append(obs.hand_counts[opponent_idx] / rules.HAND_SIZE)

    # Current player indicator (1): 1 if active, 0 if waiting
    scalars.append(1.0 if obs.active_player == perspective else 0.0)

    # Turn number normalized (1)
    scalars.append(min(obs.turn / _MAX_TURN, 1.0))

    # Queue length normalized (1)
    scalars.append(len([c for c in obs.queue if c[0] == 1]) / rules.MAX_QUEUE_LENGTH)

    tensor_parts.append(np.array(scalars, dtype=np.float32))

    # Concatenate all parts
    tensor = np.concatenate(tensor_parts)

    assert tensor.shape == (OBSERVATION_DIM,), f"Expected {OBSERVATION_DIM} dims, got {tensor.shape}"
    return tensor


def state_to_tensor(game_state: state.State, perspective: int) -> NDArray[np.float32]:
    """Convenience function: convert State directly to tensor.

    Args:
        game_state: Full game state
        perspective: Player index to observe from

    Returns:
        Flattened float32 tensor of dimension OBSERVATION_DIM
    """
    obs = build_observation(game_state, perspective, mask_hidden=True)
    return observation_to_tensor(obs, perspective)


def batch_states_to_tensor(
    states: Sequence[state.State], perspectives: Sequence[int]
) -> NDArray[np.float32]:
    """Convert multiple states to a batched tensor.

    Args:
        states: List of game states
        perspectives: List of player indices (one per state)

    Returns:
        Float32 array of shape (batch_size, OBSERVATION_DIM)
    """
    if len(states) != len(perspectives):
        raise ValueError(f"Mismatch: {len(states)} states vs {len(perspectives)} perspectives")

    if not states:
        return np.empty((0, OBSERVATION_DIM), dtype=np.float32)

    tensors = [state_to_tensor(state_obj, persp) for state_obj, persp in zip(states, perspectives, strict=False)]
    return np.stack(tensors, axis=0)


def tensor_to_observation(tensor: NDArray[np.float32], perspective: int) -> dict[str, object]:
    """Decode a tensor back into human-readable observation dict (for debugging).

    Args:
        tensor: Flattened observation tensor
        perspective: Player index

    Returns:
        Dictionary with decoded observation information
    """
    if tensor.shape != (OBSERVATION_DIM,):
        raise ValueError(f"Expected tensor shape ({OBSERVATION_DIM},), got {tensor.shape}")

    result: dict[str, object] = {"perspective": perspective}
    offset = 0

    # Decode queue (5 x 17)
    queue_cards = []
    for _i in range(rules.MAX_QUEUE_LENGTH):
        card_tensor = tensor[offset : offset + _CARD_FEATURE_DIM]
        offset += _CARD_FEATURE_DIM

        present = card_tensor[0]
        if present > 0.5:
            owner_val = card_tensor[1]
            owner = perspective if owner_val > 0.7 else (1 - perspective if owner_val < 0.3 else -1)
            species_onehot = card_tensor[2 : 2 + _NUM_SPECIES]
            species_idx = int(np.argmax(species_onehot)) if species_onehot.max() > 0.5 else -1
            strength = card_tensor[2 + _NUM_SPECIES] * _MAX_STRENGTH
            points = card_tensor[3 + _NUM_SPECIES] * _MAX_POINTS
            position = card_tensor[4 + _NUM_SPECIES]

            queue_cards.append(
                {
                    "present": True,
                    "owner": owner,
                    "species_idx": species_idx,
                    "strength": strength,
                    "points": points,
                    "position": position,
                }
            )
    result["queue"] = queue_cards

    # Decode Beasty Bar (24 x 17)
    beasty_bar_cards = []
    for _i in range(_TOTAL_CARD_SLOTS_IN_ZONES):
        card_tensor = tensor[offset : offset + _CARD_FEATURE_DIM]
        offset += _CARD_FEATURE_DIM

        present = card_tensor[0]
        if present > 0.5:
            owner_val = card_tensor[1]
            owner = perspective if owner_val > 0.7 else (1 - perspective if owner_val < 0.3 else -1)
            species_onehot = card_tensor[2 : 2 + _NUM_SPECIES]
            species_idx = int(np.argmax(species_onehot)) if species_onehot.max() > 0.5 else -1
            beasty_bar_cards.append({"present": True, "owner": owner, "species_idx": species_idx})
    result["beasty_bar"] = beasty_bar_cards

    # Decode THAT'S IT (24 x 17)
    thats_it_cards = []
    for _i in range(_TOTAL_CARD_SLOTS_IN_ZONES):
        card_tensor = tensor[offset : offset + _CARD_FEATURE_DIM]
        offset += _CARD_FEATURE_DIM

        present = card_tensor[0]
        if present > 0.5:
            owner_val = card_tensor[1]
            owner = perspective if owner_val > 0.7 else (1 - perspective if owner_val < 0.3 else -1)
            species_onehot = card_tensor[2 : 2 + _NUM_SPECIES]
            species_idx = int(np.argmax(species_onehot)) if species_onehot.max() > 0.5 else -1
            thats_it_cards.append({"present": True, "owner": owner, "species_idx": species_idx})
    result["thats_it"] = thats_it_cards

    # Decode own hand (4 x 17)
    own_hand_cards = []
    for _i in range(rules.HAND_SIZE):
        card_tensor = tensor[offset : offset + _CARD_FEATURE_DIM]
        offset += _CARD_FEATURE_DIM

        present = card_tensor[0]
        if present > 0.5:
            species_onehot = card_tensor[2 : 2 + _NUM_SPECIES]
            species_idx = int(np.argmax(species_onehot)) if species_onehot.max() > 0.5 else -1
            own_hand_cards.append({"present": True, "species_idx": species_idx})
    result["own_hand"] = own_hand_cards

    # Decode opponent hand (4 x 3)
    opponent_hand_count = 0
    for _i in range(rules.HAND_SIZE):
        masked_tensor = tensor[offset : offset + _MASKED_CARD_FEATURE_DIM]
        offset += _MASKED_CARD_FEATURE_DIM

        present = masked_tensor[0]
        if present > 0.5:
            opponent_hand_count += 1
    result["opponent_hand_count"] = opponent_hand_count

    # Decode scalars (7)
    scalars = tensor[offset : offset + _SCALARS_DIM]
    result["own_deck_count"] = scalars[0] * rules.DECK_SIZE
    result["opponent_deck_count"] = scalars[1] * rules.DECK_SIZE
    result["own_hand_count"] = scalars[2] * rules.HAND_SIZE
    result["opponent_hand_count_scalar"] = scalars[3] * rules.HAND_SIZE
    result["is_active_player"] = scalars[4] > 0.5
    result["turn_normalized"] = scalars[5]
    result["queue_length_normalized"] = scalars[6]

    return result


__all__ = [
    "OBSERVATION_DIM",
    "CardEncoding",
    "Observation",
    "batch_states_to_tensor",
    "build_observation",
    "observation_to_tensor",
    "species_index",
    "species_name",
    "state_to_tensor",
    "tensor_to_observation",
]
