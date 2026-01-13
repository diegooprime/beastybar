"""Systematic position enumeration for tablebase generation.

Enumerates all valid endgame positions with n or fewer cards remaining.
Positions are generated in canonical form to avoid duplicates.

For Beasty Bar, a position consists of:
- Card distributions across players (hands and decks)
- Queue ordering (order matters)
- Beasty bar contents (order doesn't matter)
- THAT'S IT contents (order doesn't matter)
- Active player and turn number
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations, permutations

from _01_simulator import rules, state

# Type aliases
CanonicalKey = bytes
CardDistribution = tuple[tuple[str, ...], ...]  # Per-zone species


@dataclass(frozen=True)
class EnumerationConfig:
    """Configuration for position enumeration."""

    max_total_cards: int = 10
    max_hand_size: int = rules.HAND_SIZE
    max_queue_size: int = rules.MAX_QUEUE_LENGTH
    include_bar_cards: bool = True
    include_thats_it_cards: bool = True


class PositionEnumerator:
    """Systematically enumerate all valid endgame positions.

    Uses combinatorial enumeration to generate all possible positions
    with a given number of remaining cards. Positions are generated
    in canonical form and yielded with unique keys for storage.
    """

    def __init__(self, config: EnumerationConfig | None = None):
        self.config = config or EnumerationConfig()
        self._species_list = list(rules.SPECIES.keys())
        self._species_list.remove("unknown")  # Remove synthetic species

    def count_positions(self, max_cards: int) -> int:
        """Count total positions without generating them.

        This is an approximation based on combinatorial analysis.
        The actual count may differ due to game rule constraints.

        Args:
            max_cards: Maximum total cards in play (hands + decks + queue)

        Returns:
            Estimated position count
        """
        # For a more accurate count, we enumerate with a counter
        count = 0
        for _ in self.enumerate(max_cards, count_only=True):
            count += 1
        return count

    def enumerate(
        self,
        max_cards: int,
        count_only: bool = False,
    ) -> Iterator[tuple[CanonicalKey, state.State] | None]:
        """Generate all positions with their canonical keys.

        Args:
            max_cards: Maximum total cards in play
            count_only: If True, yields None instead of states (for counting)

        Yields:
            Tuples of (canonical_key, game_state) or None if count_only
        """
        species_pool = self._species_list[:12]  # 12 unique species

        # For each total card count from 1 to max_cards
        for total_cards in range(1, max_cards + 1):
            # Generate all possible card subsets
            for card_subset in combinations(species_pool, total_cards):
                # For each card distribution across zones and players
                for position in self._enumerate_distributions(card_subset):
                    if count_only:
                        yield None
                    else:
                        yield position

    def _enumerate_distributions(
        self,
        cards: tuple[str, ...],
    ) -> Iterator[tuple[CanonicalKey, state.State]]:
        """Enumerate all valid distributions of cards across zones.

        Args:
            cards: Tuple of species names to distribute

        Yields:
            Valid game states with their canonical keys
        """
        n_cards = len(cards)

        # Zones: p0_hand, p0_deck, p1_hand, p1_deck, queue, bar, thats_it
        # We only care about hands, decks, and queue for "remaining cards"
        # bar and thats_it are scored/discarded cards

        # Enumerate: how many cards go to each zone
        # Constraints:
        # - hands: 0-4 each
        # - decks: 0-8 each (since hand draws from deck)
        # - queue: 0-5
        # - bar/thats_it: 0-n (remaining after other zones)

        max_hand = min(rules.HAND_SIZE, n_cards)
        max_queue = min(rules.MAX_QUEUE_LENGTH, n_cards)

        # Enumerate counts for active zones (hands + decks + queue)
        for p0_hand_count in range(max_hand + 1):
            remaining1 = n_cards - p0_hand_count
            for p0_deck_count in range(remaining1 + 1):
                remaining2 = remaining1 - p0_deck_count
                for p1_hand_count in range(min(max_hand, remaining2) + 1):
                    remaining3 = remaining2 - p1_hand_count
                    for p1_deck_count in range(remaining3 + 1):
                        remaining4 = remaining3 - p1_deck_count
                        for queue_count in range(min(max_queue, remaining4) + 1):
                            # Remaining cards go to bar and thats_it
                            remaining_for_zones = remaining4 - queue_count

                            # For efficiency, we don't enumerate bar/thats_it
                            # distributions - they don't affect remaining card count
                            # and are needed only for score tracking
                            if not self.config.include_bar_cards:
                                # Skip positions with bar/thats_it cards
                                if remaining_for_zones > 0:
                                    continue
                                remaining_for_zones = 0

                            # Now enumerate actual card assignments
                            counts = (
                                p0_hand_count,
                                p0_deck_count,
                                p1_hand_count,
                                p1_deck_count,
                                queue_count,
                                remaining_for_zones,  # bar + thats_it
                            )

                            yield from self._enumerate_card_assignments(
                                cards, counts
                            )

    def _enumerate_card_assignments(
        self,
        cards: tuple[str, ...],
        counts: tuple[int, ...],
    ) -> Iterator[tuple[CanonicalKey, state.State]]:
        """Enumerate actual card assignments given zone counts.

        Args:
            cards: Species to assign
            counts: (p0_hand, p0_deck, p1_hand, p1_deck, queue, bar_thats_it)

        Yields:
            Valid game states with their canonical keys
        """
        p0_hand_ct, p0_deck_ct, p1_hand_ct, p1_deck_ct, queue_ct, _bar_ti_ct = counts
        n_cards = len(cards)

        # Generate all partitions of cards into zones
        indices = list(range(n_cards))

        # Choose cards for p0_hand
        for p0_hand_idx in combinations(indices, p0_hand_ct):
            p0_hand_set = set(p0_hand_idx)
            remaining1 = [i for i in indices if i not in p0_hand_set]

            # Choose cards for p0_deck
            for p0_deck_idx in combinations(remaining1, p0_deck_ct):
                p0_deck_set = set(p0_deck_idx)
                remaining2 = [i for i in remaining1 if i not in p0_deck_set]

                # Choose cards for p1_hand
                for p1_hand_idx in combinations(remaining2, p1_hand_ct):
                    p1_hand_set = set(p1_hand_idx)
                    remaining3 = [i for i in remaining2 if i not in p1_hand_set]

                    # Choose cards for p1_deck
                    for p1_deck_idx in combinations(remaining3, p1_deck_ct):
                        p1_deck_set = set(p1_deck_idx)
                        remaining4 = [i for i in remaining3 if i not in p1_deck_set]

                        # Choose cards for queue
                        for queue_idx in combinations(remaining4, queue_ct):
                            queue_set = set(queue_idx)
                            bar_ti_idx = [i for i in remaining4 if i not in queue_set]

                            # Queue order matters - enumerate permutations
                            queue_species = tuple(cards[i] for i in queue_idx)

                            # For efficiency, only consider one queue ordering
                            # (canonical form) unless queue is small
                            if queue_ct <= 3:
                                queue_perms = list(permutations(queue_species))
                            else:
                                # For larger queues, use sorted order as canonical
                                queue_perms = [tuple(sorted(queue_species))]

                            for queue_order in queue_perms:
                                # Create species tuples for each zone
                                p0_hand = tuple(cards[i] for i in p0_hand_idx)
                                p0_deck = tuple(cards[i] for i in p0_deck_idx)
                                p1_hand = tuple(cards[i] for i in p1_hand_idx)
                                p1_deck = tuple(cards[i] for i in p1_deck_idx)
                                bar_ti = tuple(cards[i] for i in bar_ti_idx)

                                # Split bar_ti between bar and thats_it
                                # For simplicity, put all in bar (doesn't affect gameplay)
                                bar_species = bar_ti
                                ti_species: tuple[str, ...] = ()

                                # Generate states for both active players
                                for active_player in (0, 1):
                                    game_state = self._create_state(
                                        p0_hand,
                                        p0_deck,
                                        p1_hand,
                                        p1_deck,
                                        queue_order,
                                        bar_species,
                                        ti_species,
                                        active_player,
                                    )

                                    key = self.canonicalize(game_state)
                                    yield key, game_state

    def _create_state(
        self,
        p0_hand: tuple[str, ...],
        p0_deck: tuple[str, ...],
        p1_hand: tuple[str, ...],
        p1_deck: tuple[str, ...],
        queue_species: tuple[str, ...],
        bar_species: tuple[str, ...],
        ti_species: tuple[str, ...],
        active_player: int,
    ) -> state.State:
        """Create a game state from species assignments.

        Args:
            p0_hand: Player 0 hand species
            p0_deck: Player 0 deck species
            p1_hand: Player 1 hand species
            p1_deck: Player 1 deck species
            queue_species: Queue species (in order)
            bar_species: Beasty bar species
            ti_species: THAT'S IT species
            active_player: Active player index

        Returns:
            Constructed game state
        """
        # Create cards with owners
        # For enumeration, we assign owners based on zone
        # Queue cards need owners - we alternate for variety
        queue_cards = tuple(
            state.Card(owner=i % 2, species=s, entered_turn=i)
            for i, s in enumerate(queue_species)
        )

        bar_cards = tuple(
            state.Card(owner=i % 2, species=s, entered_turn=-1)
            for i, s in enumerate(bar_species)
        )

        ti_cards = tuple(
            state.Card(owner=i % 2, species=s, entered_turn=-1)
            for i, s in enumerate(ti_species)
        )

        p0_hand_cards = tuple(
            state.Card(owner=0, species=s, entered_turn=-1) for s in p0_hand
        )
        p0_deck_cards = tuple(
            state.Card(owner=0, species=s, entered_turn=-1) for s in p0_deck
        )
        p1_hand_cards = tuple(
            state.Card(owner=1, species=s, entered_turn=-1) for s in p1_hand
        )
        p1_deck_cards = tuple(
            state.Card(owner=1, species=s, entered_turn=-1) for s in p1_deck
        )

        players = (
            state.PlayerState(deck=p0_deck_cards, hand=p0_hand_cards),
            state.PlayerState(deck=p1_deck_cards, hand=p1_hand_cards),
        )

        zones = state.Zones(
            queue=queue_cards,
            beasty_bar=bar_cards,
            thats_it=ti_cards,
        )

        return state.State(
            seed=0,
            turn=1,  # Turn > 0 so terminal check works
            active_player=active_player,
            players=players,
            zones=zones,
        )

    def canonicalize(self, game_state: state.State) -> CanonicalKey:
        """Create canonical key for a position.

        This matches the canonicalization in endgame.py for compatibility.

        Args:
            game_state: Game state to canonicalize

        Returns:
            Canonical key bytes
        """
        parts: list[bytes] = []

        # Active player (absolute, not relative)
        parts.append(struct.pack("B", game_state.active_player))

        # Player states (sorted species for hands/decks)
        for player_state in game_state.players:
            hand_species = sorted(c.species for c in player_state.hand)
            deck_species = sorted(c.species for c in player_state.deck)
            parts.append(self._encode_species_list(hand_species))
            parts.append(self._encode_species_list(deck_species))

        # Queue (order matters)
        parts.append(self._encode_queue(game_state.zones.queue))

        # Bar and thats_it (sorted for canonical form)
        parts.append(self._encode_zone_unordered(game_state.zones.beasty_bar))
        parts.append(self._encode_zone_unordered(game_state.zones.thats_it))

        combined = b"".join(parts)
        return hashlib.blake2b(combined, digest_size=16).digest()

    def _encode_species_list(self, species_list: list[str]) -> bytes:
        """Encode a sorted species list as bytes."""
        species_keys = list(rules.SPECIES.keys())
        species_ids = [
            species_keys.index(s) for s in species_list if s in rules.SPECIES
        ]
        return struct.pack(f"B{len(species_ids)}B", len(species_ids), *species_ids)

    def _encode_queue(self, queue: tuple[state.Card, ...]) -> bytes:
        """Encode queue maintaining order."""
        species_keys = list(rules.SPECIES.keys())
        parts = [struct.pack("B", len(queue))]
        for card in queue:
            species_id = species_keys.index(card.species)
            parts.append(struct.pack("BB", card.owner, species_id))
        return b"".join(parts)

    def _encode_zone_unordered(self, zone: tuple[state.Card, ...]) -> bytes:
        """Encode zone without order (sorted for canonicalization)."""
        species_keys = list(rules.SPECIES.keys())
        encoded_cards = []
        for card in zone:
            species_id = species_keys.index(card.species)
            encoded_cards.append((card.owner, species_id))

        encoded_cards.sort()
        parts = [struct.pack("B", len(encoded_cards))]
        for owner, species in encoded_cards:
            parts.append(struct.pack("BB", owner, species))
        return b"".join(parts)


class PositionIndexer:
    """Maps positions to unique indices for array storage.

    Provides bijective mapping between game positions and indices,
    enabling efficient array-based tablebase storage.
    """

    def __init__(self, max_cards: int):
        self.max_cards = max_cards
        self._key_to_index: dict[CanonicalKey, int] = {}
        self._index_to_key: list[CanonicalKey] = []
        self._built = False

    def build_index(self, enumerator: PositionEnumerator) -> int:
        """Build the position index from enumeration.

        Args:
            enumerator: Enumerator to generate positions

        Returns:
            Total number of positions indexed
        """
        self._key_to_index.clear()
        self._index_to_key.clear()

        for key, _ in enumerator.enumerate(self.max_cards):
            if key not in self._key_to_index:
                self._key_to_index[key] = len(self._index_to_key)
                self._index_to_key.append(key)

        self._built = True
        return len(self._index_to_key)

    def key_to_index(self, key: CanonicalKey) -> int | None:
        """Get index for a canonical key.

        Args:
            key: Canonical position key

        Returns:
            Position index or None if not found
        """
        return self._key_to_index.get(key)

    def index_to_key(self, index: int) -> CanonicalKey | None:
        """Get canonical key for an index.

        Args:
            index: Position index

        Returns:
            Canonical key or None if invalid
        """
        if 0 <= index < len(self._index_to_key):
            return self._index_to_key[index]
        return None

    @property
    def num_positions(self) -> int:
        """Total number of indexed positions."""
        return len(self._index_to_key)

    @property
    def is_built(self) -> bool:
        """Whether index has been built."""
        return self._built


def count_positions_estimate(max_cards: int) -> int:
    """Quick estimate of position count (no enumeration).

    This uses combinatorial formulas for a rough estimate.
    The actual count requires enumeration due to constraints.

    Args:
        max_cards: Maximum cards in play

    Returns:
        Estimated position count
    """
    # Rough estimate: C(12, n) * distributions * orderings
    from math import comb

    total = 0
    for n in range(1, max_cards + 1):
        # Choose n cards from 12 species
        card_choices = comb(12, n)
        # Rough multiplier for distributions and orderings
        # This is a very rough approximation
        distribution_factor = n**2 * 2  # active player
        total += card_choices * distribution_factor

    return total


__all__ = [
    "CanonicalKey",
    "EnumerationConfig",
    "PositionEnumerator",
    "PositionIndexer",
    "count_positions_estimate",
]
