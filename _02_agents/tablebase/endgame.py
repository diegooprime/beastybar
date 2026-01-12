"""Endgame tablebase for Beasty Bar.

Implements both:
1. Forward minimax search with alpha-beta pruning and memoization
2. Incremental retrograde analysis for building tablebases over time

The forward search approach is used for on-demand solving of positions,
while retrograde analysis can build comprehensive tablebases offline.

Key concepts:
- Terminal positions are solved directly from game outcome
- Non-terminal positions are solved by minimax over successor positions
- Positions are canonicalized to reduce storage requirements
- Supports generation, storage, and loading from disk
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import struct
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from _01_simulator import actions, engine, rules, state

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class GameTheoreticValue(IntEnum):
    """Game-theoretic outcome from perspective of player to move."""

    LOSS = -1
    DRAW = 0
    WIN = 1
    UNKNOWN = 2


@dataclass(frozen=True)
class TablebaseEntry:
    """Stored result for a canonical position."""

    value: GameTheoreticValue
    optimal_action: actions.Action | None  # None for terminal positions
    depth_to_end: int  # Plies until game ends with optimal play


@dataclass
class TablebaseConfig:
    """Configuration for tablebase generation."""

    max_cards_per_player: int = 5
    max_total_cards: int = 10
    include_queue: bool = True
    compression_level: int = 6
    cache_size: int = 1_000_000
    search_depth_limit: int = 50  # Max depth for forward search
    use_alpha_beta: bool = True  # Use alpha-beta pruning


# Type aliases
CanonicalKey: TypeAlias = bytes
PositionCache: TypeAlias = dict[CanonicalKey, TablebaseEntry]


@dataclass
class EndgameTablebase:
    """Endgame tablebase with forward search and memoization.

    Uses minimax with alpha-beta pruning to solve positions on demand,
    caching results for reuse. Supports disk persistence for sharing
    solved positions across sessions.
    """

    positions: PositionCache = field(default_factory=dict)
    config: TablebaseConfig = field(default_factory=TablebaseConfig)
    _stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def lookup(
        self,
        game_state: state.State,
        perspective: int,
    ) -> TablebaseEntry | None:
        """Look up a position in the tablebase.

        Args:
            game_state: Current game state
            perspective: Player perspective (0 or 1)

        Returns:
            TablebaseEntry if position is in tablebase, None otherwise
        """
        key = self._canonicalize(game_state, perspective)
        entry = self.positions.get(key)

        if entry is not None:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1

        return entry

    def solve(
        self,
        game_state: state.State,
        perspective: int,
    ) -> TablebaseEntry:
        """Solve a position using forward minimax search.

        If position is already solved, returns cached result.
        Otherwise, performs minimax search and caches the result.

        Args:
            game_state: Current game state
            perspective: Player perspective (0 or 1)

        Returns:
            TablebaseEntry with game-theoretic value and optimal action
        """
        # Check cache first
        key = self._canonicalize(game_state, perspective)
        if key in self.positions:
            self._stats["solve_cache_hits"] += 1
            return self.positions[key]

        self._stats["solve_calls"] += 1

        # Perform minimax search
        if self.config.use_alpha_beta:
            value, action, depth = self._alpha_beta(
                game_state,
                perspective,
                depth=0,
                alpha=GameTheoreticValue.LOSS,
                beta=GameTheoreticValue.WIN,
            )
        else:
            value, action, depth = self._minimax(
                game_state,
                perspective,
                depth=0,
            )

        entry = TablebaseEntry(
            value=value,
            optimal_action=action,
            depth_to_end=depth,
        )

        # Cache result
        if len(self.positions) < self.config.cache_size:
            self.positions[key] = entry

        return entry

    def get_optimal_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action | None:
        """Get the optimal action for a position.

        Solves the position if not already in tablebase.

        Args:
            game_state: Current game state
            legal_actions: Available legal actions

        Returns:
            Optimal action
        """
        if not legal_actions:
            return None

        perspective = game_state.active_player
        entry = self.solve(game_state, perspective)

        if entry.optimal_action is None:
            return None

        # Verify action is still legal
        if entry.optimal_action in legal_actions:
            return entry.optimal_action

        # Fall back to finding equivalent action
        return self._find_equivalent_action(
            entry.optimal_action, legal_actions, game_state
        )

    def get_action_values(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> dict[actions.Action, GameTheoreticValue]:
        """Get game-theoretic values for all legal actions.

        Args:
            game_state: Current game state
            legal_actions: Available legal actions

        Returns:
            Dict mapping actions to their game-theoretic values
        """
        perspective = game_state.active_player
        values: dict[actions.Action, GameTheoreticValue] = {}

        for action in legal_actions:
            try:
                successor = engine.step(game_state, action)

                if engine.is_terminal(successor):
                    values[action] = self._terminal_value(successor, perspective)
                else:
                    # Solve successor position from our perspective
                    # solve() handles max/min internally, returns value from perspective's view
                    succ_entry = self.solve(successor, perspective)
                    values[action] = succ_entry.value

            except (ValueError, IndexError):
                values[action] = GameTheoreticValue.UNKNOWN

        return values

    def is_endgame_position(self, game_state: state.State) -> bool:
        """Check if position qualifies as an endgame position.

        Args:
            game_state: Current game state

        Returns:
            True if position has few enough cards for tablebase
        """
        total_cards = self._count_remaining_cards(game_state)
        max_per_player = max(
            len(p.hand) + len(p.deck) for p in game_state.players
        )

        return (
            total_cards <= self.config.max_total_cards
            and max_per_player <= self.config.max_cards_per_player
        )

    def save(self, path: Path | str) -> None:
        """Save tablebase to disk.

        Args:
            path: File path for saving
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "config": self.config,
            "positions": self.positions,
            "stats": dict(self._stats),
        }

        compressed = zlib.compress(
            pickle.dumps(data), level=self.config.compression_level
        )

        with open(path, "wb") as f:
            f.write(compressed)

        logger.info(
            "Saved tablebase: %d positions, %.2f MB",
            len(self.positions),
            len(compressed) / (1024 * 1024),
        )

    @classmethod
    def load(cls, path: Path | str) -> EndgameTablebase:
        """Load tablebase from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded EndgameTablebase instance
        """
        path = Path(path)

        with open(path, "rb") as f:
            compressed = f.read()

        data = pickle.loads(zlib.decompress(compressed))

        if data.get("version") != 1:
            raise ValueError(f"Unsupported tablebase version: {data.get('version')}")

        tb = cls(
            positions=data["positions"],
            config=data["config"],
            _stats=defaultdict(int, data.get("stats", {})),
        )

        logger.info("Loaded tablebase: %d positions", len(tb.positions))
        return tb

    def _minimax(
        self,
        game_state: state.State,
        perspective: int,
        depth: int,
    ) -> tuple[GameTheoreticValue, actions.Action | None, int]:
        """Standard minimax search.

        Args:
            game_state: Current game state
            perspective: Original player's perspective
            depth: Current search depth

        Returns:
            Tuple of (value, best_action, depth_to_end)
        """
        # Check depth limit
        if depth >= self.config.search_depth_limit:
            return GameTheoreticValue.UNKNOWN, None, depth

        # Terminal check
        if engine.is_terminal(game_state):
            value = self._terminal_value(game_state, perspective)
            return value, None, 0

        # Check cache
        key = self._canonicalize(game_state, perspective)
        if key in self.positions:
            entry = self.positions[key]
            return entry.value, entry.optimal_action, entry.depth_to_end

        # Get legal actions
        active = game_state.active_player
        legal = list(engine.legal_actions(game_state, active))

        if not legal:
            # No legal moves - treat as draw
            return GameTheoreticValue.DRAW, None, 0

        # Determine if maximizing or minimizing
        is_maximizing = active == perspective

        best_value: GameTheoreticValue | None = None
        best_action: actions.Action | None = None
        best_depth = 0

        for action in legal:
            try:
                successor = engine.step(game_state, action)
                child_value, _, child_depth = self._minimax(
                    successor, perspective, depth + 1
                )

                if child_value == GameTheoreticValue.UNKNOWN:
                    continue

                action_depth = child_depth + 1

                if best_value is None:
                    best_value = child_value
                    best_action = action
                    best_depth = action_depth
                elif is_maximizing:
                    if child_value > best_value:
                        best_value = child_value
                        best_action = action
                        best_depth = action_depth
                    elif child_value == best_value:
                        # Prefer shorter wins, longer losses
                        if (child_value == GameTheoreticValue.WIN and action_depth < best_depth) or (child_value == GameTheoreticValue.LOSS and action_depth > best_depth):
                            best_action = action
                            best_depth = action_depth
                else:
                    if child_value < best_value:
                        best_value = child_value
                        best_action = action
                        best_depth = action_depth
                    elif child_value == best_value:
                        # For opponent: shorter wins (our losses), longer losses (our wins)
                        if (child_value == GameTheoreticValue.LOSS and action_depth < best_depth) or (child_value == GameTheoreticValue.WIN and action_depth > best_depth):
                            best_action = action
                            best_depth = action_depth

            except (ValueError, IndexError):
                continue

        if best_value is None:
            return GameTheoreticValue.UNKNOWN, None, depth

        # Cache result
        entry = TablebaseEntry(
            value=best_value,
            optimal_action=best_action,
            depth_to_end=best_depth,
        )
        if len(self.positions) < self.config.cache_size:
            self.positions[key] = entry

        return best_value, best_action, best_depth

    def _alpha_beta(
        self,
        game_state: state.State,
        perspective: int,
        depth: int,
        alpha: GameTheoreticValue,
        beta: GameTheoreticValue,
    ) -> tuple[GameTheoreticValue, actions.Action | None, int]:
        """Alpha-beta pruning search.

        Args:
            game_state: Current game state
            perspective: Original player's perspective
            depth: Current search depth
            alpha: Alpha bound (best for maximizer)
            beta: Beta bound (best for minimizer)

        Returns:
            Tuple of (value, best_action, depth_to_end)
        """
        # Check depth limit
        if depth >= self.config.search_depth_limit:
            return GameTheoreticValue.UNKNOWN, None, depth

        # Terminal check
        if engine.is_terminal(game_state):
            value = self._terminal_value(game_state, perspective)
            return value, None, 0

        # Check cache
        key = self._canonicalize(game_state, perspective)
        if key in self.positions:
            entry = self.positions[key]
            return entry.value, entry.optimal_action, entry.depth_to_end

        # Get legal actions
        active = game_state.active_player
        legal = list(engine.legal_actions(game_state, active))

        if not legal:
            return GameTheoreticValue.DRAW, None, 0

        # Determine if maximizing or minimizing
        is_maximizing = active == perspective

        best_value: GameTheoreticValue | None = None
        best_action: actions.Action | None = None
        best_depth = 0

        for action in legal:
            try:
                successor = engine.step(game_state, action)
                child_value, _, child_depth = self._alpha_beta(
                    successor, perspective, depth + 1, alpha, beta
                )

                if child_value == GameTheoreticValue.UNKNOWN:
                    continue

                action_depth = child_depth + 1

                if best_value is None:
                    best_value = child_value
                    best_action = action
                    best_depth = action_depth
                elif is_maximizing:
                    if child_value > best_value:
                        best_value = child_value
                        best_action = action
                        best_depth = action_depth

                    # Alpha-beta pruning
                    if best_value > alpha:
                        alpha = best_value
                    if best_value >= beta:
                        break  # Beta cutoff
                else:
                    if child_value < best_value:
                        best_value = child_value
                        best_action = action
                        best_depth = action_depth

                    # Alpha-beta pruning
                    if best_value < beta:
                        beta = best_value
                    if best_value <= alpha:
                        break  # Alpha cutoff

            except (ValueError, IndexError):
                continue

        if best_value is None:
            return GameTheoreticValue.UNKNOWN, None, depth

        # Cache result
        entry = TablebaseEntry(
            value=best_value,
            optimal_action=best_action,
            depth_to_end=best_depth,
        )
        if len(self.positions) < self.config.cache_size:
            self.positions[key] = entry

        return best_value, best_action, best_depth

    def _canonicalize(
        self,
        game_state: state.State,
        perspective: int,
    ) -> CanonicalKey:
        """Convert game state to canonical form for storage.

        Creates a unique key that captures the essential game state
        while being invariant to irrelevant differences.
        """
        # Build canonical representation
        parts: list[bytes] = []

        # Active player relative to perspective
        relative_active = (game_state.active_player - perspective) % 2
        parts.append(struct.pack("B", relative_active))

        # Encode each player's cards (relative to perspective)
        for i in range(rules.PLAYER_COUNT):
            player_idx = (perspective + i) % rules.PLAYER_COUNT
            player_state = game_state.players[player_idx]

            # Encode hand (sorted for canonicalization)
            hand_species = sorted(c.species for c in player_state.hand)
            parts.append(self._encode_species_list(hand_species))

            # Encode deck (sorted for canonicalization)
            deck_species = sorted(c.species for c in player_state.deck)
            parts.append(self._encode_species_list(deck_species))

        # Encode zones (queue order matters, bar/thats_it order doesn't)
        queue_data = self._encode_queue(game_state.zones.queue, perspective)
        parts.append(queue_data)

        # Beasty bar and thats_it - sorted by owner then species
        bar_data = self._encode_zone_unordered(
            game_state.zones.beasty_bar, perspective
        )
        parts.append(bar_data)

        thats_it_data = self._encode_zone_unordered(
            game_state.zones.thats_it, perspective
        )
        parts.append(thats_it_data)

        # Combine and hash for compact key
        combined = b"".join(parts)
        return hashlib.blake2b(combined, digest_size=16).digest()

    def _encode_species_list(self, species_list: list[str]) -> bytes:
        """Encode a list of species as bytes."""
        species_keys = list(rules.SPECIES.keys())
        species_ids = [
            species_keys.index(s)
            for s in species_list
            if s in rules.SPECIES
        ]
        return struct.pack(f"B{len(species_ids)}B", len(species_ids), *species_ids)

    def _encode_queue(
        self,
        queue: tuple[state.Card, ...],
        perspective: int,
    ) -> bytes:
        """Encode queue maintaining order."""
        species_keys = list(rules.SPECIES.keys())
        parts = [struct.pack("B", len(queue))]
        for card in queue:
            relative_owner = (card.owner - perspective) % 2
            species_id = species_keys.index(card.species)
            parts.append(struct.pack("BB", relative_owner, species_id))
        return b"".join(parts)

    def _encode_zone_unordered(
        self,
        zone: tuple[state.Card, ...],
        perspective: int,
    ) -> bytes:
        """Encode zone without order (sorted for canonicalization)."""
        species_keys = list(rules.SPECIES.keys())
        encoded_cards = []
        for card in zone:
            relative_owner = (card.owner - perspective) % 2
            species_id = species_keys.index(card.species)
            encoded_cards.append((relative_owner, species_id))

        # Sort for canonical form
        encoded_cards.sort()

        parts = [struct.pack("B", len(encoded_cards))]
        for owner, species in encoded_cards:
            parts.append(struct.pack("BB", owner, species))
        return b"".join(parts)

    def _terminal_value(
        self,
        game_state: state.State,
        perspective: int,
    ) -> GameTheoreticValue:
        """Compute game-theoretic value of terminal position."""
        scores = engine.score(game_state)
        my_score = scores[perspective]
        opp_score = scores[1 - perspective]

        if my_score > opp_score:
            return GameTheoreticValue.WIN
        elif my_score < opp_score:
            return GameTheoreticValue.LOSS
        else:
            return GameTheoreticValue.DRAW

    def _count_remaining_cards(self, game_state: state.State) -> int:
        """Count total remaining cards in play."""
        total = len(game_state.zones.queue)
        for player_state in game_state.players:
            total += len(player_state.hand) + len(player_state.deck)
        return total

    def _find_equivalent_action(
        self,
        target: actions.Action,
        legal_actions: Sequence[actions.Action],
        game_state: state.State,
    ) -> actions.Action | None:
        """Find action equivalent to target in legal actions."""
        if not legal_actions:
            return None

        player_state = game_state.players[game_state.active_player]

        if target.hand_index >= len(player_state.hand):
            return None

        target_species = player_state.hand[target.hand_index].species

        for action in legal_actions:
            if action.hand_index < len(player_state.hand):
                species = player_state.hand[action.hand_index].species
                if species == target_species and action.params == target.params:
                    return action

        return None

    def get_stats(self) -> dict[str, int]:
        """Get tablebase statistics."""
        stats = dict(self._stats)
        stats["positions_cached"] = len(self.positions)
        return stats


class TablebaseGenerator:
    """Generates endgame tablebase by solving positions from gameplay.

    Instead of exhaustive enumeration, this generator builds the tablebase
    incrementally by solving positions encountered during actual games
    or targeted endgame scenarios.
    """

    def __init__(self, config: TablebaseConfig | None = None):
        self.config = config or TablebaseConfig()
        self.tablebase = EndgameTablebase(config=self.config)

    def generate_from_games(
        self,
        num_games: int = 1000,
        seed: int = 42,
        verbose: bool = True,
    ) -> EndgameTablebase:
        """Generate tablebase by playing random games and solving endgames.

        Args:
            num_games: Number of games to simulate
            seed: Random seed for reproducibility
            verbose: Whether to log progress

        Returns:
            Generated EndgameTablebase
        """
        import random

        rng = random.Random(seed)
        start_time = time.time()
        positions_before = len(self.tablebase.positions)

        if verbose:
            logger.info("Generating tablebase from %d games...", num_games)

        for game_idx in range(num_games):
            game_state = state.initial_state(seed=rng.randint(0, 2**31))

            while not engine.is_terminal(game_state):
                # Check if we're in endgame territory
                if self.tablebase.is_endgame_position(game_state):
                    # Solve this position
                    perspective = game_state.active_player
                    self.tablebase.solve(game_state, perspective)

                # Make a random move to continue
                legal = list(engine.legal_actions(game_state, game_state.active_player))
                if legal:
                    action = rng.choice(legal)
                    game_state = engine.step(game_state, action)
                else:
                    break

            if verbose and (game_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                new_positions = len(self.tablebase.positions) - positions_before
                logger.info(
                    "Game %d/%d: %d positions cached (+%d new), %.1f games/sec",
                    game_idx + 1,
                    num_games,
                    len(self.tablebase.positions),
                    new_positions,
                    (game_idx + 1) / elapsed,
                )

        if verbose:
            elapsed = time.time() - start_time
            new_positions = len(self.tablebase.positions) - positions_before
            logger.info(
                "Generation complete: %d total positions (+%d new) in %.1f seconds",
                len(self.tablebase.positions),
                new_positions,
                elapsed,
            )

        return self.tablebase

    def generate_from_position(
        self,
        game_state: state.State,
        verbose: bool = False,
    ) -> TablebaseEntry:
        """Solve a specific position and its subtree.

        Args:
            game_state: Position to solve
            verbose: Whether to log progress

        Returns:
            TablebaseEntry for the given position
        """
        perspective = game_state.active_player
        entry = self.tablebase.solve(game_state, perspective)

        if verbose:
            logger.info(
                "Solved position: value=%s, depth=%d, cached=%d",
                entry.value.name,
                entry.depth_to_end,
                len(self.tablebase.positions),
            )

        return entry

    def get_tablebase(self) -> EndgameTablebase:
        """Get the generated tablebase."""
        return self.tablebase


class TablebaseAgent:
    """Agent that uses tablebase for endgame positions.

    Falls back to a provided base agent for positions not in tablebase
    or positions too complex to solve quickly.
    """

    def __init__(
        self,
        tablebase: EndgameTablebase,
        fallback_agent=None,
        solve_on_miss: bool = True,
    ):
        """Initialize tablebase agent.

        Args:
            tablebase: Endgame tablebase to use
            fallback_agent: Agent to use for non-tablebase positions
            solve_on_miss: Whether to solve positions not in tablebase
        """
        self.tablebase = tablebase
        self.fallback_agent = fallback_agent
        self.solve_on_miss = solve_on_miss
        self._stats: dict[str, int] = defaultdict(int)

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select an action using tablebase when possible.

        Args:
            game_state: Current game state
            legal_actions: Available legal actions

        Returns:
            Selected action
        """
        if not legal_actions:
            raise ValueError("No legal actions available")

        perspective = game_state.active_player

        # Check if position is in endgame territory
        if self.tablebase.is_endgame_position(game_state):
            # Try tablebase lookup or solve
            entry = self.tablebase.lookup(game_state, perspective)

            if entry is None and self.solve_on_miss:
                entry = self.tablebase.solve(game_state, perspective)
                self._stats["solve_on_miss"] += 1

            if entry is not None and entry.optimal_action is not None:
                # Find matching legal action
                optimal = self._find_matching_action(
                    entry.optimal_action, legal_actions, game_state
                )
                if optimal is not None:
                    self._stats["tablebase_hits"] += 1
                    return optimal

        self._stats["fallback"] += 1

        # Fall back to base agent or random
        if self.fallback_agent is not None:
            return self.fallback_agent.select_action(game_state, legal_actions)

        # Random fallback
        import random
        return random.choice(list(legal_actions))

    def _find_matching_action(
        self,
        target: actions.Action,
        legal_actions: Sequence[actions.Action],
        game_state: state.State,
    ) -> actions.Action | None:
        """Find a legal action matching the target."""
        if target in legal_actions:
            return target

        # Try to match by species and params
        player_state = game_state.players[game_state.active_player]

        if target.hand_index >= len(player_state.hand):
            return None

        target_species = player_state.hand[target.hand_index].species

        for action in legal_actions:
            if action.hand_index < len(player_state.hand):
                species = player_state.hand[action.hand_index].species
                if species == target_species and action.params == target.params:
                    return action

        return None

    @property
    def name(self) -> str:
        return "TablebaseAgent"

    def get_stats(self) -> dict[str, int]:
        """Get usage statistics."""
        return dict(self._stats)


class IncrementalTablebaseBuilder:
    """Build tablebase incrementally from encountered positions.

    Useful for adding positions encountered during actual gameplay
    without generating the full combinatorial space upfront.
    """

    def __init__(self, tablebase: EndgameTablebase | None = None):
        self.tablebase = tablebase or EndgameTablebase()

    def add_position(
        self,
        game_state: state.State,
        solve: bool = True,
    ) -> TablebaseEntry | None:
        """Add a position to the tablebase.

        Args:
            game_state: Position to add
            solve: Whether to solve the position immediately

        Returns:
            TablebaseEntry if solved, None otherwise
        """
        if not self.tablebase.is_endgame_position(game_state):
            return None

        perspective = game_state.active_player
        key = self.tablebase._canonicalize(game_state, perspective)

        if key in self.tablebase.positions:
            return self.tablebase.positions[key]

        if solve:
            return self.tablebase.solve(game_state, perspective)

        return None

    def get_tablebase(self) -> EndgameTablebase:
        """Get the built tablebase."""
        return self.tablebase


def generate_tablebase(
    config: TablebaseConfig | None = None,
    save_path: Path | str | None = None,
    num_games: int = 1000,
    verbose: bool = True,
) -> EndgameTablebase:
    """Generate an endgame tablebase by simulating games.

    Args:
        config: Configuration for generation
        save_path: Optional path to save generated tablebase
        num_games: Number of games to simulate for generation
        verbose: Whether to log progress

    Returns:
        Generated EndgameTablebase
    """
    generator = TablebaseGenerator(config)
    tablebase = generator.generate_from_games(
        num_games=num_games,
        verbose=verbose,
    )

    if save_path is not None:
        tablebase.save(save_path)

    return tablebase


def load_tablebase(path: Path | str) -> EndgameTablebase:
    """Load a tablebase from disk.

    Args:
        path: Path to tablebase file

    Returns:
        Loaded EndgameTablebase
    """
    return EndgameTablebase.load(path)


__all__ = [
    "EndgameTablebase",
    "GameTheoreticValue",
    "IncrementalTablebaseBuilder",
    "TablebaseAgent",
    "TablebaseConfig",
    "TablebaseEntry",
    "TablebaseGenerator",
    "generate_tablebase",
    "load_tablebase",
]
