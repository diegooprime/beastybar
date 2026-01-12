"""Opening book generator and lookup for Beasty Bar.

This module provides pre-computed optimal openings using high-simulation MCTS.
The opening book stores the first 2-3 moves with their visit counts and values,
enabling fast lookup at inference time without running MCTS for known positions.

Key Features:
- Generate opening book from MCTS analysis with configurable simulation counts
- Store position hashes with visit distributions and value estimates
- Efficient O(1) lookup at inference time
- Support for multiple book depths (moves 1-3)
- Save/load to JSON or pickle format

Usage:
    # Generate a new book
    book = OpeningBook()
    generator = OpeningBookGenerator(network, num_simulations=1600)
    generator.generate(book, num_seeds=10000, depth=3)
    book.save("opening_book.json")

    # Use at inference time
    book = OpeningBook.load("opening_book.json")
    if book.has_position(state):
        action_probs = book.lookup(state)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from _01_simulator import action_space, engine
from _01_simulator import state as state_module

if TYPE_CHECKING:

    import torch

    from _02_agents.neural.network import BeastyBarNetwork


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpeningEntry:
    """Single entry in the opening book.

    Stores the MCTS analysis results for a specific game position,
    including visit distribution and value estimate.
    """

    # Action distribution from MCTS (action_index -> probability)
    action_probs: dict[int, float]
    # Visit counts for each action (action_index -> count)
    visit_counts: dict[int, int]
    # Value estimate from MCTS perspective
    value: float
    # Number of MCTS simulations used
    num_simulations: int
    # Game turn when this position occurs
    turn: int

    def best_action(self) -> int:
        """Return action with highest probability."""
        if not self.action_probs:
            raise ValueError("No actions in entry")
        return max(self.action_probs, key=lambda a: self.action_probs[a])

    def sample_action(self, temperature: float = 1.0) -> int:
        """Sample action from distribution with temperature.

        Args:
            temperature: Controls randomness (0=greedy, 1=stochastic)

        Returns:
            Sampled action index
        """
        if not self.action_probs:
            raise ValueError("No actions in entry")

        actions = list(self.action_probs.keys())
        probs = np.array([self.action_probs[a] for a in actions])

        if temperature == 0:
            return actions[int(np.argmax(probs))]

        if temperature != 1.0:
            # Apply temperature scaling
            log_probs = np.log(probs + 1e-10)
            scaled = log_probs / temperature
            exp_scaled = np.exp(scaled - np.max(scaled))
            probs = exp_scaled / np.sum(exp_scaled)

        return int(np.random.choice(actions, p=probs))

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "action_probs": {str(k): v for k, v in self.action_probs.items()},
            "visit_counts": {str(k): v for k, v in self.visit_counts.items()},
            "value": self.value,
            "num_simulations": self.num_simulations,
            "turn": self.turn,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OpeningEntry:
        """Create entry from dictionary."""
        return cls(
            action_probs={int(k): v for k, v in data["action_probs"].items()},
            visit_counts={int(k): v for k, v in data["visit_counts"].items()},
            value=data["value"],
            num_simulations=data["num_simulations"],
            turn=data["turn"],
        )


@dataclass
class BookStats:
    """Statistics about the opening book."""

    total_positions: int = 0
    positions_by_turn: dict[int, int] = field(default_factory=dict)
    avg_value: float = 0.0
    avg_simulations: int = 0
    unique_first_moves: int = 0


class OpeningBook:
    """Opening book for fast lookup of pre-computed optimal moves.

    The book stores MCTS analysis results indexed by position hash,
    allowing O(1) lookup at inference time.

    Position Hashing:
        Positions are hashed based on observable game state features:
        - Active player
        - Turn number
        - Queue contents (species and owners)
        - Both players' hands (sorted by species for player-agnostic hashing)
        - Deck sizes (not contents, which are hidden)

    Thread Safety:
        The book is NOT thread-safe for concurrent writes during generation.
        Use separate generator instances or external synchronization.
    """

    def __init__(self) -> None:
        """Initialize empty opening book."""
        self._entries: dict[str, OpeningEntry] = {}
        self._max_depth: int = 0
        self._num_simulations: int = 0
        self._metadata: dict[str, object] = {}

    def __len__(self) -> int:
        """Return number of positions in the book."""
        return len(self._entries)

    def __contains__(self, game_state: state_module.State) -> bool:
        """Check if position exists in book."""
        return self.has_position(game_state)

    def has_position(self, game_state: state_module.State) -> bool:
        """Check if a position exists in the book.

        Args:
            game_state: Game state to check

        Returns:
            True if position is in the book
        """
        pos_hash = self._hash_position(game_state)
        return pos_hash in self._entries

    def lookup(self, game_state: state_module.State) -> OpeningEntry | None:
        """Look up a position in the book.

        Args:
            game_state: Game state to look up

        Returns:
            OpeningEntry if found, None otherwise
        """
        pos_hash = self._hash_position(game_state)
        return self._entries.get(pos_hash)

    def lookup_best_action(self, game_state: state_module.State) -> int | None:
        """Look up best action for a position.

        Args:
            game_state: Game state to look up

        Returns:
            Best action index if found, None otherwise
        """
        entry = self.lookup(game_state)
        if entry is None:
            return None
        return entry.best_action()

    def add_entry(
        self,
        game_state: state_module.State,
        entry: OpeningEntry,
    ) -> None:
        """Add an entry to the book.

        If an entry already exists for this position, it will be replaced
        if the new entry has more simulations.

        Args:
            game_state: Game state for the entry
            entry: Opening book entry to add
        """
        pos_hash = self._hash_position(game_state)

        # Only replace if new entry has more simulations
        existing = self._entries.get(pos_hash)
        if existing is not None and existing.num_simulations >= entry.num_simulations:
            return

        self._entries[pos_hash] = entry
        self._max_depth = max(self._max_depth, entry.turn + 1)

    def get_stats(self) -> BookStats:
        """Get statistics about the book.

        Returns:
            BookStats with aggregated information
        """
        if not self._entries:
            return BookStats()

        positions_by_turn: dict[int, int] = {}
        total_value = 0.0
        total_sims = 0
        first_move_actions: set[int] = set()

        for entry in self._entries.values():
            turn = entry.turn
            positions_by_turn[turn] = positions_by_turn.get(turn, 0) + 1
            total_value += entry.value
            total_sims += entry.num_simulations

            if turn == 0:
                first_move_actions.add(entry.best_action())

        return BookStats(
            total_positions=len(self._entries),
            positions_by_turn=positions_by_turn,
            avg_value=total_value / len(self._entries),
            avg_simulations=total_sims // len(self._entries),
            unique_first_moves=len(first_move_actions),
        )

    def save(self, path: str | Path, format: str = "json") -> None:
        """Save the opening book to disk.

        Args:
            path: File path to save to
            format: File format ('json' or 'pickle')

        Raises:
            ValueError: If format is not supported
        """
        path = Path(path)

        data = {
            "version": 1,
            "max_depth": self._max_depth,
            "num_simulations": self._num_simulations,
            "metadata": self._metadata,
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
        }

        if format == "json":
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump(data, f, indent=2)
        elif format == "pickle":
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved opening book with {len(self)} positions to {path}")

    @classmethod
    def load(cls, path: str | Path) -> OpeningBook:
        """Load an opening book from disk.

        Automatically detects format based on file extension.

        Args:
            path: File path to load from

        Returns:
            Loaded OpeningBook instance

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If format cannot be determined
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Opening book not found: {path}")

        # Detect format
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r") as f:
                data = json.load(f)
        elif suffix in (".pkl", ".pickle"):
            with path.open("rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Cannot determine format from extension: {suffix}")

        book = cls()
        book._max_depth = data.get("max_depth", 0)
        book._num_simulations = data.get("num_simulations", 0)
        book._metadata = data.get("metadata", {})
        book._entries = {
            k: OpeningEntry.from_dict(v) for k, v in data["entries"].items()
        }

        logger.info(f"Loaded opening book with {len(book)} positions from {path}")
        return book

    def merge(self, other: OpeningBook) -> None:
        """Merge another book into this one.

        Entries from the other book are added if they don't exist
        or have more simulations.

        Args:
            other: Book to merge from
        """
        for pos_hash, entry in other._entries.items():
            existing = self._entries.get(pos_hash)
            if existing is None or existing.num_simulations < entry.num_simulations:
                self._entries[pos_hash] = entry
                self._max_depth = max(self._max_depth, entry.turn + 1)

    def _hash_position(self, game_state: state_module.State) -> str:
        """Generate hash for a game position.

        The hash captures observable features that determine optimal play:
        - Turn number and active player
        - Queue state (cards in order)
        - Both players' hands (sorted for consistency)
        - Beasty Bar and That's It contents

        Note: Deck contents are not included as they are hidden information.

        Args:
            game_state: Game state to hash

        Returns:
            Hex string hash of the position
        """
        parts = []

        # Basic game info
        parts.append(f"t{game_state.turn}")
        parts.append(f"p{game_state.active_player}")

        # Queue (ordered)
        queue_str = ",".join(
            f"{c.species}:{c.owner}" for c in game_state.zones.queue
        )
        parts.append(f"q[{queue_str}]")

        # Hands (sorted within each player for consistency)
        for player_idx in range(2):
            hand = game_state.players[player_idx].hand
            # Sort by species name for deterministic hashing
            sorted_hand = sorted(hand, key=lambda c: c.species)
            hand_str = ",".join(c.species for c in sorted_hand)
            parts.append(f"h{player_idx}[{hand_str}]")

        # Beasty Bar contents (affects scoring, but order doesn't matter for strategy)
        bar_cards = sorted(
            f"{c.species}:{c.owner}" for c in game_state.zones.beasty_bar
        )
        parts.append(f"bar[{','.join(bar_cards)}]")

        # That's It contents
        ti_cards = sorted(
            f"{c.species}:{c.owner}" for c in game_state.zones.thats_it
        )
        parts.append(f"ti[{','.join(ti_cards)}]")

        # Deck sizes (not contents)
        for player_idx in range(2):
            deck_size = len(game_state.players[player_idx].deck)
            parts.append(f"d{player_idx}:{deck_size}")

        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def set_metadata(self, key: str, value: object) -> None:
        """Set metadata value.

        Args:
            key: Metadata key
            value: Metadata value (must be JSON serializable for json format)
        """
        self._metadata[key] = value

    def get_metadata(self, key: str, default: object = None) -> object:
        """Get metadata value.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self._metadata.get(key, default)


class OpeningBookGenerator:
    """Generator for creating opening books using MCTS analysis.

    Generates opening book entries by running high-simulation MCTS
    on positions from random game seeds up to a specified depth.

    Generation Strategy:
        1. Start from random initial game states (different shuffles)
        2. Run MCTS from each position to get policy distribution
        3. Follow most likely moves to explore common lines
        4. Store all positions up to max_depth

    Example:
        >>> from _02_agents.neural.network import BeastyBarNetwork
        >>> network = BeastyBarNetwork.load("checkpoint.pt")
        >>> generator = OpeningBookGenerator(network, num_simulations=1600)
        >>> book = OpeningBook()
        >>> generator.generate(book, num_seeds=10000, depth=3)
        >>> book.save("opening_book.json")
    """

    def __init__(
        self,
        network: BeastyBarNetwork,
        num_simulations: int = 1600,
        c_puct: float = 1.5,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the generator with a neural network.

        Args:
            network: Policy-value network for MCTS evaluation
            num_simulations: MCTS simulations per position (higher = better quality)
            c_puct: Exploration constant for MCTS
            device: Device to run network on
        """
        # Import here to avoid circular dependencies
        from _02_agents.mcts.search import MCTS

        self.network = network
        self.num_simulations = num_simulations

        self.mcts = MCTS(
            network=network,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=0.03,  # Lower noise for book generation
            dirichlet_epsilon=0.0,  # No noise - we want best analysis
            device=device,
        )

    def generate(
        self,
        book: OpeningBook,
        num_seeds: int = 10000,
        depth: int = 3,
        *,
        both_perspectives: bool = True,
        progress_callback: callable | None = None,
    ) -> None:
        """Generate opening book entries from random game seeds.

        Args:
            book: Opening book to populate
            num_seeds: Number of random game seeds to analyze
            depth: Maximum move depth to analyze (1-3 recommended)
            both_perspectives: If True, analyze from both players' perspectives
            progress_callback: Optional callback(seed_idx, num_seeds) for progress
        """
        book._num_simulations = self.num_simulations
        book.set_metadata("generation_depth", depth)
        book.set_metadata("num_seeds", num_seeds)

        for seed_idx in range(num_seeds):
            if progress_callback is not None:
                progress_callback(seed_idx, num_seeds)

            seed = seed_idx  # Use index as seed for reproducibility

            # Analyze from player 0's perspective
            self._generate_from_seed(book, seed, starting_player=0, depth=depth)

            # Optionally analyze from player 1's perspective
            if both_perspectives:
                self._generate_from_seed(book, seed, starting_player=1, depth=depth)

        logger.info(
            f"Generated {len(book)} positions from {num_seeds} seeds at depth {depth}"
        )

    def _generate_from_seed(
        self,
        book: OpeningBook,
        seed: int,
        starting_player: int,
        depth: int,
    ) -> None:
        """Generate entries from a single game seed.

        Performs depth-limited tree exploration from the initial position,
        following the most likely moves according to MCTS.

        Args:
            book: Book to add entries to
            seed: Random seed for initial state
            starting_player: Which player moves first
            depth: Maximum depth to explore
        """
        initial = state_module.initial_state(seed=seed, starting_player=starting_player)
        self._explore_position(book, initial, depth, remaining_depth=depth)

    def _explore_position(
        self,
        book: OpeningBook,
        game_state: state_module.State,
        max_depth: int,
        remaining_depth: int,
    ) -> None:
        """Recursively explore and analyze a position.

        Args:
            book: Book to add entries to
            game_state: Current game state
            max_depth: Original max depth for turn calculation
            remaining_depth: Remaining depth to explore
        """
        if remaining_depth <= 0:
            return

        if engine.is_terminal(game_state):
            return

        # Check if we already have this position with enough simulations
        existing = book.lookup(game_state)
        if existing is not None and existing.num_simulations >= self.num_simulations:
            # Still explore children for depth coverage
            self._explore_children(book, game_state, max_depth, remaining_depth, existing)
            return

        # Run MCTS analysis
        perspective = game_state.active_player
        visit_distribution = self.mcts.search(
            game_state,
            perspective,
            temperature=1.0,
            add_root_noise=False,  # No noise for book generation
        )

        if not visit_distribution:
            return

        # Extract root value from MCTS
        # We re-run a quick search to get the value, or estimate from distribution
        value = self._estimate_value(game_state, perspective)

        # Convert visit distribution to counts (estimate from proportions)
        visit_counts = {
            action: int(prob * self.num_simulations)
            for action, prob in visit_distribution.items()
        }

        entry = OpeningEntry(
            action_probs=visit_distribution,
            visit_counts=visit_counts,
            value=value,
            num_simulations=self.num_simulations,
            turn=game_state.turn,
        )

        book.add_entry(game_state, entry)

        # Explore children
        self._explore_children(book, game_state, max_depth, remaining_depth, entry)

    def _explore_children(
        self,
        book: OpeningBook,
        game_state: state_module.State,
        max_depth: int,
        remaining_depth: int,
        entry: OpeningEntry,
    ) -> None:
        """Explore child positions from a node.

        Follows the top actions to explore likely continuations.

        Args:
            book: Book to add entries to
            game_state: Current game state
            max_depth: Original max depth
            remaining_depth: Remaining depth to explore
            entry: Entry for current position
        """
        # Sort actions by probability and explore top ones
        sorted_actions = sorted(
            entry.action_probs.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Explore top 3 actions or all if fewer
        num_to_explore = min(3, len(sorted_actions))

        for action_idx, prob in sorted_actions[:num_to_explore]:
            # Only explore significant moves
            if prob < 0.05:
                continue

            try:
                action = action_space.index_to_action(action_idx)
                next_state = engine.step(game_state, action)
                self._explore_position(book, next_state, max_depth, remaining_depth - 1)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to explore action {action_idx}: {e}")

    def _estimate_value(self, game_state: state_module.State, perspective: int) -> float:
        """Estimate position value using the network.

        Args:
            game_state: Game state to evaluate
            perspective: Player perspective

        Returns:
            Value estimate in [-1, 1]
        """
        import torch

        from _01_simulator import observations

        obs = observations.state_to_tensor(game_state, perspective)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        action_mask = action_space.legal_action_mask_tensor(game_state, perspective)
        mask_tensor = torch.from_numpy(action_mask).unsqueeze(0)

        # Get device from network
        device = next(self.network.parameters()).device
        obs_tensor = obs_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        with torch.no_grad():
            _, value = self.network(obs_tensor, mask_tensor)

        return float(value.squeeze().item())

    def generate_parallel(
        self,
        book: OpeningBook,
        num_seeds: int = 10000,
        depth: int = 3,
        num_workers: int = 4,
    ) -> None:
        """Generate opening book using parallel workers.

        Splits seed range across workers and merges results.
        Note: Requires network to support multi-processing.

        Args:
            book: Opening book to populate
            num_seeds: Number of random seeds to analyze
            depth: Maximum depth to explore
            num_workers: Number of parallel workers
        """

        # Split seeds across workers
        seeds_per_worker = num_seeds // num_workers
        seed_ranges = [
            (i * seeds_per_worker, (i + 1) * seeds_per_worker)
            for i in range(num_workers)
        ]
        # Handle remainder
        if num_seeds % num_workers != 0:
            seed_ranges[-1] = (seed_ranges[-1][0], num_seeds)

        # Note: This is a simplified implementation
        # Full parallel generation would require:
        # 1. Serializable network state
        # 2. Proper worker initialization
        # 3. Result aggregation

        logger.warning(
            "Parallel generation not fully implemented. "
            "Using sequential generation."
        )
        self.generate(book, num_seeds, depth)


class OpeningBookAgent:
    """Wrapper to use opening book with any agent.

    Falls back to the base agent when position is not in the book.

    Example:
        >>> book = OpeningBook.load("opening_book.json")
        >>> base_agent = MCTSAgent(network)
        >>> agent = OpeningBookAgent(book, base_agent)
    """

    def __init__(
        self,
        book: OpeningBook,
        fallback_agent: callable,
        temperature: float = 0.5,
    ) -> None:
        """Initialize the opening book agent.

        Args:
            book: Opening book for lookups
            fallback_agent: Agent to use when position not in book
            temperature: Temperature for sampling from book
        """
        self.book = book
        self.fallback_agent = fallback_agent
        self.temperature = temperature
        self._book_hits = 0
        self._book_misses = 0

    @property
    def name(self) -> str:
        """Return agent name."""
        fallback_name = getattr(self.fallback_agent, "name", "Unknown")
        return f"OpeningBook+{fallback_name}"

    def select_action(
        self,
        game_state: state_module.State,
        legal_actions,
    ):
        """Select action, using book if available.

        Args:
            game_state: Current game state
            legal_actions: Available legal actions

        Returns:
            Selected action
        """
        entry = self.book.lookup(game_state)

        if entry is not None:
            self._book_hits += 1
            action_idx = entry.sample_action(self.temperature)
            return action_space.index_to_action(action_idx)

        self._book_misses += 1
        return self.fallback_agent.select_action(game_state, legal_actions)

    def __call__(self, game_state: state_module.State, legal_actions):
        """Make agent callable."""
        return self.select_action(game_state, legal_actions)

    def get_stats(self) -> dict[str, int]:
        """Get book usage statistics.

        Returns:
            Dictionary with hits and misses
        """
        return {
            "book_hits": self._book_hits,
            "book_misses": self._book_misses,
            "hit_rate": (
                self._book_hits / (self._book_hits + self._book_misses)
                if (self._book_hits + self._book_misses) > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._book_hits = 0
        self._book_misses = 0


def generate_opening_book_cli(
    network_path: str,
    output_path: str,
    num_seeds: int = 10000,
    depth: int = 3,
    num_simulations: int = 1600,
) -> None:
    """CLI entry point for generating an opening book.

    Args:
        network_path: Path to network checkpoint
        output_path: Path to save the opening book
        num_seeds: Number of game seeds to analyze
        depth: Maximum depth to explore
        num_simulations: MCTS simulations per position
    """
    import torch

    from _02_agents.neural.network import BeastyBarNetwork

    # Load network
    logger.info(f"Loading network from {network_path}")
    checkpoint = torch.load(network_path, map_location="cpu")
    network = BeastyBarNetwork(checkpoint.get("config"))
    network.load_state_dict(checkpoint["network_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.eval()

    # Create generator
    generator = OpeningBookGenerator(
        network=network,
        num_simulations=num_simulations,
    )

    # Generate book
    book = OpeningBook()

    def progress(idx: int, total: int) -> None:
        if idx % 100 == 0:
            logger.info(f"Progress: {idx}/{total} seeds ({100*idx/total:.1f}%)")

    generator.generate(
        book,
        num_seeds=num_seeds,
        depth=depth,
        progress_callback=progress,
    )

    # Save book
    output_path_obj = Path(output_path)
    format_type = "pickle" if output_path_obj.suffix in (".pkl", ".pickle") else "json"
    book.save(output_path, format=format_type)

    # Print stats
    stats = book.get_stats()
    logger.info(f"Book statistics: {stats}")


__all__ = [
    "BookStats",
    "OpeningBook",
    "OpeningBookAgent",
    "OpeningBookGenerator",
    "OpeningEntry",
    "generate_opening_book_cli",
]
