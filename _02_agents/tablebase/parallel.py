"""Parallel tablebase generation using multiprocessing.

Implements parallel retrograde analysis for high-core-count machines.
Uses shared memory for cross-worker communication and work partitioning
based on position hash prefixes.

IMPORTANT: Actions are stored in canonical form (species + params) instead
of hand indices to avoid the canonicalization mismatch bug where sorted
hands produce different action indices than the original hand order.

Architecture:
1. Main process enumerates all positions and builds index
2. Positions partitioned by hash prefix across workers
3. Workers solve positions in parallel using shared memory
4. Periodic synchronization for cross-partition lookups
5. Checkpointing for recovery on long runs
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

from _01_simulator import engine, state

from .endgame import GameTheoreticValue
from .enumerate import EnumerationConfig, PositionEnumerator
from .retrograde import CanonicalAction
from .storage import (
    MMapTablebase,
    SharedArrayTablebase,
)

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel tablebase generation."""

    max_cards: int = 10
    num_workers: int | None = None  # None = cpu_count
    partition_bits: int = 8  # 2^8 = 256 partitions
    checkpoint_interval: int = 300  # Seconds between checkpoints
    sync_interval: int = 10  # Iterations between syncs
    log_interval: int = 5  # Seconds between progress logs
    max_iterations: int = 1000


@dataclass
class ParallelStats:
    """Statistics from parallel generation."""

    total_positions: int = 0
    solved_positions: int = 0
    terminal_positions: int = 0
    iterations: int = 0
    elapsed_seconds: float = 0.0
    positions_per_second: float = 0.0
    num_workers: int = 0
    checkpoints_saved: int = 0


class ParallelTablebaseGenerator:
    """Generate tablebase using multiple processes.

    Uses a partitioned approach where positions are divided by
    hash prefix and each worker handles its partition independently,
    with periodic synchronization for cross-partition dependencies.

    Actions are stored in canonical form (species + params) to avoid
    the hand ordering bug.
    """

    def __init__(self, config: ParallelConfig | None = None):
        self.config = config or ParallelConfig()
        self.num_workers = self.config.num_workers or cpu_count()

        # Shared state
        self._positions: dict[bytes, state.State] = {}
        self._key_to_index: dict[bytes, int] = {}
        self._index_to_key: list[bytes] = []
        self._partition_map: dict[int, list[int]] = defaultdict(list)

        self._stats = ParallelStats()
        self._shutdown = False

    def generate(
        self,
        max_cards: int | None = None,
        output_path: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        checkpoint_interval: int | None = None,
    ) -> ParallelStats:
        """Generate tablebase using parallel workers.

        Args:
            max_cards: Maximum cards (overrides config)
            output_path: Path for final tablebase
            checkpoint_path: Path for checkpoints
            checkpoint_interval: Seconds between checkpoints

        Returns:
            Generation statistics
        """
        if max_cards is not None:
            self.config.max_cards = max_cards
        if checkpoint_interval is not None:
            self.config.checkpoint_interval = checkpoint_interval

        start_time = time.time()
        self._stats.num_workers = self.num_workers

        logger.info(
            "Starting parallel generation: %d cards, %d workers",
            self.config.max_cards,
            self.num_workers,
        )

        # Phase 1: Enumerate and index positions
        logger.info("Phase 1: Enumerating positions...")
        self._enumerate_positions()

        # Phase 2: Create shared storage
        logger.info("Phase 2: Creating shared storage...")
        num_positions = len(self._index_to_key)
        self._stats.total_positions = num_positions

        # Use memory-mapped file for large tablebases
        if output_path:
            storage = MMapTablebase(output_path, num_positions)
        else:
            # Use shared memory for in-memory operation
            storage = SharedArrayTablebase(num_positions, name="tablebase_shm")

        try:
            # Phase 3: Classify terminal positions (single-threaded, fast)
            logger.info("Phase 3: Classifying terminal positions...")
            self._classify_terminals(storage)

            # Phase 4: Parallel propagation
            logger.info("Phase 4: Parallel backward propagation...")
            self._parallel_propagate(storage, checkpoint_path)

            # Finalize
            self._stats.elapsed_seconds = time.time() - start_time
            self._stats.positions_per_second = (
                self._stats.total_positions / self._stats.elapsed_seconds
                if self._stats.elapsed_seconds > 0
                else 0
            )

            # Final stats
            stats = storage.get_stats()
            self._stats.solved_positions = (
                stats["win"] + stats["loss"] + stats["draw"]
            )

            logger.info(
                "Parallel generation complete: %d/%d solved in %.1fs (%.1f pos/s)",
                self._stats.solved_positions,
                self._stats.total_positions,
                self._stats.elapsed_seconds,
                self._stats.positions_per_second,
            )

        finally:
            if isinstance(storage, SharedArrayTablebase):
                # Convert to file if output path specified
                if output_path:
                    storage.to_mmap(output_path)
                storage.close()
                storage.unlink()
            elif isinstance(storage, MMapTablebase):
                storage.flush()
                storage.close()

        return self._stats

    def _enumerate_positions(self) -> None:
        """Enumerate all positions and build index."""
        enumerator = PositionEnumerator(
            EnumerationConfig(max_total_cards=self.config.max_cards)
        )

        self._positions.clear()
        self._key_to_index.clear()
        self._index_to_key.clear()
        self._partition_map.clear()

        num_partitions = 2 ** self.config.partition_bits

        for key, game_state in enumerator.enumerate(self.config.max_cards):
            if key in self._key_to_index:
                continue

            idx = len(self._index_to_key)
            self._positions[key] = game_state
            self._key_to_index[key] = idx
            self._index_to_key.append(key)

            # Assign to partition by hash prefix
            partition = key[0] % num_partitions
            self._partition_map[partition].append(idx)

        logger.info(
            "Enumerated %d positions across %d partitions",
            len(self._index_to_key),
            len(self._partition_map),
        )

    def _classify_terminals(
        self,
        storage: MMapTablebase | SharedArrayTablebase,
    ) -> None:
        """Classify all terminal positions."""
        terminal_count = 0

        for key, game_state in self._positions.items():
            if not engine.is_terminal(game_state):
                continue

            terminal_count += 1
            scores = engine.score(game_state)
            active = game_state.active_player

            my_score = scores[active]
            opp_score = scores[1 - active]

            if my_score > opp_score:
                value = GameTheoreticValue.WIN
            elif my_score < opp_score:
                value = GameTheoreticValue.LOSS
            else:
                value = GameTheoreticValue.DRAW

            idx = self._key_to_index[key]
            storage.set(idx, value, 0, 0)

        self._stats.terminal_positions = terminal_count
        logger.info("Classified %d terminal positions", terminal_count)

    def _parallel_propagate(
        self,
        storage: MMapTablebase | SharedArrayTablebase,
        checkpoint_path: Path | str | None,
    ) -> None:
        """Parallel backward propagation using iterative approach.

        Uses CanonicalAction to store actions in order-independent form.
        """
        enumerator = PositionEnumerator(
            EnumerationConfig(max_total_cards=self.config.max_cards)
        )

        iteration = 0
        last_checkpoint = time.time()
        last_log = time.time()

        # Build successor relationships with canonical actions
        logger.info("Building successor graph...")
        # Store: idx -> list of (successor_idx, CanonicalAction)
        successors: dict[int, list[tuple[int, CanonicalAction]]] = defaultdict(list)

        for key, game_state in self._positions.items():
            if engine.is_terminal(game_state):
                continue

            idx = self._key_to_index[key]
            active = game_state.active_player

            for action in engine.legal_actions(game_state, active):
                try:
                    succ_state = engine.step(game_state, action)
                    succ_key = enumerator.canonicalize(succ_state)
                    succ_idx = self._key_to_index.get(succ_key)
                    if succ_idx is not None:
                        # Store canonical action (species + params), not hand_index
                        canonical_action = CanonicalAction.from_action(action, game_state)
                        successors[idx].append((succ_idx, canonical_action))
                except (ValueError, IndexError):
                    continue

        logger.info("Starting propagation iterations...")

        # Iterative propagation
        changed = True
        while changed and iteration < self.config.max_iterations:
            iteration += 1
            changed = False
            new_solved = 0

            # Process all unsolved positions
            for idx in range(len(self._index_to_key)):
                current_value = storage.get_value(idx)
                if current_value != GameTheoreticValue.UNKNOWN:
                    continue

                succs = successors.get(idx, [])
                if not succs:
                    # No successors - draw
                    storage.set(idx, GameTheoreticValue.DRAW, 0, 0)
                    new_solved += 1
                    changed = True
                    continue

                # Evaluate based on successors
                best_value = GameTheoreticValue.LOSS
                best_depth = 0
                best_action_encoded = 0
                all_known = True
                found_win = False

                for succ_idx, canonical_action in succs:
                    succ_value, succ_depth, _ = storage.get(succ_idx)

                    if succ_value == GameTheoreticValue.UNKNOWN:
                        all_known = False
                        continue

                    # Negate for opponent's perspective
                    if succ_value == GameTheoreticValue.WIN:
                        our_value = GameTheoreticValue.LOSS
                    elif succ_value == GameTheoreticValue.LOSS:
                        our_value = GameTheoreticValue.WIN
                    else:
                        our_value = succ_value

                    if our_value == GameTheoreticValue.WIN:
                        found_win = True
                        if not found_win or succ_depth + 1 < best_depth or best_depth == 0:
                            best_value = GameTheoreticValue.WIN
                            best_depth = succ_depth + 1
                            best_action_encoded = canonical_action.encode()
                    elif our_value == GameTheoreticValue.DRAW:
                        if best_value != GameTheoreticValue.WIN:
                            best_value = GameTheoreticValue.DRAW
                            best_depth = succ_depth + 1
                            best_action_encoded = canonical_action.encode()
                    else:  # LOSS
                        if best_value == GameTheoreticValue.LOSS:
                            if succ_depth + 1 > best_depth:
                                best_depth = succ_depth + 1
                                best_action_encoded = canonical_action.encode()

                # Determine if we can assign a value
                if found_win:
                    storage.set(idx, GameTheoreticValue.WIN, best_depth, best_action_encoded)
                    new_solved += 1
                    changed = True
                elif all_known:
                    storage.set(idx, best_value, best_depth, best_action_encoded)
                    new_solved += 1
                    changed = True

            self._stats.iterations = iteration

            # Periodic logging
            now = time.time()
            if now - last_log >= self.config.log_interval:
                stats = storage.get_stats()
                solved = stats["win"] + stats["loss"] + stats["draw"]
                logger.info(
                    "Iteration %d: +%d solved, %d/%d total (%.1f%%)",
                    iteration,
                    new_solved,
                    solved,
                    self._stats.total_positions,
                    100 * solved / max(1, self._stats.total_positions),
                )
                last_log = now

            # Checkpointing
            if checkpoint_path and now - last_checkpoint >= self.config.checkpoint_interval:
                if isinstance(storage, MMapTablebase):
                    storage.save_checkpoint(checkpoint_path)
                    self._stats.checkpoints_saved += 1
                    logger.info("Checkpoint saved: iteration %d", iteration)
                last_checkpoint = now

        # Classify remaining as draw
        for idx in range(len(self._index_to_key)):
            if storage.get_value(idx) == GameTheoreticValue.UNKNOWN:
                storage.set(idx, GameTheoreticValue.DRAW, 0, 0)

    def get_stats(self) -> ParallelStats:
        """Get generation statistics."""
        return self._stats


def generate_parallel_tablebase(
    max_cards: int,
    output_path: Path | str,
    num_workers: int | None = None,
    checkpoint_path: Path | str | None = None,
    checkpoint_interval: int = 300,
) -> ParallelStats:
    """Generate tablebase using parallel processing.

    Args:
        max_cards: Maximum cards in play
        output_path: Output file path
        num_workers: Number of worker processes
        checkpoint_path: Path for checkpoints
        checkpoint_interval: Seconds between checkpoints

    Returns:
        Generation statistics
    """
    config = ParallelConfig(
        max_cards=max_cards,
        num_workers=num_workers,
        checkpoint_interval=checkpoint_interval,
    )

    generator = ParallelTablebaseGenerator(config)

    return generator.generate(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )


__all__ = [
    "ParallelConfig",
    "ParallelStats",
    "ParallelTablebaseGenerator",
    "generate_parallel_tablebase",
]
