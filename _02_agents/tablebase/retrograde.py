"""Retrograde analysis for tablebase generation.

Implements backward analysis from terminal positions to build
complete endgame tablebases. Works by:

1. Enumerate all positions with n cards
2. Classify terminal positions (WIN/LOSS/DRAW)
3. Propagate values backward until fixed point:
   - If ANY successor is LOSS for opponent -> WIN
   - If ALL successors are WIN for opponent -> LOSS
   - Otherwise -> DRAW (if all successors known) or UNKNOWN

This is more efficient than forward minimax for large tablebases
because each position is visited at most once.

IMPORTANT: Actions are stored in canonical form (species + params) instead
of hand indices to avoid the canonicalization mismatch bug where sorted
hands produce different action indices than the original hand order.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from _01_simulator import actions, engine, rules, state

from .endgame import GameTheoreticValue
from .enumerate import CanonicalKey, EnumerationConfig, PositionEnumerator, PositionIndexer
from .storage import MMapTablebase, SharedArrayTablebase

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# Canonical action encoding
# Instead of storing hand_index (which depends on hand order),
# we store (species_id, params) which is order-independent.
# species_id is the index in rules.SPECIES keys (0-11 for 12 species)
# This avoids the bug where canonicalized hands are sorted but
# action indices refer to the unsorted hand order.

SPECIES_LIST = [s for s in rules.SPECIES if s != "unknown"]
SPECIES_TO_ID = {s: i for i, s in enumerate(SPECIES_LIST)}
ID_TO_SPECIES = dict(enumerate(SPECIES_LIST))


@dataclass(frozen=True)
class CanonicalAction:
    """Canonical representation of an action.

    Stores species + params instead of hand_index to avoid
    canonicalization mismatch when hands are sorted.
    """

    species: str
    params: tuple[int, ...]

    def to_action(self, game_state: state.State) -> actions.Action | None:
        """Convert to actual Action for given game state.

        Finds the hand_index for the stored species in the actual hand.

        Args:
            game_state: Current game state

        Returns:
            Action with correct hand_index, or None if species not in hand
        """
        hand = game_state.players[game_state.active_player].hand
        for idx, card in enumerate(hand):
            if card.species == self.species:
                return actions.Action(hand_index=idx, params=self.params)
        return None

    @classmethod
    def from_action(
        cls,
        action: actions.Action,
        game_state: state.State,
    ) -> CanonicalAction:
        """Create canonical action from actual action.

        Args:
            action: Actual action with hand_index
            game_state: Game state to look up species

        Returns:
            CanonicalAction with species instead of hand_index
        """
        hand = game_state.players[game_state.active_player].hand
        species = hand[action.hand_index].species
        return cls(species=species, params=action.params)

    def encode(self) -> int:
        """Encode as integer for compact storage.

        Format: species_id (4 bits) + first param (4 bits)
        This fits common cases but truncates complex params.
        """
        species_id = SPECIES_TO_ID.get(self.species, 0)
        first_param = self.params[0] if self.params else 0
        return (species_id & 0x0F) | ((first_param & 0x0F) << 4)

    @classmethod
    def decode(cls, encoded: int) -> CanonicalAction:
        """Decode from integer.

        Note: This only recovers species and first param.
        For full params, need game state context.
        """
        species_id = encoded & 0x0F
        first_param = (encoded >> 4) & 0x0F
        species = ID_TO_SPECIES.get(species_id, "lion")
        params = (first_param,) if first_param > 0 else ()
        return cls(species=species, params=params)


@dataclass
class RetrogradeConfig:
    """Configuration for retrograde analysis."""

    max_cards: int = 10
    max_iterations: int = 1000
    log_interval: int = 1
    use_numpy: bool = True
    store_actions: bool = True


@dataclass
class RetrogradeStats:
    """Statistics from retrograde analysis."""

    total_positions: int = 0
    terminal_positions: int = 0
    solved_positions: int = 0
    iterations: int = 0
    elapsed_seconds: float = 0.0
    positions_per_second: float = 0.0


class RetrogradeTablebase:
    """Build tablebase backward from terminal positions.

    Uses retrograde analysis to systematically solve all positions
    with a given number of cards. More efficient than forward search
    for complete tablebase generation.

    Actions are stored in canonical form (species + params) to avoid
    the hand ordering bug where sorted canonicalized hands produce
    different indices than the original hand order.
    """

    def __init__(self, config: RetrogradeConfig | None = None):
        self.config = config or RetrogradeConfig()
        self.enumerator = PositionEnumerator(
            EnumerationConfig(max_total_cards=self.config.max_cards)
        )
        self.indexer = PositionIndexer(self.config.max_cards)

        # Storage - using CanonicalAction instead of raw indices
        self._positions: dict[CanonicalKey, state.State] = {}
        self._values: dict[CanonicalKey, int] = {}  # GameTheoreticValue ints
        self._depths: dict[CanonicalKey, int] = {}
        self._canonical_actions: dict[CanonicalKey, CanonicalAction | None] = {}
        self._successors: dict[CanonicalKey, list[tuple[CanonicalKey, CanonicalAction]]] = {}
        self._predecessors: dict[CanonicalKey, list[CanonicalKey]] = {}

        self._stats = RetrogradeStats()

    def generate(
        self,
        max_cards: int | None = None,
        storage: MMapTablebase | SharedArrayTablebase | None = None,
    ) -> RetrogradeStats:
        """Generate tablebase using retrograde analysis.

        Args:
            max_cards: Maximum cards (overrides config)
            storage: Optional external storage

        Returns:
            Generation statistics
        """
        if max_cards is not None:
            self.config.max_cards = max_cards

        start_time = time.time()
        logger.info("Starting retrograde analysis for %d cards", self.config.max_cards)

        # Phase 1: Enumerate all positions
        logger.info("Phase 1: Enumerating positions...")
        self._enumerate_all_positions()

        # Phase 2: Build successor/predecessor graph
        logger.info("Phase 2: Building position graph...")
        self._build_position_graph()

        # Phase 3: Classify terminal positions
        logger.info("Phase 3: Classifying terminal positions...")
        self._classify_terminal_positions()

        # Phase 4: Propagate backward
        logger.info("Phase 4: Propagating values backward...")
        self._propagate_backward()

        # Finalize stats
        self._stats.elapsed_seconds = time.time() - start_time
        self._stats.positions_per_second = (
            self._stats.total_positions / self._stats.elapsed_seconds
            if self._stats.elapsed_seconds > 0
            else 0
        )

        logger.info(
            "Retrograde complete: %d/%d positions solved in %.1fs (%.1f pos/s)",
            self._stats.solved_positions,
            self._stats.total_positions,
            self._stats.elapsed_seconds,
            self._stats.positions_per_second,
        )

        # Transfer to external storage if provided
        if storage is not None:
            self._transfer_to_storage(storage)

        return self._stats

    def _enumerate_all_positions(self) -> None:
        """Enumerate and index all positions."""
        self._positions.clear()
        self._values.clear()
        self._depths.clear()
        self._canonical_actions.clear()

        count = 0
        for key, game_state in self.enumerator.enumerate(self.config.max_cards):
            if key not in self._positions:
                self._positions[key] = game_state
                self._values[key] = GameTheoreticValue.UNKNOWN
                self._depths[key] = 0
                self._canonical_actions[key] = None
                count += 1

        self._stats.total_positions = count
        logger.info("Enumerated %d unique positions", count)

    def _build_position_graph(self) -> None:
        """Build successor and predecessor relationships.

        Stores CanonicalAction for each edge to maintain order-independence.
        """
        self._successors.clear()
        self._predecessors.clear()

        for key in self._positions:
            self._successors[key] = []
            self._predecessors[key] = []

        for key, game_state in self._positions.items():
            if engine.is_terminal(game_state):
                continue

            # Get legal actions and successors
            active = game_state.active_player
            legal = list(engine.legal_actions(game_state, active))

            for action in legal:
                try:
                    successor_state = engine.step(game_state, action)
                    successor_key = self.enumerator.canonicalize(successor_state)

                    if successor_key in self._positions:
                        # Store canonical action (species + params), not hand_index
                        canonical_action = CanonicalAction.from_action(action, game_state)
                        self._successors[key].append((successor_key, canonical_action))
                        self._predecessors[successor_key].append(key)
                except (ValueError, IndexError):
                    continue

        logger.info(
            "Built graph: avg %.1f successors per position",
            sum(len(s) for s in self._successors.values()) / max(1, len(self._successors)),
        )

    def _classify_terminal_positions(self) -> None:
        """Classify all terminal positions."""
        terminal_count = 0
        win_count = 0
        loss_count = 0
        draw_count = 0

        for key, game_state in self._positions.items():
            if not engine.is_terminal(game_state):
                continue

            terminal_count += 1
            scores = engine.score(game_state)
            active = game_state.active_player

            my_score = scores[active]
            opp_score = scores[1 - active]

            if my_score > opp_score:
                self._values[key] = GameTheoreticValue.WIN
                win_count += 1
            elif my_score < opp_score:
                self._values[key] = GameTheoreticValue.LOSS
                loss_count += 1
            else:
                self._values[key] = GameTheoreticValue.DRAW
                draw_count += 1

            self._depths[key] = 0

        self._stats.terminal_positions = terminal_count
        self._stats.solved_positions = terminal_count
        logger.info(
            "Classified %d terminal positions: %d WIN, %d LOSS, %d DRAW",
            terminal_count,
            win_count,
            loss_count,
            draw_count,
        )

    def _propagate_backward(self) -> None:
        """Propagate values backward until fixed point."""
        iteration = 0
        changed = True

        # Initialize worklist with predecessors of terminal positions
        worklist = set()
        for key, value in self._values.items():
            if value != GameTheoreticValue.UNKNOWN:
                for pred_key in self._predecessors.get(key, []):
                    if self._values[pred_key] == GameTheoreticValue.UNKNOWN:
                        worklist.add(pred_key)

        logger.info("Initial worklist: %d positions", len(worklist))

        while changed and iteration < self.config.max_iterations:
            iteration += 1
            changed = False
            new_solved = 0
            next_worklist = set()

            for key in worklist:
                if self._values[key] != GameTheoreticValue.UNKNOWN:
                    continue

                game_state = self._positions[key]
                value, depth, canonical_action = self._evaluate_position(key, game_state)

                if value != GameTheoreticValue.UNKNOWN:
                    self._values[key] = value
                    self._depths[key] = depth
                    self._canonical_actions[key] = canonical_action
                    self._stats.solved_positions += 1
                    new_solved += 1
                    changed = True

                    # Add predecessors to next worklist
                    for pred_key in self._predecessors.get(key, []):
                        if self._values[pred_key] == GameTheoreticValue.UNKNOWN:
                            next_worklist.add(pred_key)

            worklist = next_worklist

            if iteration % self.config.log_interval == 0:
                logger.info(
                    "Iteration %d: +%d solved, %d/%d total (%.1f%%), worklist: %d",
                    iteration,
                    new_solved,
                    self._stats.solved_positions,
                    self._stats.total_positions,
                    100 * self._stats.solved_positions / max(1, self._stats.total_positions),
                    len(worklist),
                )

        self._stats.iterations = iteration

        # Check for remaining unsolved
        unsolved = sum(1 for v in self._values.values() if v == GameTheoreticValue.UNKNOWN)
        if unsolved > 0:
            logger.warning(
                "%d positions remain UNKNOWN after %d iterations",
                unsolved,
                iteration,
            )
            # Try to classify remaining as DRAW
            self._classify_remaining_as_draw()

    def _evaluate_position(
        self,
        key: CanonicalKey,
        game_state: state.State,
    ) -> tuple[int, int, CanonicalAction | None]:
        """Evaluate a non-terminal position from successors.

        Args:
            key: Position key
            game_state: Game state

        Returns:
            Tuple of (value, depth, canonical_action)
        """
        successors = self._successors.get(key, [])

        if not successors:
            # No legal moves - treat as draw
            return GameTheoreticValue.DRAW, 0, None

        # Check successor values
        # For the active player:
        # - If ANY successor is LOSS for opponent (WIN for us) -> WIN
        # - If ALL successors are WIN for opponent (LOSS for us) -> LOSS
        # - If all known and mixed -> DRAW

        best_value = GameTheoreticValue.LOSS
        best_depth = 0
        best_action: CanonicalAction | None = None
        all_known = True
        found_win = False
        all_loss = True

        for succ_key, canonical_action in successors:
            succ_value = self._values[succ_key]

            if succ_value == GameTheoreticValue.UNKNOWN:
                all_known = False
                all_loss = False
                continue

            succ_depth = self._depths[succ_key]

            # Successor value is from opponent's perspective after our move
            # So opponent's LOSS = our WIN, opponent's WIN = our LOSS
            our_value = self._negate_value(succ_value)

            if our_value == GameTheoreticValue.WIN:
                found_win = True
                if not found_win or succ_depth + 1 < best_depth or best_depth == 0:
                    best_value = GameTheoreticValue.WIN
                    best_depth = succ_depth + 1
                    best_action = canonical_action
            elif our_value == GameTheoreticValue.DRAW:
                all_loss = False
                if best_value != GameTheoreticValue.WIN:
                    best_value = GameTheoreticValue.DRAW
                    best_depth = succ_depth + 1
                    best_action = canonical_action
            else:  # our_value == LOSS
                if best_value == GameTheoreticValue.LOSS:
                    # Prefer longer losses (more plies to lose)
                    if succ_depth + 1 > best_depth:
                        best_depth = succ_depth + 1
                        best_action = canonical_action

        # Determine final value
        if found_win:
            return GameTheoreticValue.WIN, best_depth, best_action
        elif all_known:
            if all_loss:
                return GameTheoreticValue.LOSS, best_depth, best_action
            else:
                return GameTheoreticValue.DRAW, best_depth, best_action
        else:
            # Not all successors known yet
            return GameTheoreticValue.UNKNOWN, 0, None

    def _negate_value(self, value: int) -> int:
        """Negate value for opponent's perspective.

        Args:
            value: GameTheoreticValue int

        Returns:
            Negated value
        """
        if value == GameTheoreticValue.WIN:
            return GameTheoreticValue.LOSS
        elif value == GameTheoreticValue.LOSS:
            return GameTheoreticValue.WIN
        else:
            return value  # DRAW and UNKNOWN stay same

    def _classify_remaining_as_draw(self) -> None:
        """Classify remaining UNKNOWN positions as DRAW.

        Positions that can't be proven WIN or LOSS after full
        propagation are theoretical draws.
        """
        draw_count = 0
        for key in self._values:
            if self._values[key] == GameTheoreticValue.UNKNOWN:
                self._values[key] = GameTheoreticValue.DRAW
                self._depths[key] = 0
                self._stats.solved_positions += 1
                draw_count += 1

        if draw_count > 0:
            logger.info("Classified %d remaining positions as DRAW", draw_count)

    def _transfer_to_storage(
        self,
        storage: MMapTablebase | SharedArrayTablebase,
    ) -> None:
        """Transfer results to external storage.

        Args:
            storage: Storage to transfer to
        """
        # Build index if needed
        if not self.indexer.is_built:
            self.indexer.build_index(self.enumerator)

        transferred = 0
        for key, value in self._values.items():
            idx = self.indexer.key_to_index(key)
            if idx is not None:
                depth = self._depths.get(key, 0)
                canonical_action = self._canonical_actions.get(key)
                action_encoded = canonical_action.encode() if canonical_action else 0
                storage.set(idx, value, depth, action_encoded)
                transferred += 1

        logger.info("Transferred %d positions to storage", transferred)

    def get_value(self, key: CanonicalKey) -> int:
        """Get value for a position.

        Args:
            key: Position key

        Returns:
            GameTheoreticValue int
        """
        return self._values.get(key, GameTheoreticValue.UNKNOWN)

    def get_entry(self, key: CanonicalKey) -> tuple[int, int, CanonicalAction | None] | None:
        """Get full entry for a position.

        Args:
            key: Position key

        Returns:
            Tuple of (value, depth, canonical_action) or None
        """
        if key not in self._values:
            return None
        return (
            self._values[key],
            self._depths.get(key, 0),
            self._canonical_actions.get(key),
        )

    def lookup(self, game_state: state.State) -> tuple[int, int, actions.Action | None] | None:
        """Look up a game state in the tablebase.

        Returns the actual Action (with correct hand_index) for this state.

        Args:
            game_state: Game state to look up

        Returns:
            Tuple of (value, depth, action) or None
        """
        key = self.enumerator.canonicalize(game_state)
        entry = self.get_entry(key)
        if entry is None:
            return None

        value, depth, canonical_action = entry
        if canonical_action is None:
            return value, depth, None

        # Convert canonical action to actual action for this game state
        actual_action = canonical_action.to_action(game_state)
        return value, depth, actual_action

    def get_optimal_action(
        self,
        game_state: state.State,
    ) -> actions.Action | None:
        """Get optimal action for a game state.

        Args:
            game_state: Game state

        Returns:
            Optimal action or None
        """
        result = self.lookup(game_state)
        if result is None:
            return None
        _, _, action = result
        return action

    def get_stats(self) -> RetrogradeStats:
        """Get generation statistics."""
        return self._stats

    def validate_against_forward(
        self,
        sample_size: int = 100,
        seed: int = 42,
    ) -> dict[str, int]:
        """Validate retrograde results against forward search.

        Args:
            sample_size: Number of positions to validate
            seed: Random seed for sampling

        Returns:
            Dict with match/mismatch counts
        """
        import random

        from .endgame import EndgameTablebase

        rng = random.Random(seed)
        forward_tb = EndgameTablebase()

        keys = list(self._positions.keys())
        sample_keys = rng.sample(keys, min(sample_size, len(keys)))

        matches = 0
        mismatches = 0
        errors = 0

        for key in sample_keys:
            game_state = self._positions[key]
            retro_value = self._values[key]

            try:
                forward_entry = forward_tb.solve(game_state, game_state.active_player)
                forward_value = forward_entry.value

                if retro_value == forward_value:
                    matches += 1
                else:
                    mismatches += 1
                    logger.debug(
                        "Mismatch: retro=%s, forward=%s",
                        GameTheoreticValue(retro_value).name,
                        forward_value.name,
                    )
            except Exception as e:
                errors += 1
                logger.debug("Error validating position: %s", e)

        return {
            "matches": matches,
            "mismatches": mismatches,
            "errors": errors,
            "total": len(sample_keys),
        }


def generate_retrograde_tablebase(
    max_cards: int,
    output_path: Path | str | None = None,
    validate: bool = False,
) -> tuple[RetrogradeTablebase, RetrogradeStats]:
    """Generate tablebase using retrograde analysis.

    Args:
        max_cards: Maximum cards in play
        output_path: Optional path to save tablebase
        validate: Whether to validate against forward search

    Returns:
        Tuple of (tablebase, stats)
    """
    config = RetrogradeConfig(max_cards=max_cards)
    tablebase = RetrogradeTablebase(config)

    stats = tablebase.generate()

    if validate:
        validation = tablebase.validate_against_forward()
        logger.info(
            "Validation: %d matches, %d mismatches, %d errors",
            validation["matches"],
            validation["mismatches"],
            validation["errors"],
        )

    if output_path is not None:
        from pathlib import Path

        path = Path(output_path)
        num_positions = tablebase._stats.total_positions

        with MMapTablebase(path, num_positions) as storage:
            tablebase._transfer_to_storage(storage)

        logger.info("Saved tablebase to %s", path)

    return tablebase, stats


__all__ = [
    "CanonicalAction",
    "RetrogradeConfig",
    "RetrogradeStats",
    "RetrogradeTablebase",
    "generate_retrograde_tablebase",
]
