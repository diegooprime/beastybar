"""Perfect information solver for Beasty Bar using alpha-beta search.

This solver provides exact game-theoretic values when opponent's hand is known/deducible.
It implements:
1. Alpha-beta pruning with negamax formulation
2. Game-specific evaluation function for non-terminal positions
3. Transposition table for efficiency
4. Move ordering for better pruning
5. Iterative deepening with time limits

Use cases:
- Late game when opponent hand can be deduced
- Training data generation (ground truth values)
- Analysis mode

References:
- Section 9 of ROADMAP_TO_SUPERHUMAN.md
- outcome_heuristic.py for evaluation heuristics
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from _01_simulator import actions, engine, state
from _01_simulator.exceptions import BeastyBarError
from _02_agents.base import Agent
from _02_agents.outcome_heuristic import (
    ATTENTION_WEIGHTS,
    RECURRING_SPECIES,
    compute_queue_value,
    count_front_cards,
    count_threats,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Constants
# =============================================================================

# Value bounds for minimax
INF = 100_000.0
WIN_VALUE = 10_000.0
DRAW_VALUE = 0.0

# Default depth limits
DEFAULT_MAX_DEPTH = 20
DEFAULT_TIME_LIMIT_MS = 5000

# Transposition table entry types
class TTFlag(IntEnum):
    """Transposition table entry type."""
    EXACT = 0      # Exact value
    LOWER = 1      # Lower bound (fail-high)
    UPPER = 2      # Upper bound (fail-low)


# =============================================================================
# Game State Hashing
# =============================================================================

def hash_state(game_state: state.State) -> int:
    """Compute a hash for the game state for transposition table.

    The hash captures all game-relevant information:
    - Active player
    - Cards in each zone (queue, bar, thats_it)
    - Each player's hand and deck (for perfect info solver)

    Returns:
        Integer hash value
    """
    # Build a tuple representation of the state
    components = [game_state.active_player]

    # Queue (ordered)
    for card in game_state.zones.queue:
        components.extend([card.owner, card.species])
    components.append(-1)  # Separator

    # Beasty bar (ordered by entry)
    for card in game_state.zones.beasty_bar:
        components.extend([card.owner, card.species])
    components.append(-1)

    # That's it (ordered by entry)
    for card in game_state.zones.thats_it:
        components.extend([card.owner, card.species])
    components.append(-1)

    # Player hands and decks (perfect info - we know everything)
    for player_state in game_state.players:
        # Hand (sorted for consistency since order in hand doesn't matter strategically)
        hand_species = sorted(c.species for c in player_state.hand)
        components.extend(hand_species)
        components.append(-2)

        # Deck (order matters for draw)
        for card in player_state.deck:
            components.append(card.species)
        components.append(-2)

    return hash(tuple(components))


# =============================================================================
# Transposition Table
# =============================================================================

@dataclass
class TTEntry:
    """Transposition table entry."""
    depth: int
    value: float
    flag: TTFlag
    best_action: actions.Action | None = None


class TranspositionTable:
    """Hash table for caching search results.

    Uses replacement scheme where deeper searches replace shallower ones.
    """

    def __init__(self, max_size: int = 1_000_000) -> None:
        """Initialize transposition table.

        Args:
            max_size: Maximum number of entries (uses LRU-style eviction)
        """
        self.max_size = max_size
        self._table: dict[int, TTEntry] = {}
        self.hits = 0
        self.misses = 0
        self.stores = 0

    def probe(self, key: int, depth: int, alpha: float, beta: float) -> tuple[bool, float, actions.Action | None]:
        """Probe the transposition table for a cached result.

        Args:
            key: State hash
            depth: Current search depth
            alpha: Current alpha bound
            beta: Current beta bound

        Returns:
            Tuple of (found, value, best_action)
            If found is True, value can be used to cutoff or narrow bounds
        """
        entry = self._table.get(key)

        if entry is None:
            self.misses += 1
            return False, 0.0, None

        # Only use entries from equal or deeper searches
        if entry.depth < depth:
            self.misses += 1
            return False, 0.0, entry.best_action  # Still return best action for move ordering

        self.hits += 1

        if (
            entry.flag == TTFlag.EXACT
            or (entry.flag == TTFlag.LOWER and entry.value >= beta)
            or (entry.flag == TTFlag.UPPER and entry.value <= alpha)
        ):
            return True, entry.value, entry.best_action

        # Entry exists but can't cause cutoff - return best action for ordering
        return False, 0.0, entry.best_action

    def store(self, key: int, depth: int, value: float, flag: TTFlag, best_action: actions.Action | None) -> None:
        """Store a result in the transposition table.

        Args:
            key: State hash
            depth: Search depth
            value: Evaluation value
            flag: Entry type (exact, lower, upper bound)
            best_action: Best action found
        """
        # Simple replacement: deeper search replaces shallower
        existing = self._table.get(key)
        if existing is None or existing.depth <= depth:
            # Evict if at capacity (simple random eviction)
            if len(self._table) >= self.max_size and key not in self._table:
                # Remove an arbitrary entry
                self._table.pop(next(iter(self._table)))

            self._table[key] = TTEntry(depth, value, flag, best_action)
            self.stores += 1

    def clear(self) -> None:
        """Clear the transposition table."""
        self._table.clear()
        self.hits = 0
        self.misses = 0
        self.stores = 0

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._table)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_terminal(game_state: state.State, perspective: int) -> float:
    """Evaluate a terminal game state.

    Args:
        game_state: Terminal state to evaluate
        perspective: Player perspective for evaluation

    Returns:
        Value in [-WIN_VALUE, WIN_VALUE] range
    """
    scores = engine.score(game_state)
    my_score = scores[perspective]
    opp_score = scores[1 - perspective]

    margin = my_score - opp_score

    if margin > 0:
        # Win - add margin bonus
        return WIN_VALUE + margin
    elif margin < 0:
        # Loss - subtract margin penalty
        return -WIN_VALUE + margin
    else:
        # Draw
        return DRAW_VALUE


def evaluate_position(game_state: state.State, perspective: int) -> float:
    """Evaluate a non-terminal position heuristically.

    This evaluation function combines multiple factors:
    1. Current score differential
    2. Queue position value (front positions worth more)
    3. Threat assessment
    4. Hand strength
    5. Card count advantage

    Args:
        game_state: Current game state
        perspective: Player perspective

    Returns:
        Heuristic value estimate
    """
    opponent = 1 - perspective

    # 1. Current score differential (bar points)
    my_bar = sum(c.points for c in game_state.zones.beasty_bar if c.owner == perspective)
    opp_bar = sum(c.points for c in game_state.zones.beasty_bar if c.owner == opponent)
    score_value = (my_bar - opp_bar) * 100.0

    # 2. That's It penalty (cards lost)
    my_lost = sum(c.points for c in game_state.zones.thats_it if c.owner == perspective)
    opp_lost = sum(c.points for c in game_state.zones.thats_it if c.owner == opponent)
    thats_it_value = (opp_lost - my_lost) * 50.0

    # 3. Queue position value
    queue_value = compute_queue_value(game_state, perspective) * 40.0

    # 4. Front position bonus (controlling the entrance to the bar)
    my_front = count_front_cards(game_state, perspective, n=2)
    front_bonus = my_front * 30.0

    # 5. Threat assessment
    threats = count_threats(game_state, perspective)
    threat_penalty = threats * 20.0

    # 6. Hand value (expected potential)
    my_hand_value = _evaluate_hand(game_state.players[perspective].hand)
    opp_hand_value = _evaluate_hand(game_state.players[opponent].hand)
    hand_value = (my_hand_value - opp_hand_value) * 15.0

    # 7. Deck size (tempo - having more cards to play)
    my_cards = len(game_state.players[perspective].hand) + len(game_state.players[perspective].deck)
    opp_cards = len(game_state.players[opponent].hand) + len(game_state.players[opponent].deck)
    tempo_value = (my_cards - opp_cards) * 5.0

    # 8. Recurring animal bonus (giraffe, hippo, crocodile in good queue positions)
    recurring_value = _evaluate_recurring(game_state, perspective) * 25.0

    total = (
        score_value +
        thats_it_value +
        queue_value +
        front_bonus -
        threat_penalty +
        hand_value +
        tempo_value +
        recurring_value
    )

    return total


def _evaluate_hand(hand: tuple[state.Card, ...]) -> float:
    """Evaluate the potential value of a hand.

    Higher point cards and versatile cards are worth more.
    """
    if not hand:
        return 0.0

    value = 0.0
    for card in hand:
        # Base value from points
        value += card.points

        # Bonus for high-strength cards (can push to front)
        value += card.strength * 0.1

        # Bonus for versatile cards
        if card.species in {"parrot", "chameleon"}:
            value += 1.0

        # Bonus for recurring (accumulating value)
        if card.species in RECURRING_SPECIES:
            value += 0.5

    return value


def _evaluate_recurring(game_state: state.State, perspective: int) -> float:
    """Evaluate value of recurring animals in queue.

    Recurring animals in front positions are very valuable.
    """
    value = 0.0
    queue = game_state.zones.queue

    for i, card in enumerate(queue):
        if card.owner != perspective:
            continue
        if card.species not in RECURRING_SPECIES:
            continue

        # Position-weighted value (front is better)
        pos_weight = ATTENTION_WEIGHTS[i] if i < len(ATTENTION_WEIGHTS) else 0.05
        value += card.points * pos_weight * 2.0

    return value


# =============================================================================
# Move Ordering
# =============================================================================

def order_moves(
    game_state: state.State,
    legal_actions: list[actions.Action],
    tt_best: actions.Action | None,
    perspective: int,
) -> list[actions.Action]:
    """Order moves to improve alpha-beta pruning efficiency.

    Move ordering heuristics:
    1. TT best move first (if available)
    2. Captures/high-impact moves
    3. High point cards
    4. Species-specific priorities

    Args:
        game_state: Current game state
        legal_actions: List of legal actions
        tt_best: Best action from transposition table (if any)
        perspective: Current player perspective

    Returns:
        Ordered list of actions (best first)
    """
    if len(legal_actions) <= 1:
        return legal_actions

    scored_actions: list[tuple[float, int, actions.Action]] = []

    for idx, action in enumerate(legal_actions):
        score = 0.0

        # TT best move gets highest priority
        if tt_best is not None and action == tt_best:
            score += 10000.0

        # Get card info
        hand = game_state.players[game_state.active_player].hand
        if action.hand_index >= len(hand):
            scored_actions.append((score, idx, action))
            continue

        card = hand[action.hand_index]

        # High point cards are generally good
        score += card.points * 10.0

        # Species-specific ordering
        if card.species == "lion":
            # Lion pushes to front - very strong
            score += 50.0
        elif card.species == "parrot":
            # Parrot effectiveness depends on target
            if action.params:
                target_idx = action.params[0]
                queue = game_state.zones.queue
                if 0 <= target_idx < len(queue):
                    target = queue[target_idx]
                    if target.owner != game_state.active_player:
                        # Removing opponent card
                        score += 40.0 + target.points * 5.0
                    else:
                        # Removing own card is usually bad
                        score -= 30.0
        elif card.species == "monkey":
            # Check for combo potential
            opponent = 1 - game_state.active_player
            opp_monkey = any(c.species == "monkey" and c.owner == opponent for c in game_state.zones.queue)
            if opp_monkey:
                score += 60.0  # Combo potential
        elif card.species == "skunk":
            # Skunk is situational but often good in endgame
            queue = game_state.zones.queue
            opponent = 1 - game_state.active_player
            opp_high_in_queue = sum(1 for c in queue if c.owner == opponent and c.strength >= 10)
            score += opp_high_in_queue * 15.0
        elif card.species == "snake":
            # Snake reorders - good when we have low strength cards ahead
            score += 25.0
        elif card.species in RECURRING_SPECIES:
            # Recurring animals - good early
            cards_played = len(game_state.zones.queue) + len(game_state.zones.beasty_bar) + len(game_state.zones.thats_it)
            if cards_played < 6:
                score += 20.0

        # Use index as tiebreaker for stability
        scored_actions.append((score, idx, action))

    # Sort by score descending
    scored_actions.sort(key=lambda x: -x[0])

    return [action for _, _, action in scored_actions]


# =============================================================================
# Alpha-Beta Search with Negamax
# =============================================================================

@dataclass
class SearchResult:
    """Result from a search."""
    value: float
    best_action: actions.Action | None
    depth_reached: int
    nodes_searched: int
    tt_hits: int
    time_ms: float


@dataclass
class SearchStats:
    """Statistics from search."""
    nodes: int = 0
    cutoffs: int = 0
    tt_hits: int = 0
    tt_stores: int = 0


def alpha_beta(
    game_state: state.State,
    depth: int,
    alpha: float,
    beta: float,
    perspective: int,
    tt: TranspositionTable,
    stats: SearchStats,
    deadline: float | None = None,
) -> tuple[float, actions.Action | None]:
    """Alpha-beta search with negamax formulation.

    Uses negamax: value for current player is negation of recursive value.

    Args:
        game_state: Current game state
        depth: Remaining search depth
        alpha: Alpha bound (best value for maximizing player)
        beta: Beta bound (best value for minimizing player)
        perspective: Player whose turn it is (for evaluation)
        tt: Transposition table
        stats: Search statistics
        deadline: Time deadline (None for no limit)

    Returns:
        Tuple of (value, best_action) from perspective of active player
    """
    stats.nodes += 1

    # Check time limit
    if deadline is not None and time.perf_counter() > deadline:
        return 0.0, None

    # Terminal check
    if engine.is_terminal(game_state):
        # Evaluate from active player's perspective
        active = game_state.active_player
        return evaluate_terminal(game_state, active), None

    # Depth limit reached - use heuristic
    if depth <= 0:
        active = game_state.active_player
        return evaluate_position(game_state, active), None

    # Transposition table probe
    state_hash = hash_state(game_state)
    found, tt_value, tt_best = tt.probe(state_hash, depth, alpha, beta)

    if found:
        stats.tt_hits += 1
        return tt_value, tt_best

    # Get and order legal moves
    active = game_state.active_player
    legal = list(engine.legal_actions(game_state, active))

    if not legal:
        # No legal moves - evaluate position
        return evaluate_position(game_state, active), None

    # Order moves for better pruning
    ordered_moves = order_moves(game_state, legal, tt_best, active)

    best_value = -INF
    best_action = ordered_moves[0]  # Default to first move

    for action in ordered_moves:
        # Execute action
        try:
            next_state = engine.step(game_state, action)
        except BeastyBarError:
            continue

        # Recurse with negation (negamax)
        child_value, _ = alpha_beta(
            next_state,
            depth - 1,
            -beta,
            -alpha,
            1 - perspective,  # Opponent's perspective
            tt,
            stats,
            deadline,
        )

        # Negate for current player
        value = -child_value

        if value > best_value:
            best_value = value
            best_action = action

        alpha = max(alpha, value)

        # Beta cutoff
        if alpha >= beta:
            stats.cutoffs += 1
            break

    # Store in transposition table
    if best_value <= alpha:
        flag = TTFlag.UPPER
    elif best_value >= beta:
        flag = TTFlag.LOWER
    else:
        flag = TTFlag.EXACT

    tt.store(state_hash, depth, best_value, flag, best_action)
    stats.tt_stores += 1

    return best_value, best_action


# =============================================================================
# Iterative Deepening
# =============================================================================

def iterative_deepening(
    game_state: state.State,
    max_depth: int = DEFAULT_MAX_DEPTH,
    time_limit_ms: int = DEFAULT_TIME_LIMIT_MS,
    tt: TranspositionTable | None = None,
) -> SearchResult:
    """Search using iterative deepening with time limit.

    Iterative deepening provides:
    1. Anytime behavior (can stop early with best result so far)
    2. Better move ordering (TT populated from shallower searches)
    3. More accurate time management

    Args:
        game_state: Current game state
        max_depth: Maximum search depth
        time_limit_ms: Time limit in milliseconds
        tt: Optional transposition table (created if None)

    Returns:
        SearchResult with best action and statistics
    """
    if tt is None:
        tt = TranspositionTable()

    start_time = time.perf_counter()
    deadline = start_time + time_limit_ms / 1000.0

    best_result = SearchResult(
        value=0.0,
        best_action=None,
        depth_reached=0,
        nodes_searched=0,
        tt_hits=0,
        time_ms=0.0,
    )

    perspective = game_state.active_player

    # Get initial legal actions
    legal = list(engine.legal_actions(game_state, perspective))
    if not legal:
        return best_result

    if len(legal) == 1:
        # Only one move - no need to search
        best_result.best_action = legal[0]
        best_result.depth_reached = 0
        return best_result

    total_nodes = 0

    for depth in range(1, max_depth + 1):
        # Check if we have time for this depth
        if time.perf_counter() > deadline:
            break

        stats = SearchStats()

        value, action = alpha_beta(
            game_state,
            depth,
            -INF,
            INF,
            perspective,
            tt,
            stats,
            deadline,
        )

        total_nodes += stats.nodes

        # Update best result if search completed
        if action is not None:
            best_result.value = value
            best_result.best_action = action
            best_result.depth_reached = depth
            best_result.nodes_searched = total_nodes
            best_result.tt_hits = stats.tt_hits

        # Check for proven win/loss (can stop early)
        if abs(value) > WIN_VALUE - 1000:
            break

    best_result.time_ms = (time.perf_counter() - start_time) * 1000.0
    return best_result


# =============================================================================
# Perfect Information Solver Agent
# =============================================================================

class PerfectInfoSolver(Agent):
    """Perfect information solver agent using alpha-beta search.

    This agent assumes complete knowledge of the opponent's hand and deck.
    Use cases:
    - Late game analysis when opponent's hand can be deduced
    - Training data generation (ground truth values)
    - Post-game analysis

    The solver uses:
    - Alpha-beta pruning with negamax
    - Transposition table for caching
    - Move ordering for efficiency
    - Iterative deepening for time management
    """

    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_DEPTH,
        time_limit_ms: int = DEFAULT_TIME_LIMIT_MS,
        use_tt: bool = True,
        tt_size: int = 1_000_000,
    ) -> None:
        """Initialize the perfect information solver.

        Args:
            max_depth: Maximum search depth
            time_limit_ms: Time limit per move in milliseconds
            use_tt: Whether to use transposition table
            tt_size: Maximum entries in transposition table
        """
        self.max_depth = max_depth
        self.time_limit_ms = time_limit_ms
        self.use_tt = use_tt
        self._tt = TranspositionTable(tt_size) if use_tt else None
        self._last_result: SearchResult | None = None

    @property
    def name(self) -> str:
        """Return agent name."""
        return f"PerfectInfoSolver(d={self.max_depth}, t={self.time_limit_ms}ms)"

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select action using alpha-beta search.

        Args:
            game_state: Current game state (must have full information)
            legal_actions: Available legal actions

        Returns:
            Best action according to search
        """
        if len(legal_actions) == 1:
            return legal_actions[0]

        result = iterative_deepening(
            game_state,
            max_depth=self.max_depth,
            time_limit_ms=self.time_limit_ms,
            tt=self._tt,
        )

        self._last_result = result

        if result.best_action is not None:
            return result.best_action

        # Fallback to first legal action if search failed
        return legal_actions[0]

    def get_last_result(self) -> SearchResult | None:
        """Get the result from the last search.

        Returns:
            SearchResult from most recent select_action call
        """
        return self._last_result

    def clear_tt(self) -> None:
        """Clear the transposition table."""
        if self._tt is not None:
            self._tt.clear()

    def get_tt_stats(self) -> dict[str, float]:
        """Get transposition table statistics.

        Returns:
            Dictionary with hit rate, size, etc.
        """
        if self._tt is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "size": self._tt.size,
            "hits": self._tt.hits,
            "misses": self._tt.misses,
            "hit_rate": self._tt.hit_rate,
            "stores": self._tt.stores,
        }


# =============================================================================
# Analysis Utilities
# =============================================================================

def analyze_position(
    game_state: state.State,
    max_depth: int = DEFAULT_MAX_DEPTH,
    time_limit_ms: int = DEFAULT_TIME_LIMIT_MS,
) -> dict[actions.Action, SearchResult]:
    """Analyze all legal moves in a position.

    Args:
        game_state: Position to analyze
        max_depth: Maximum search depth
        time_limit_ms: Time limit per move

    Returns:
        Dictionary mapping each legal action to its search result
    """
    perspective = game_state.active_player
    legal = list(engine.legal_actions(game_state, perspective))

    results = {}
    tt = TranspositionTable()

    for action in legal:
        try:
            next_state = engine.step(game_state, action)
        except BeastyBarError:
            continue

        # Search from opponent's perspective
        result = iterative_deepening(
            next_state,
            max_depth=max_depth,
            time_limit_ms=time_limit_ms,
            tt=tt,
        )

        # Negate value for current player's perspective
        result.value = -result.value
        results[action] = result

    return results


def is_position_solved(game_state: state.State, max_depth: int = 50) -> tuple[bool, float]:
    """Check if a position can be solved exactly.

    A position is "solved" if search reaches all terminal states within depth.

    Args:
        game_state: Position to check
        max_depth: Maximum depth to search

    Returns:
        Tuple of (is_solved, value)
    """
    if engine.is_terminal(game_state):
        perspective = game_state.active_player
        return True, evaluate_terminal(game_state, perspective)

    tt = TranspositionTable()
    result = iterative_deepening(
        game_state,
        max_depth=max_depth,
        time_limit_ms=60000,  # 1 minute
        tt=tt,
    )

    # Check if we reached a proven win/loss
    is_proven = abs(result.value) > WIN_VALUE - 1000

    return is_proven, result.value


def generate_training_data(
    game_state: state.State,
    max_depth: int = DEFAULT_MAX_DEPTH,
    time_limit_ms: int = DEFAULT_TIME_LIMIT_MS,
) -> tuple[float, dict[int, float]]:
    """Generate training data from a position.

    Returns:
    - Value estimate for the position
    - Policy targets (action values)

    Args:
        game_state: Position to evaluate
        max_depth: Maximum search depth
        time_limit_ms: Time limit for search

    Returns:
        Tuple of (value, policy_dict) where policy_dict maps hand_index to score
    """
    perspective = game_state.active_player
    legal = list(engine.legal_actions(game_state, perspective))

    if not legal:
        return 0.0, {}

    tt = TranspositionTable()

    # Get main position value
    main_result = iterative_deepening(
        game_state,
        max_depth=max_depth,
        time_limit_ms=time_limit_ms,
        tt=tt,
    )

    # Compute value for each action
    action_values = {}
    for action in legal:
        try:
            next_state = engine.step(game_state, action)
        except BeastyBarError:
            continue

        result = iterative_deepening(
            next_state,
            max_depth=max_depth - 1,
            time_limit_ms=time_limit_ms // len(legal),
            tt=tt,
        )

        # Negate for current player
        action_values[action] = -result.value

    # Normalize to policy (softmax-style)
    if action_values:
        max_val = max(action_values.values())
        exp_values = {a: pow(2.718, (v - max_val) / 100.0) for a, v in action_values.items()}
        total = sum(exp_values.values())
        policy = {a: v / total for a, v in exp_values.items()}
    else:
        policy = {}

    return main_result.value, policy


__all__ = [
    "PerfectInfoSolver",
    "SearchResult",
    "TTEntry",
    "TTFlag",
    "TranspositionTable",
    "alpha_beta",
    "analyze_position",
    "evaluate_position",
    "evaluate_terminal",
    "generate_training_data",
    "hash_state",
    "is_position_solved",
    "iterative_deepening",
    "order_moves",
]
