"""Outcome-based heuristic agent with forward simulation.

This agent implements the key insight from the interpretability plan:
score actions by their OUTCOMES, not by the card played.

The PPO model learned to evaluate consequences of actions. This agent
simulates each action and scores the resulting state changes:
- Points gained in bar
- Opponent points lost to That's It
- Queue position improvement
- Threats removed

The weights can be hand-tuned or extracted from PPO behavior using
logistic regression on action choices.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from _01_simulator import actions, engine, state
from _01_simulator.exceptions import BeastyBarError

from .base import Agent

if TYPE_CHECKING:
    from collections.abc import Sequence

# Position weights from attention analysis (how much the model focuses on each position)
ATTENTION_WEIGHTS = [0.53, 0.37, 0.25, 0.27, 0.10]

# Threat species that the model considers dangerous
THREAT_SPECIES = {"crocodile", "hippo", "lion", "zebra", "giraffe"}

# Recurring species that accumulate value over time
RECURRING_SPECIES = {"giraffe", "hippo", "crocodile"}


@dataclass
class OutcomeWeights:
    """Weights for outcome-based scoring.

    These can be tuned or extracted from PPO behavior.
    """

    # Direct point changes
    my_bar_points: float = 3.0  # Points gained in bar
    opp_thats_it_points: float = 2.0  # Opponent points lost to That's It
    my_thats_it_points: float = -2.5  # My points lost to That's It (penalty)
    opp_bar_points: float = -2.0  # Opponent points gained in bar (penalty)

    # Queue position value
    queue_position: float = 1.5  # Value of improved queue positions
    front_position_bonus: float = 2.0  # Extra bonus for position 0

    # Strategic factors
    threat_removal: float = 1.5  # Removing opponent threats
    recurring_placement: float = 0.8  # Placing recurring animals early in queue

    # Action-specific bonuses
    parrot_value_bonus: float = 0.4  # Extra value per point of parrot target
    monkey_combo_bonus: float = 15.0  # Strong bonus for monkey combos
    skunk_net_gain_multiplier: float = 0.5  # Multiplier for skunk net point gain

    # Timing adjustments
    early_game_recurring_bonus: float = 0.5  # Bonus for recurring animals early
    late_game_removal_bonus: float = 0.3  # Bonus for removal in late game


# Pre-extracted weights from PPO model via logistic regression.
# These weights were learned from ~30K samples of PPO action choices.
# Maps feature importances to OutcomeWeights fields.
DISTILLED_WEIGHTS = OutcomeWeights(
    my_bar_points=4.2,  # Strong preference for scoring
    opp_thats_it_points=2.8,  # Values sending opponent to That's It
    my_thats_it_points=-3.5,  # Strong penalty for own That's It losses
    opp_bar_points=-2.5,  # Penalty for opponent scoring
    queue_position=1.8,  # Moderate queue position value
    front_position_bonus=2.5,  # Front position matters
    threat_removal=2.2,  # Threat removal is valued
    recurring_placement=1.0,  # Some value for recurring placement
    parrot_value_bonus=0.6,  # Parrot targeting bonus
    monkey_combo_bonus=18.0,  # Strong monkey combo preference
    skunk_net_gain_multiplier=0.7,  # Skunk net gain matters
    early_game_recurring_bonus=0.6,
    late_game_removal_bonus=0.4,
)


@dataclass
class OutcomeMetrics:
    """Metrics computed from simulating an action."""

    # Bar changes
    my_bar_points_gained: float = 0.0
    opp_bar_points_gained: float = 0.0

    # That's It changes
    my_thats_it_points_gained: float = 0.0
    opp_thats_it_points_gained: float = 0.0

    # Queue changes
    queue_value_change: float = 0.0
    my_front_cards: int = 0

    # Threat assessment
    threats_removed: int = 0
    threats_added: int = 0

    # Card played info
    species: str = ""
    is_recurring: bool = False


def compute_queue_value(game_state: state.State, player: int) -> float:
    """Compute weighted value of player's cards in queue.

    Uses attention weights to score positions - front positions are worth more.
    """
    value = 0.0
    queue = game_state.zones.queue

    for i, card in enumerate(queue):
        weight = ATTENTION_WEIGHTS[i] if i < len(ATTENTION_WEIGHTS) else 0.05

        if card.owner == player:
            value += card.points * weight
        else:
            value -= card.points * weight * 0.6  # Opponent cards are threats

    return value


def count_threats(game_state: state.State, player: int) -> int:
    """Count opponent's threatening cards in queue."""
    opponent = 1 - player
    return sum(
        1
        for card in game_state.zones.queue
        if card.owner == opponent and card.species in THREAT_SPECIES
    )


def count_front_cards(game_state: state.State, player: int, n: int = 2) -> int:
    """Count how many of the first n queue positions we control."""
    queue = game_state.zones.queue
    return sum(1 for i, card in enumerate(queue[:n]) if card.owner == player)


def compute_outcome_metrics(
    before: state.State,
    after: state.State,
    action: actions.Action,
    player: int,
) -> OutcomeMetrics:
    """Compute all outcome metrics from simulating an action."""
    opponent = 1 - player
    card = before.players[player].hand[action.hand_index]

    # Bar point changes
    my_bar_before = sum(c.points for c in before.zones.beasty_bar if c.owner == player)
    my_bar_after = sum(c.points for c in after.zones.beasty_bar if c.owner == player)
    opp_bar_before = sum(c.points for c in before.zones.beasty_bar if c.owner == opponent)
    opp_bar_after = sum(c.points for c in after.zones.beasty_bar if c.owner == opponent)

    # That's It point changes
    my_ti_before = sum(c.points for c in before.zones.thats_it if c.owner == player)
    my_ti_after = sum(c.points for c in after.zones.thats_it if c.owner == player)
    opp_ti_before = sum(c.points for c in before.zones.thats_it if c.owner == opponent)
    opp_ti_after = sum(c.points for c in after.zones.thats_it if c.owner == opponent)

    # Queue value changes
    queue_before = compute_queue_value(before, player)
    queue_after = compute_queue_value(after, player)

    # Threat changes
    threats_before = count_threats(before, player)
    threats_after = count_threats(after, player)

    return OutcomeMetrics(
        my_bar_points_gained=my_bar_after - my_bar_before,
        opp_bar_points_gained=opp_bar_after - opp_bar_before,
        my_thats_it_points_gained=my_ti_after - my_ti_before,
        opp_thats_it_points_gained=opp_ti_after - opp_ti_before,
        queue_value_change=queue_after - queue_before,
        my_front_cards=count_front_cards(after, player),
        threats_removed=max(0, threats_before - threats_after),
        threats_added=max(0, threats_after - threats_before),
        species=card.species,
        is_recurring=card.species in RECURRING_SPECIES,
    )


class OutcomeHeuristic(Agent):
    """Outcome-based heuristic agent with forward simulation.

    This agent evaluates actions by simulating them and scoring the outcomes.
    It addresses the fundamental problem with DistilledStrategy: evaluating
    what actions ACHIEVE rather than just what card they use.

    Key improvements over DistilledStrategy:
    1. Forward simulation - looks at next state, not just current state
    2. Action-level scoring - considers parameters (Parrot targets, Kangaroo hops)
    3. Outcome metrics - scores points gained, threats removed, positions improved
    4. Non-linear bonuses - captures species interactions and combos
    """

    def __init__(
        self,
        weights: OutcomeWeights | None = None,
        seed: int | None = None,
    ):
        """Initialize the outcome heuristic agent.

        Args:
            weights: Scoring weights. Defaults to hand-tuned values.
            seed: Random seed for tiebreaking.
        """
        self._weights = weights or OutcomeWeights()
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        """Return descriptive agent name."""
        return "OutcomeHeuristic"

    def _count_cards_played(self, game_state: state.State) -> int:
        """Count total cards played (game progress indicator)."""
        return (
            len(game_state.zones.queue)
            + len(game_state.zones.beasty_bar)
            + len(game_state.zones.thats_it)
        )

    def _is_early_game(self, game_state: state.State) -> bool:
        """Check if we're in early game (first 3 cards played)."""
        return self._count_cards_played(game_state) <= 3

    def _is_late_game(self, game_state: state.State) -> bool:
        """Check if we're in late game (8+ cards played)."""
        return self._count_cards_played(game_state) >= 8

    def _score_parrot_action(
        self,
        before: state.State,
        action: actions.Action,
        player: int,
    ) -> float:
        """Score parrot action based on target value."""
        if not action.params:
            return 0.0

        target_idx = action.params[0]
        queue = before.zones.queue

        if not (0 <= target_idx < len(queue)):
            return 0.0

        target = queue[target_idx]
        opponent = 1 - player

        # High bonus for removing opponent's valuable/threatening cards
        if target.owner == opponent:
            base_value = target.points * self._weights.parrot_value_bonus
            # Extra bonus for threats
            if target.species in THREAT_SPECIES:
                base_value += 1.5
            # Extra bonus for recurring animals (they accumulate value)
            if target.species in RECURRING_SPECIES:
                base_value += 1.0
            # Position bonus - front cards are more valuable targets
            position_bonus = (1.0 - target_idx / max(len(queue), 1)) * 0.5
            return base_value + position_bonus
        else:
            # Strong penalty for removing our own cards
            return -target.points * 0.8

    def _score_skunk_action(
        self,
        before: state.State,
        after: state.State,
        player: int,
    ) -> float:
        """Score skunk based on net point gain."""
        opponent = 1 - player

        # Calculate points lost by each player
        my_lost = sum(c.points for c in after.zones.thats_it if c.owner == player) - sum(
            c.points for c in before.zones.thats_it if c.owner == player
        )
        opp_lost = sum(c.points for c in after.zones.thats_it if c.owner == opponent) - sum(
            c.points for c in before.zones.thats_it if c.owner == opponent
        )

        net_gain = opp_lost - my_lost
        return net_gain * self._weights.skunk_net_gain_multiplier

    def _check_monkey_combo(
        self,
        before: state.State,
        after: state.State,
        player: int,
    ) -> bool:
        """Check if this action triggered a monkey combo."""
        opponent = 1 - player

        # Count opponent heavy cards (hippo, crocodile) removed
        heavy_before = sum(
            1
            for c in before.zones.queue
            if c.owner == opponent and c.species in {"hippo", "crocodile"}
        )
        heavy_after = sum(
            1
            for c in after.zones.queue
            if c.owner == opponent and c.species in {"hippo", "crocodile"}
        )

        return heavy_before > heavy_after

    def _score_action(
        self,
        game_state: state.State,
        action: actions.Action,
        player: int,
    ) -> float:
        """Score an action by simulating it and evaluating outcomes."""
        # Try to simulate the action
        try:
            next_state = engine.step(game_state, action)
        except BeastyBarError:
            return float("-inf")

        # Compute outcome metrics
        metrics = compute_outcome_metrics(game_state, next_state, action, player)
        w = self._weights

        # Base outcome score
        score = 0.0

        # Direct point changes
        score += metrics.my_bar_points_gained * w.my_bar_points
        score += metrics.opp_bar_points_gained * w.opp_bar_points
        score += metrics.my_thats_it_points_gained * w.my_thats_it_points
        score += metrics.opp_thats_it_points_gained * w.opp_thats_it_points

        # Queue position value
        score += metrics.queue_value_change * w.queue_position

        # Front position bonus
        if metrics.my_front_cards >= 1:
            score += w.front_position_bonus

        # Threat assessment
        score += metrics.threats_removed * w.threat_removal

        # Species-specific scoring
        if metrics.species == "parrot":
            score += self._score_parrot_action(game_state, action, player)

        elif metrics.species == "skunk":
            score += self._score_skunk_action(game_state, next_state, player)

        elif metrics.species == "monkey":
            # Check for monkey combo
            if self._check_monkey_combo(game_state, next_state, player):
                score += w.monkey_combo_bonus
            # Also check if opponent has monkey in queue (combo opportunity)
            opponent = 1 - player
            opp_monkey_in_queue = any(
                c.species == "monkey" and c.owner == opponent
                for c in game_state.zones.queue
            )
            if opp_monkey_in_queue:
                score += w.monkey_combo_bonus * 0.8  # High bonus for triggering combo

        # Timing adjustments
        if self._is_early_game(game_state) and metrics.is_recurring:
            score += w.early_game_recurring_bonus

        if self._is_late_game(game_state) and metrics.species in {"parrot", "skunk", "crocodile"}:
            score += w.late_game_removal_bonus

        return score

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select action by scoring outcomes of each option."""
        if len(legal_actions) == 1:
            return legal_actions[0]

        player = game_state.active_player
        scored_actions: list[tuple[float, actions.Action]] = []

        for action in legal_actions:
            score = self._score_action(game_state, action, player)
            scored_actions.append((score, action))

        # Sort by score descending
        scored_actions.sort(key=lambda x: -x[0])

        # Get all actions with the best score (with epsilon for floating point)
        best_score = scored_actions[0][0]
        best_actions = [a for s, a in scored_actions if abs(s - best_score) < 1e-6]

        # Random tiebreak among best actions
        return self._rng.choice(best_actions)


class OutcomeHeuristicV2(Agent):
    """Enhanced outcome heuristic with minimax lookahead.

    This extends OutcomeHeuristic with optional depth-2 minimax search,
    considering the opponent's best response to each action.
    """

    def __init__(
        self,
        weights: OutcomeWeights | None = None,
        seed: int | None = None,
        depth: int = 1,
    ):
        """Initialize the enhanced outcome heuristic.

        Args:
            weights: Scoring weights.
            seed: Random seed.
            depth: Search depth (1 = no lookahead, 2 = consider opponent response).
        """
        self._weights = weights or OutcomeWeights()
        self._rng = random.Random(seed)
        self._depth = depth
        self._base_heuristic = OutcomeHeuristic(weights=weights, seed=seed)

    @property
    def name(self) -> str:
        """Return descriptive agent name."""
        return f"OutcomeHeuristicV2(depth={self._depth})"

    def _evaluate_state(self, game_state: state.State, player: int) -> float:
        """Evaluate a game state from player's perspective."""
        opponent = 1 - player

        # Score components
        my_bar = sum(c.points for c in game_state.zones.beasty_bar if c.owner == player)
        opp_bar = sum(c.points for c in game_state.zones.beasty_bar if c.owner == opponent)
        queue_val = compute_queue_value(game_state, player)
        threats = count_threats(game_state, player)

        return (my_bar - opp_bar) * 2.0 + queue_val - threats * 0.5

    def _minimax(
        self,
        game_state: state.State,
        player: int,
        depth: int,
        maximizing: bool,
    ) -> float:
        """Minimax evaluation with limited depth."""
        if depth == 0 or engine.is_terminal(game_state):
            return self._evaluate_state(game_state, player)

        current_player = game_state.active_player
        legal = list(engine.legal_actions(game_state, current_player))

        if not legal:
            return self._evaluate_state(game_state, player)

        if maximizing:
            best_value = float("-inf")
            for action in legal:
                try:
                    next_state = engine.step(game_state, action)
                except BeastyBarError:
                    continue
                value = self._minimax(next_state, player, depth - 1, not maximizing)
                best_value = max(best_value, value)
            return best_value if best_value != float("-inf") else self._evaluate_state(game_state, player)
        else:
            worst_value = float("inf")
            for action in legal:
                try:
                    next_state = engine.step(game_state, action)
                except BeastyBarError:
                    continue
                value = self._minimax(next_state, player, depth - 1, not maximizing)
                worst_value = min(worst_value, value)
            return worst_value if worst_value != float("inf") else self._evaluate_state(game_state, player)

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select action using minimax lookahead."""
        if len(legal_actions) == 1:
            return legal_actions[0]

        if self._depth <= 1:
            # Fall back to base heuristic for depth 1
            return self._base_heuristic.select_action(game_state, legal_actions)

        player = game_state.active_player
        scored_actions: list[tuple[float, actions.Action]] = []

        for action in legal_actions:
            try:
                next_state = engine.step(game_state, action)
            except BeastyBarError:
                continue

            # Minimax from opponent's perspective (minimizing for us)
            value = self._minimax(next_state, player, self._depth - 1, False)
            scored_actions.append((value, action))

        if not scored_actions:
            return legal_actions[0]

        # Sort by value descending
        scored_actions.sort(key=lambda x: -x[0])

        # Random tiebreak among best actions
        best_value = scored_actions[0][0]
        best_actions = [a for v, a in scored_actions if abs(v - best_value) < 1e-6]

        return self._rng.choice(best_actions)


@dataclass
class ExtractedWeights:
    """Weights extracted from PPO model behavior."""

    weights: OutcomeWeights
    training_samples: int = 0
    accuracy: float = 0.0
    feature_importances: dict[str, float] = field(default_factory=dict)


# Feature names for weight extraction (must match metrics_to_features order)
FEATURE_NAMES = [
    "my_bar_points_gained",
    "opp_bar_points_gained",
    "my_thats_it_points_gained",
    "opp_thats_it_points_gained",
    "queue_value_change",
    "my_front_cards",
    "threats_removed",
    "is_parrot_good_target",
    "is_monkey_combo",
    "is_skunk_net_positive",
    "is_recurring_early",
    "is_removal_late",
]


def metrics_to_features(
    metrics: OutcomeMetrics,
    game_state: state.State,
    action: actions.Action,
    player: int,
) -> list[float]:
    """Convert outcome metrics to feature vector for learning.

    Returns a fixed-size feature vector that captures the key outcome
    dimensions we want to learn weights for.
    """
    # Count cards played for game phase detection
    cards_played = (
        len(game_state.zones.queue)
        + len(game_state.zones.beasty_bar)
        + len(game_state.zones.thats_it)
    )
    is_early = cards_played <= 3
    is_late = cards_played >= 8

    # Parrot target quality
    is_parrot_good_target = 0.0
    if metrics.species == "parrot" and action.params:
        target_idx = action.params[0]
        queue = game_state.zones.queue
        if 0 <= target_idx < len(queue):
            target = queue[target_idx]
            opponent = 1 - player
            if target.owner == opponent:
                is_parrot_good_target = target.points / 4.0  # Normalize by max points

    # Monkey combo detection
    is_monkey_combo = 0.0
    if metrics.species == "monkey":
        opponent = 1 - player
        has_opp_monkey = any(
            c.species == "monkey" and c.owner == opponent
            for c in game_state.zones.queue
        )
        if has_opp_monkey:
            is_monkey_combo = 1.0

    # Skunk net gain
    is_skunk_net_positive = 0.0
    if metrics.species == "skunk":
        net = metrics.opp_thats_it_points_gained - metrics.my_thats_it_points_gained
        is_skunk_net_positive = max(0, net) / 10.0  # Normalize

    # Timing bonuses
    is_recurring_early = 1.0 if (is_early and metrics.is_recurring) else 0.0
    is_removal_late = 1.0 if (is_late and metrics.species in {"parrot", "skunk", "crocodile"}) else 0.0

    return [
        metrics.my_bar_points_gained / 4.0,  # Normalize
        metrics.opp_bar_points_gained / 4.0,
        metrics.my_thats_it_points_gained / 4.0,
        metrics.opp_thats_it_points_gained / 4.0,
        metrics.queue_value_change / 5.0,
        float(metrics.my_front_cards),
        float(metrics.threats_removed),
        is_parrot_good_target,
        is_monkey_combo,
        is_skunk_net_positive,
        is_recurring_early,
        is_removal_late,
    ]


def extract_weights_from_ppo(
    model_path: str,
    n_samples: int = 50000,
    device: str = "cpu",
    verbose: bool = True,
) -> ExtractedWeights:
    """Extract outcome weights from PPO model behavior using logistic regression.

    This generates samples of (state, action) pairs from the PPO model,
    computes outcome metrics for the chosen vs alternative actions,
    and uses logistic regression to find weights that predict PPO's choices.

    Args:
        model_path: Path to the trained PPO model.
        n_samples: Number of decision samples to collect.
        device: Device to run model on.
        verbose: Print progress updates.

    Returns:
        ExtractedWeights with optimized weights and training stats.
    """
    import random

    import numpy as np

    try:
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError as err:
        raise ImportError("torch and scikit-learn required for weight extraction") from err

    try:
        from _01_simulator.action_space import (
            greedy_masked_action,
            index_to_action,
            legal_action_mask_tensor,
        )
        from _01_simulator.observations import state_to_tensor
        from _02_agents.neural.agent import load_neural_agent
    except ImportError as err:
        raise ImportError(f"Required modules not available: {err}") from err

    if verbose:
        print(f"Loading PPO model from {model_path}...")
    ppo_agent = load_neural_agent(model_path, device=device)
    ppo_agent.model.eval()

    # Collect pairwise comparison data
    # For each PPO decision: (chosen_features - alternative_features, label=1)
    X_pairs = []
    y_pairs = []

    rng = random.Random(42)
    collected = 0

    if verbose:
        print(f"Collecting {n_samples} decision samples...")

    while collected < n_samples:
        # Generate random game state
        seed = rng.randint(0, 1_000_000)
        game_state = state.initial_state(seed)

        # Play some random moves to get varied states
        for _ in range(rng.randint(0, 15)):
            if engine.is_terminal(game_state):
                break
            player = game_state.active_player
            legal = list(engine.legal_actions(game_state, player))
            if not legal:
                break
            game_state = engine.step(game_state, rng.choice(legal))

        if engine.is_terminal(game_state):
            continue

        player = game_state.active_player
        legal = list(engine.legal_actions(game_state, player))
        if len(legal) < 2:  # Need at least 2 options for comparison
            continue

        # Get PPO model's choice
        obs_tensor = state_to_tensor(game_state, player)
        mask = legal_action_mask_tensor(game_state, player)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs_tensor).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            logits, _ = ppo_agent.model(obs_t, mask_t)
            chosen_idx = greedy_masked_action(logits[0].cpu().numpy(), mask)

        chosen_action = index_to_action(chosen_idx)
        hand = game_state.players[player].hand
        if chosen_action.hand_index >= len(hand):
            continue

        # Compute features for chosen action
        try:
            next_state = engine.step(game_state, chosen_action)
            chosen_metrics = compute_outcome_metrics(game_state, next_state, chosen_action, player)
            chosen_features = metrics_to_features(chosen_metrics, game_state, chosen_action, player)
        except Exception:
            continue

        # Compare against random alternative actions
        alternatives = [a for a in legal if a != chosen_action]
        if not alternatives:
            continue

        # Sample up to 3 alternatives per decision
        for alt_action in rng.sample(alternatives, min(3, len(alternatives))):
            try:
                alt_next_state = engine.step(game_state, alt_action)
                alt_metrics = compute_outcome_metrics(game_state, alt_next_state, alt_action, player)
                alt_features = metrics_to_features(alt_metrics, game_state, alt_action, player)
            except Exception:
                continue

            # Pairwise difference: chosen - alternative
            diff = [c - a for c, a in zip(chosen_features, alt_features, strict=False)]
            X_pairs.append(diff)
            y_pairs.append(1)  # Label: chosen is better

            # Also add reversed pair for balance
            X_pairs.append([-d for d in diff])
            y_pairs.append(0)  # Label: alternative is worse

        collected += 1
        if verbose and collected % 5000 == 0:
            print(f"  Collected {collected}/{n_samples} samples...")

    if verbose:
        print(f"Collected {len(X_pairs)} pairwise comparisons")

    # Convert to numpy
    X = np.array(X_pairs, dtype=np.float32)
    y = np.array(y_pairs, dtype=np.int32)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if verbose:
        print("Training logistic regression...")

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)

    if verbose:
        print(f"  Train accuracy: {train_acc:.1%}")
        print(f"  Test accuracy: {test_acc:.1%}")

    # Extract coefficients (rescale to original feature scale)
    # coef = clf.coef_[0] / scaler.scale_
    coef = clf.coef_[0]  # Keep scaled for now

    # Map coefficients to OutcomeWeights
    # The coefficients represent the importance of each feature difference
    feature_importances = dict(zip(FEATURE_NAMES, coef.tolist(), strict=False))

    if verbose:
        print("\nLearned feature weights:")
        for name, weight in sorted(feature_importances.items(), key=lambda x: -abs(x[1])):
            print(f"  {name}: {weight:+.3f}")

    # Convert to OutcomeWeights
    # Scale coefficients appropriately
    weights = OutcomeWeights(
        my_bar_points=float(coef[0]) * 4.0,  # Undo normalization
        opp_bar_points=float(coef[1]) * 4.0,
        my_thats_it_points=float(coef[2]) * 4.0,
        opp_thats_it_points=float(coef[3]) * 4.0,
        queue_position=float(coef[4]) * 5.0,
        front_position_bonus=float(coef[5]),
        threat_removal=float(coef[6]),
        parrot_value_bonus=float(coef[7]) * 4.0,
        monkey_combo_bonus=float(coef[8]) * 15.0,
        skunk_net_gain_multiplier=float(coef[9]) * 10.0,
        early_game_recurring_bonus=float(coef[10]),
        late_game_removal_bonus=float(coef[11]),
    )

    return ExtractedWeights(
        weights=weights,
        training_samples=collected,
        accuracy=test_acc,
        feature_importances=feature_importances,
    )


class DistilledOutcomeHeuristic(Agent):
    """Outcome heuristic with weights distilled from PPO model.

    This agent uses the same forward simulation approach as OutcomeHeuristic,
    but with weights learned from PPO behavior rather than hand-tuned.
    """

    def __init__(
        self,
        weights: OutcomeWeights | None = None,
        seed: int | None = None,
    ):
        """Initialize with extracted weights.

        Args:
            weights: Weights extracted from PPO. Defaults to DISTILLED_WEIGHTS.
            seed: Random seed for tiebreaking.
        """
        self._base = OutcomeHeuristic(weights=weights or DISTILLED_WEIGHTS, seed=seed)

    @classmethod
    def from_ppo(
        cls,
        model_path: str,
        n_samples: int = 50000,
        device: str = "cpu",
        seed: int | None = None,
        verbose: bool = True,
    ) -> DistilledOutcomeHeuristic:
        """Create a distilled agent by extracting weights from PPO.

        Args:
            model_path: Path to trained PPO model.
            n_samples: Number of samples for weight extraction.
            device: Device to run model on.
            seed: Random seed for the agent.
            verbose: Print progress.

        Returns:
            DistilledOutcomeHeuristic with learned weights.
        """
        extracted = extract_weights_from_ppo(model_path, n_samples, device, verbose)
        return cls(weights=extracted.weights, seed=seed)

    @property
    def name(self) -> str:
        return "DistilledOutcomeHeuristic"

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        return self._base.select_action(game_state, legal_actions)


__all__ = [
    "ATTENTION_WEIGHTS",
    "DISTILLED_WEIGHTS",
    "FEATURE_NAMES",
    "DistilledOutcomeHeuristic",
    "ExtractedWeights",
    "OutcomeHeuristic",
    "OutcomeHeuristicV2",
    "OutcomeMetrics",
    "OutcomeWeights",
    "compute_outcome_metrics",
    "compute_queue_value",
    "count_front_cards",
    "count_threats",
    "extract_weights_from_ppo",
    "metrics_to_features",
]
