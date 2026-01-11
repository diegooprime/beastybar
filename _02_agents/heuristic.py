"""Heuristic-based agent implementation."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from _01_simulator import actions, engine, state
from _01_simulator.exceptions import BeastyBarError

from .base import Agent


@dataclass
class HeuristicConfig:
    """Configuration for parameterized heuristic agents."""

    # Material evaluation weights
    bar_weight: float = 2.0
    queue_front_weight: float = 1.1
    queue_back_weight: float = 0.3
    thats_it_weight: float = -0.5
    hand_weight: float = 0.1

    # Behavioral parameters
    aggression: float = 0.5  # 0=defensive, 1=aggressive
    noise_epsilon: float = 0.0  # Random noise for bounded rationality
    species_weights: dict[str, float] | None = None  # Per-species multipliers
    seed: int | None = None  # Random seed for noise


@dataclass
class MaterialEvaluator:
    """Configurable position evaluator using material advantage."""

    bar_weight: float = 2.0
    queue_front_weight: float = 1.1
    queue_back_weight: float = 0.3
    thats_it_weight: float = -0.5
    hand_weight: float = 0.1
    species_weights: dict[str, float] | None = None

    @classmethod
    def from_config(cls, config: HeuristicConfig) -> MaterialEvaluator:
        """Create an evaluator from a HeuristicConfig."""
        return cls(
            bar_weight=config.bar_weight,
            queue_front_weight=config.queue_front_weight,
            queue_back_weight=config.queue_back_weight,
            thats_it_weight=config.thats_it_weight,
            hand_weight=config.hand_weight,
            species_weights=config.species_weights,
        )

    def _get_species_multiplier(self, species: str) -> float:
        """Get the species-specific weight multiplier."""
        if self.species_weights is None:
            return 1.0
        return self.species_weights.get(species, 1.0)

    def __call__(self, game_state: state.State, player: int) -> float:
        """Evaluate the position from player's perspective."""
        score = 0.0

        # Score cards in beasty bar
        for card in game_state.zones.beasty_bar:
            weight = self.bar_weight if card.owner == player else -self.bar_weight
            multiplier = self._get_species_multiplier(card.species)
            score += weight * card.points * multiplier

        # Score cards in queue (position-weighted)
        queue = game_state.zones.queue
        queue_len = len(queue)
        for i, card in enumerate(queue):
            # Front of queue is better (closer to entering bar)
            position_factor = 1.0 - (i / max(queue_len, 1))
            weight = self.queue_front_weight * position_factor + self.queue_back_weight * (1 - position_factor)
            if card.owner != player:
                weight = -weight
            multiplier = self._get_species_multiplier(card.species)
            score += weight * card.points * multiplier

        # Score cards in that's it (negative)
        for card in game_state.zones.thats_it:
            weight = self.thats_it_weight if card.owner == player else -self.thats_it_weight
            multiplier = self._get_species_multiplier(card.species)
            score += weight * card.points * multiplier

        # Small bonus for cards in hand (potential)
        for card in game_state.players[player].hand:
            multiplier = self._get_species_multiplier(card.species)
            score += self.hand_weight * card.points * multiplier

        return score


EvaluatorFn = Callable[[state.State, int], float]


class HeuristicAgent(Agent):
    """Strong heuristic agent with species-specific strategies."""

    def __init__(
        self,
        evaluator: EvaluatorFn | None = None,
        seed: int | None = None,
        lookahead: bool = True,
        config: HeuristicConfig | None = None,
    ):
        self._config = config
        # If config provided, create evaluator from it; otherwise use provided or default
        if config is not None:
            self._evaluator = evaluator or MaterialEvaluator.from_config(config)
            self._aggression = config.aggression
            self._noise_epsilon = config.noise_epsilon
            effective_seed = config.seed if seed is None else seed
        else:
            self._evaluator = evaluator or MaterialEvaluator()
            self._aggression = 0.5
            self._noise_epsilon = 0.0
            effective_seed = seed

        self._rng = random.Random(effective_seed)
        self._lookahead = lookahead

    @property
    def name(self) -> str:
        """Return a descriptive name based on configuration."""
        if self._config is None:
            return "HeuristicAgent"

        parts = ["HeuristicAgent"]
        cfg = self._config

        # Describe key deviations from defaults
        if cfg.aggression >= 0.7:
            parts.append("aggressive")
        elif cfg.aggression <= 0.3:
            parts.append("defensive")

        if cfg.bar_weight != 2.0:
            parts.append(f"bar={cfg.bar_weight:.1f}")

        if cfg.queue_front_weight != 1.1:
            parts.append(f"qfront={cfg.queue_front_weight:.1f}")

        if cfg.noise_epsilon > 0:
            parts.append(f"noise={cfg.noise_epsilon:.2f}")

        if cfg.species_weights:
            species_str = ",".join(f"{k}={v}" for k, v in cfg.species_weights.items())
            parts.append(f"species({species_str})")

        if len(parts) == 1:
            return "HeuristicAgent(default)"

        return f"{parts[0]}({', '.join(parts[1:])})"

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        player = game_state.active_player
        scored_actions: list[tuple[float, actions.Action]] = []

        for action in legal_actions:
            score = self._evaluate_action(game_state, action, player)

            # Add aggression bias: higher aggression prefers playing cards
            # (lower hand_index = playing a card vs. passing)
            if self._aggression != 0.5:
                aggression_bias = (self._aggression - 0.5) * 2.0  # Range: -1 to 1
                # Slight bias toward playing (lower hand indices are typically plays)
                score += aggression_bias * 0.5

            # Add noise for bounded rationality
            if self._noise_epsilon > 0:
                noise = self._rng.gauss(0, self._noise_epsilon)
                score += noise

            scored_actions.append((score, action))

        # Sort by score descending
        scored_actions.sort(key=lambda x: -x[0])

        # Get all actions with the best score (with small epsilon for floating point)
        best_score = scored_actions[0][0]
        best_actions = [a for s, a in scored_actions if abs(s - best_score) < 1e-6]

        # Random tiebreak among best actions
        return self._rng.choice(best_actions)

    def _evaluate_action(
        self,
        game_state: state.State,
        action: actions.Action,
        player: int,
    ) -> float:
        """Evaluate an action using heuristics and lookahead."""
        card = game_state.players[player].hand[action.hand_index]

        # Apply the action and evaluate the resulting state
        try:
            next_state = engine.step(game_state, action)
        except BeastyBarError:
            return float("-inf")

        base_score = self._evaluator(next_state, player)

        # Apply species-specific heuristic adjustments
        adjustment = self._species_heuristic(game_state, next_state, action, card, player)

        return base_score + adjustment

    def _species_heuristic(
        self,
        before: state.State,
        after: state.State,
        action: actions.Action,
        card: state.Card,
        player: int,
    ) -> float:
        """Apply species-specific heuristic bonuses/penalties."""
        species = card.species
        opponent = 1 - player
        adjustment = 0.0

        # Count what changed
        my_bar_before = sum(1 for c in before.zones.beasty_bar if c.owner == player)
        my_bar_after = sum(1 for c in after.zones.beasty_bar if c.owner == player)
        opp_bar_before = sum(1 for c in before.zones.beasty_bar if c.owner == opponent)
        opp_bar_after = sum(1 for c in after.zones.beasty_bar if c.owner == opponent)

        my_thats_it_before = sum(1 for c in before.zones.thats_it if c.owner == player)
        my_thats_it_after = sum(1 for c in after.zones.thats_it if c.owner == player)
        opp_thats_it_before = sum(1 for c in before.zones.thats_it if c.owner == opponent)
        opp_thats_it_after = sum(1 for c in after.zones.thats_it if c.owner == opponent)

        # Track changes for species-specific heuristics
        _ = my_bar_after - my_bar_before  # my_bar_gained - reserved for future use
        _ = opp_bar_after - opp_bar_before  # opp_bar_gained - reserved for future use
        _ = my_thats_it_after - my_thats_it_before  # my_cards_bounced - reserved for future use
        opp_cards_bounced = opp_thats_it_after - opp_thats_it_before

        # Calculate point changes
        my_points_gained = sum(c.points for c in after.zones.beasty_bar if c.owner == player) - sum(
            c.points for c in before.zones.beasty_bar if c.owner == player
        )
        opp_points_gained = sum(c.points for c in after.zones.beasty_bar if c.owner == opponent) - sum(
            c.points for c in before.zones.beasty_bar if c.owner == opponent
        )
        my_points_lost = sum(c.points for c in after.zones.thats_it if c.owner == player) - sum(
            c.points for c in before.zones.thats_it if c.owner == player
        )
        opp_points_lost = sum(c.points for c in after.zones.thats_it if c.owner == opponent) - sum(
            c.points for c in before.zones.thats_it if c.owner == opponent
        )

        # Species-specific logic
        if species == "skunk":
            # Skunk is good when it removes high-value opponent cards
            if opp_points_lost >= 5:
                adjustment += 3.0
            elif opp_points_lost >= 3:
                adjustment += 1.5
            elif opp_points_lost == 0 and my_points_lost > 0:
                adjustment -= 2.0  # Don't play skunk if it only hurts us

        elif species == "parrot":
            # Parrot is good when targeting high-value opponent cards
            target_idx = action.params[0] if action.params else -1
            if 0 <= target_idx < len(before.zones.queue):
                target = before.zones.queue[target_idx]
                if target.owner == opponent:
                    adjustment += target.points * 0.5
                else:
                    adjustment -= target.points * 0.5  # Penalty for removing own card

        elif species == "lion":
            # Lion is great when it can reach the front and monkeys aren't a threat
            queue_after = after.zones.queue
            lion_in_queue = any(c.species == "lion" and c.owner == player for c in queue_after)
            lion_at_front = (
                lion_in_queue and queue_after and queue_after[0].species == "lion" and queue_after[0].owner == player
            )
            if lion_at_front:
                adjustment += 2.0
            # Check for monkey threat in opponent's known cards
            # (we can't see opponent's hand, but we can be cautious)

        elif species == "monkey":
            # Monkey is good when there are 2+ monkeys and it removes opponent heavyweights
            if opp_cards_bounced > 0 and opp_points_lost >= 4:
                adjustment += 2.5

        elif species == "seal":
            # Seal is good when it improves our queue position significantly
            if my_points_gained > opp_points_gained:
                adjustment += 1.5
            elif opp_points_gained > my_points_gained:
                adjustment -= 1.0

        elif species == "snake":
            # Snake (sorting) is good when it improves our position
            # Check if our high-strength cards moved forward
            my_queue_before = [c for c in before.zones.queue if c.owner == player]
            my_queue_after = [c for c in after.zones.queue if c.owner == player]
            if len(my_queue_after) > len(my_queue_before):
                # We have more cards in queue, check positions
                for i, c in enumerate(after.zones.queue):
                    if c.owner == player and i < 2:  # Near front
                        adjustment += 0.5 * c.points

        elif species == "zebra":
            # Zebra is a defensive play - good when we need to block
            opp_hippos_crocs = sum(
                1 for c in after.zones.queue if c.owner == opponent and c.species in ("hippo", "crocodile")
            )
            if opp_hippos_crocs > 0:
                adjustment += 1.5
            else:
                adjustment -= 0.5  # Less valuable without threats

        elif species == "kangaroo":
            # Kangaroo is good when it can hop to a safe position near front
            # Check where kangaroo ended up
            for i, c in enumerate(after.zones.queue):
                if c.species == "kangaroo" and c.owner == player:
                    if i < 2:  # Near front is good
                        adjustment += 1.0
                    break

        elif species == "giraffe":
            # Giraffe recurring hop is good - check if it reached front
            for i, c in enumerate(after.zones.queue):
                if c.species == "giraffe" and c.owner == player:
                    if i == 0:
                        adjustment += 1.5
                    elif i == 1:
                        adjustment += 0.5
                    break

        elif species == "crocodile":
            # Crocodile eating is valuable
            if opp_cards_bounced > 0:
                adjustment += opp_points_lost * 0.3

        elif species == "hippo":
            # Hippo pushing is good for positioning
            pass  # Base evaluation handles this

        elif species == "chameleon" and action.params:
            # Chameleon value depends on what it copies
            target_idx = action.params[0]
            if 0 <= target_idx < len(before.zones.queue):
                target = before.zones.queue[target_idx]
                # Higher strength copies are more valuable
                adjustment += target.strength * 0.1

        return adjustment


class OnlineStrategies(Agent):
    """Reactive agent that holds counter cards and punishes opponent mistakes.

    This agent tracks played cards to infer what the opponent likely still holds,
    then applies reactive rules to exploit timing windows:

    1. Crocodile timing: Hold Crocodile until opponent plays Chameleon.
       Chameleon copies abilities but reverts to strength 5, easy prey for Crocodile (10).

    2. Skunk timing: Hold Skunk until the top 2 STRENGTH species in queue
       belong primarily to the opponent. Skunk removes all animals of those species.
       Considers net point gain (opponent lost - our lost) before playing.

    3. Lion timing: Hold Lion until opponent's Lion is gone from the queue.
       When 2+ lions are in queue, the newly played lion is bounced to THAT'S IT.
       Playing when opponent's lion is in queue means OUR lion gets bounced.

    4. Parrot targeting: Prioritize opponent's cards near the front of the queue.
       Front cards are more valuable targets since they're closer to entering the bar.

    Falls back to MaterialEvaluator scoring when no punishment opportunity exists.
    """

    def __init__(self, seed: int | None = None):
        """Initialize the agent.

        Args:
            seed: Random seed for tiebreaking.
        """
        self._rng = random.Random(seed)
        self._evaluator = MaterialEvaluator()

    @property
    def name(self) -> str:
        """Return descriptive agent name."""
        return "OnlineStrategies"

    def _get_opponent_played_species(self, game_state: state.State, opponent: int) -> set[str]:
        """Get species the opponent has already played (visible in all zones).

        Args:
            game_state: Current game state.
            opponent: Opponent player index.

        Returns:
            Set of species names opponent has played.
        """
        played: set[str] = set()
        for card in game_state.zones.beasty_bar:
            if card.owner == opponent:
                played.add(card.species)
        for card in game_state.zones.thats_it:
            if card.owner == opponent:
                played.add(card.species)
        for card in game_state.zones.queue:
            if card.owner == opponent:
                played.add(card.species)
        return played

    def _get_opponent_remaining_species(self, game_state: state.State, opponent: int) -> set[str]:
        """Infer what species the opponent likely still holds.

        Each player starts with exactly one of each species (12 cards).
        Cards visible in any zone have been played.

        Args:
            game_state: Current game state.
            opponent: Opponent player index.

        Returns:
            Set of species names opponent likely still has.
        """
        from _01_simulator.rules import BASE_DECK

        played = self._get_opponent_played_species(game_state, opponent)
        return set(BASE_DECK) - played

    def _calculate_reactive_bonus(
        self,
        game_state: state.State,
        action: actions.Action,
        card: state.Card,
        player: int,
    ) -> float:
        """Calculate reactive bonus/penalty for playing a card.

        Applies strategic timing rules based on board state and opponent tracking.

        Args:
            game_state: Current game state.
            action: The action being evaluated.
            card: The card being played.
            player: The player index.

        Returns:
            Bonus (positive) or penalty (negative) adjustment.
        """
        bonus = 0.0
        opponent = 1 - player
        queue = game_state.zones.queue
        queue_len = len(queue)
        species = card.species

        opponent_played = self._get_opponent_played_species(game_state, opponent)

        # Rule 1: Crocodile timing - hold until Chameleon is vulnerable
        # Chameleon copies abilities but reverts to strength 5, easy prey for Crocodile (10)
        if species == "crocodile":
            chameleon_in_queue = any(c.species == "chameleon" for c in queue)
            if chameleon_in_queue:
                bonus += 5.0  # Big opportunity to eat the Chameleon
            else:
                # Check if opponent still has Chameleon - might want to wait
                if "chameleon" not in opponent_played:
                    bonus -= 1.0  # Slight penalty for playing early

        # Rule 2: Skunk timing - hold until skunk removes valuable opponent cards
        # Skunk expels all animals of the top 2 STRENGTH species (not points!)
        if species == "skunk":
            # Compute which species would be removed (by strength)
            species_strengths: dict[str, int] = {}
            for c in queue:
                if c.species != "skunk":
                    species_strengths[c.species] = c.strength

            if len(species_strengths) >= 2:
                # Get top 2 strength species that would be removed
                sorted_species = sorted(species_strengths.items(), key=lambda x: x[1], reverse=True)
                top_two = {sp for sp, _ in sorted_species[:2]}

                # Calculate net point gain (opponent removed - our removed)
                opp_removed_points = sum(c.points for c in queue if c.owner == opponent and c.species in top_two)
                my_removed_points = sum(c.points for c in queue if c.owner == player and c.species in top_two)
                net_gain = opp_removed_points - my_removed_points

                if net_gain >= 4:
                    bonus += 4.0  # Strong skunk opportunity
                elif net_gain >= 2:
                    bonus += 2.0  # Decent skunk opportunity
                elif net_gain <= -2:
                    bonus -= 3.0  # Skunk would hurt us more than opponent
            else:
                bonus -= 1.5  # Not enough distinct species to target

        # Rule 3: Lion timing - avoid being bounced
        # When 2+ lions in queue, the newly played lion is sent to THAT'S IT
        # If opponent's lion is in queue, playing our lion means OUR lion bounces
        if species == "lion":
            opp_lion_in_queue = any(c.species == "lion" and c.owner == opponent for c in queue)
            if opp_lion_in_queue:
                bonus -= 5.0  # DANGER! Our lion will be bounced
            elif "lion" in opponent_played:
                bonus += 3.0  # Safe - opponent's lion already played and gone from queue
            else:
                # Opponent may still have Lion in hand - risky
                bonus -= 2.0  # Hold back to avoid being bounced later

        # Rule 4: Parrot targeting - prioritize opponent cards near the front
        # Cards at the front of the queue are more valuable targets because they're
        # closer to entering the bar. Remove them now to deny future bar entries.
        # Note: Playing parrot does NOT trigger the five-card check because parrot
        # removes a card, keeping queue length the same or reducing it.
        if species == "parrot" and action.params:
            target_idx = action.params[0]
            if 0 <= target_idx < queue_len:
                target = queue[target_idx]
                if target.owner == opponent:
                    # Bonus based on position: front cards (low index) are more valuable
                    # Position 0 is best, position queue_len-1 is worst
                    position_value = 1.0 - (target_idx / max(queue_len, 1))
                    bonus += target.points * 0.3 + position_value * 2.0
                else:
                    # Penalty for removing our own cards
                    bonus -= target.points * 0.5

        return bonus

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        """Select action using reactive strategy with MaterialEvaluator fallback.

        Args:
            game_state: Current game state.
            legal_actions: Available legal actions.

        Returns:
            The selected action.
        """
        if len(legal_actions) == 1:
            return legal_actions[0]

        player = game_state.active_player
        scored_actions: list[tuple[float, actions.Action]] = []

        for action in legal_actions:
            card = game_state.players[player].hand[action.hand_index]

            # Simulate the action to get base evaluation
            try:
                next_state = engine.step(game_state, action)
            except BeastyBarError:
                scored_actions.append((float("-inf"), action))
                continue

            # Base score from MaterialEvaluator
            base_score = self._evaluator(next_state, player)

            # Add reactive bonus/penalty
            reactive_bonus = self._calculate_reactive_bonus(game_state, action, card, player)

            total_score = base_score + reactive_bonus
            scored_actions.append((total_score, action))

        # Sort by score descending
        scored_actions.sort(key=lambda x: -x[0])

        # Get all actions with the best score (with epsilon for floating point)
        best_score = scored_actions[0][0]
        best_actions = [a for s, a in scored_actions if abs(s - best_score) < 1e-6]

        # Random tiebreak among best actions
        return self._rng.choice(best_actions)


def create_heuristic_variants() -> list[Agent]:
    """Create a list of pre-configured heuristic agent variants.

    Returns 8 variants with different play styles:
    - Aggressive: Prioritizes bar entry, plays aggressively
    - Defensive: Conservative play, lower aggression
    - Queue Controller: Emphasizes queue front positioning
    - Skunk Specialist: Values skunk-related plays higher
    - Noisy/Human-like: Adds random noise for bounded rationality
    - OnlineStrategies: Reactive counter-play and opponent tracking
    - OutcomeHeuristic: Forward simulation with hand-tuned weights
    - DistilledOutcomeHeuristic: Forward simulation with PPO-extracted weights
    """
    from .outcome_heuristic import DistilledOutcomeHeuristic, OutcomeHeuristic

    variants: list[Agent] = []

    # 1. Aggressive variant
    aggressive_config = HeuristicConfig(
        bar_weight=3.0,
        aggression=0.8,
    )
    variants.append(HeuristicAgent(config=aggressive_config))

    # 2. Defensive variant
    defensive_config = HeuristicConfig(
        bar_weight=1.0,
        aggression=0.2,
    )
    variants.append(HeuristicAgent(config=defensive_config))

    # 3. Queue controller variant
    queue_controller_config = HeuristicConfig(
        queue_front_weight=2.0,
    )
    variants.append(HeuristicAgent(config=queue_controller_config))

    # 4. Skunk specialist variant
    skunk_specialist_config = HeuristicConfig(
        species_weights={"skunk": 2.0},
    )
    variants.append(HeuristicAgent(config=skunk_specialist_config))

    # 5. Noisy/human-like variant
    noisy_config = HeuristicConfig(
        noise_epsilon=0.15,
    )
    variants.append(HeuristicAgent(config=noisy_config))

    # 6. OnlineStrategies - reactive counter-play
    variants.append(OnlineStrategies())

    # 7. OutcomeHeuristic - forward simulation with hand-tuned weights
    variants.append(OutcomeHeuristic())

    # 8. DistilledOutcomeHeuristic - forward simulation with PPO-extracted weights
    variants.append(DistilledOutcomeHeuristic())

    return variants


__all__ = [
    "HeuristicAgent",
    "HeuristicConfig",
    "MaterialEvaluator",
    "OnlineStrategies",
    "create_heuristic_variants",
]
