"""Heuristic-based agent implementation."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from _01_simulator import actions, engine, state
from _01_simulator.exceptions import BeastyBarError

from .base import Agent


@dataclass
class MaterialEvaluator:
    """Configurable position evaluator using material advantage."""

    bar_weight: float = 2.0
    queue_front_weight: float = 1.1
    queue_back_weight: float = 0.3
    thats_it_weight: float = -0.5
    hand_weight: float = 0.1

    def __call__(self, game_state: state.State, player: int) -> float:
        """Evaluate the position from player's perspective."""
        score = 0.0

        # Score cards in beasty bar
        for card in game_state.zones.beasty_bar:
            weight = self.bar_weight if card.owner == player else -self.bar_weight
            score += weight * card.points

        # Score cards in queue (position-weighted)
        queue = game_state.zones.queue
        queue_len = len(queue)
        for i, card in enumerate(queue):
            # Front of queue is better (closer to entering bar)
            position_factor = 1.0 - (i / max(queue_len, 1))
            weight = self.queue_front_weight * position_factor + self.queue_back_weight * (1 - position_factor)
            if card.owner != player:
                weight = -weight
            score += weight * card.points

        # Score cards in that's it (negative)
        for card in game_state.zones.thats_it:
            weight = self.thats_it_weight if card.owner == player else -self.thats_it_weight
            score += weight * card.points

        # Small bonus for cards in hand (potential)
        for card in game_state.players[player].hand:
            score += self.hand_weight * card.points

        return score


EvaluatorFn = Callable[[state.State, int], float]


class HeuristicAgent(Agent):
    """Strong heuristic agent with species-specific strategies."""

    def __init__(
        self,
        evaluator: EvaluatorFn | None = None,
        seed: int | None = None,
        lookahead: bool = True,
    ):
        self._evaluator = evaluator or MaterialEvaluator()
        self._rng = random.Random(seed)
        self._lookahead = lookahead

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


__all__ = ["HeuristicAgent", "MaterialEvaluator"]
