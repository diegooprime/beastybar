"""Monte Carlo Tree Search agent with Information Set handling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from _01_simulator import actions, engine, rules, state
from _01_simulator.exceptions import BeastyBarError

from .base import Agent

# Safety limit to prevent infinite rollouts in case of terminal detection bugs
_MAX_ROLLOUT_DEPTH = 100

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""

    state: state.State
    action: actions.Action | None = None
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list[actions.Action] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return engine.is_terminal(self.state)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def ucb1(self, exploration: float = 1.414) -> float:
        """Calculate UCB1 score for node selection."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return 0.0
        exploitation = self.total_reward / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_term


class MCTSAgent(Agent):
    """Monte Carlo Tree Search agent with determinization for hidden info."""

    def __init__(
        self,
        iterations: int = 1000,
        exploration: float = 1.414,
        seed: int | None = None,
        determinizations: int = 10,
    ):
        """Initialize MCTS agent.

        Args:
            iterations: Number of MCTS iterations per move.
            exploration: UCB1 exploration constant.
            seed: Random seed for reproducibility.
            determinizations: Number of opponent hand samples to average over.
        """
        self._iterations = iterations
        self._exploration = exploration
        self._rng = random.Random(seed)
        self._determinizations = determinizations

    def select_action(
        self,
        game_state: state.State,
        legal_actions: Sequence[actions.Action],
    ) -> actions.Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        player = game_state.active_player

        # Aggregate scores across multiple determinizations
        action_scores: dict[actions.Action, list[float]] = {a: [] for a in legal_actions}

        for _ in range(self._determinizations):
            # Sample a possible complete state (determinize hidden info)
            determinized = self._determinize(game_state, player)

            # Run MCTS on this determinization
            root = self._create_root(determinized, player)
            for _ in range(self._iterations // self._determinizations):
                self._iterate(root, player)

            # Collect action scores
            for child in root.children:
                if child.action in action_scores and child.visits > 0:
                    action_scores[child.action].append(child.total_reward / child.visits)

        # Select action with best average score
        best_action = legal_actions[0]
        best_score = float("-inf")

        for action, scores in action_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = action

        return best_action

    def _determinize(self, game_state: state.State, perspective: int) -> state.State:
        """Create a determinized state by sampling opponent's hidden cards."""
        opponent = 1 - perspective

        # Get all cards we know about
        known_species = set()
        for card in game_state.zones.queue:
            known_species.add((card.owner, card.species))
        for card in game_state.zones.beasty_bar:
            known_species.add((card.owner, card.species))
        for card in game_state.zones.thats_it:
            known_species.add((card.owner, card.species))

        # Our own cards
        for card in game_state.players[perspective].hand:
            known_species.add((perspective, card.species))
        for card in game_state.players[perspective].deck:
            known_species.add((perspective, card.species))

        # Figure out which opponent cards are unknown
        opp_hand = game_state.players[opponent].hand
        opp_deck = game_state.players[opponent].deck

        # Get all possible opponent cards
        all_opp_species = list(rules.BASE_DECK)
        for owner, species in known_species:
            if owner == opponent and species in all_opp_species:
                all_opp_species.remove(species)

        # Shuffle and assign to hand and deck
        self._rng.shuffle(all_opp_species)

        num_hand = len(opp_hand)
        num_deck = len(opp_deck)

        if len(all_opp_species) < num_hand + num_deck:
            # Not enough cards - use what we have
            return game_state

        new_hand = tuple(state.Card(owner=opponent, species=s) for s in all_opp_species[:num_hand])
        new_deck = tuple(state.Card(owner=opponent, species=s) for s in all_opp_species[num_hand : num_hand + num_deck])

        # Build new player states
        new_players = list(game_state.players)
        new_players[opponent] = state.PlayerState(deck=new_deck, hand=new_hand)

        return state.State(
            seed=game_state.seed,
            turn=game_state.turn,
            active_player=game_state.active_player,
            players=tuple(new_players),
            zones=game_state.zones,
        )

    def _create_root(self, game_state: state.State, player: int) -> MCTSNode:
        """Create root node for MCTS tree."""
        legal = list(engine.legal_actions(game_state, game_state.active_player))
        return MCTSNode(
            state=game_state,
            untried_actions=legal,
        )

    def _iterate(self, root: MCTSNode, player: int) -> None:
        """Perform one MCTS iteration: select, expand, simulate, backprop."""
        # Selection
        node = root
        while not node.is_terminal and node.is_fully_expanded:
            node = self._select_child(node)

        # Expansion
        if not node.is_terminal and node.untried_actions:
            node = self._expand(node)

        # Simulation
        reward = self._simulate(node.state, player)

        # Backpropagation
        self._backpropagate(node, reward)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1."""
        return max(node.children, key=lambda c: c.ucb1(self._exploration))

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by trying an untried action."""
        action = node.untried_actions.pop()
        next_state = engine.step(node.state, action)

        # Get legal actions for next state
        if not engine.is_terminal(next_state):
            legal = list(engine.legal_actions(next_state, next_state.active_player))
        else:
            legal = []

        child = MCTSNode(
            state=next_state,
            action=action,
            parent=node,
            untried_actions=legal,
        )
        node.children.append(child)
        return child

    def _simulate(self, game_state: state.State, player: int) -> float:
        """Run a heuristic-guided playout and return reward for player."""
        current = game_state
        depth = 0

        while not engine.is_terminal(current) and depth < _MAX_ROLLOUT_DEPTH:
            legal = list(engine.legal_actions(current, current.active_player))
            if not legal:
                break

            # Use simple heuristic: prefer actions that result in better material
            if len(legal) > 1 and self._rng.random() < 0.7:
                # 70% of the time, use greedy selection
                action = self._greedy_select(current, legal)
            else:
                action = self._rng.choice(legal)

            current = engine.step(current, action)
            depth += 1

        # Calculate reward with margin bonus
        scores = engine.score(current)
        margin = scores[player] - scores[1 - player]
        max_margin = 36  # Theoretical max point difference

        if scores[player] > scores[1 - player]:
            return 1.0 + 0.2 * (margin / max_margin)
        elif scores[player] < scores[1 - player]:
            return -1.0 + 0.2 * (margin / max_margin)
        return 0.0

    def _greedy_select(
        self,
        game_state: state.State,
        legal: list[actions.Action],
    ) -> actions.Action:
        """Select action greedily based on immediate material evaluation."""
        player = game_state.active_player
        best_action = legal[0]
        best_score = float("-inf")

        for action in legal:
            try:
                next_state = engine.step(game_state, action)
                score = self._evaluate_material(next_state, player)
                if score > best_score:
                    best_score = score
                    best_action = action
            except BeastyBarError:
                continue

        return best_action

    def _evaluate_material(self, game_state: state.State, player: int) -> float:
        """Quick material evaluation."""
        score = 0.0

        # Points in bar (most important)
        for card in game_state.zones.beasty_bar:
            if card.owner == player:
                score += 2.0 * card.points
            else:
                score -= 2.0 * card.points

        # Points in queue (position weighted)
        queue = game_state.zones.queue
        for i, card in enumerate(queue):
            weight = 1.0 - (i / max(len(queue), 1)) * 0.5
            if card.owner == player:
                score += weight * card.points
            else:
                score -= weight * card.points

        # Points lost
        for card in game_state.zones.thats_it:
            if card.owner == player:
                score -= 0.5 * card.points
            else:
                score += 0.5 * card.points

        return score

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree."""
        current: MCTSNode | None = node
        sign = 1.0

        while current is not None:
            current.visits += 1
            current.total_reward += sign * reward
            # Flip reward sign for opponent's nodes
            if current.parent and current.state.active_player != current.parent.state.active_player:
                sign *= -1
            current = current.parent


__all__ = ["MCTSAgent"]
