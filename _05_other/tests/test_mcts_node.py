"""Tests for Simple MCTS Node implementation."""

from __future__ import annotations

import pytest

from _01_simulator import actions, engine, state
from _02_agents.mcts.simple_node import SimpleMCTSNode as MCTSNode


class TestMCTSNodeBasics:
    """Test basic node properties and initialization."""

    def test_node_creation(self) -> None:
        """Test creating a node with initial state."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        assert node.state == game_state
        assert node.parent is None
        assert node.children == {}
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.0
        assert not node.is_expanded
        assert not node.is_terminal

    def test_node_with_prior(self) -> None:
        """Test creating a node with non-zero prior."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state, prior=0.25)

        assert node.prior == 0.25

    def test_terminal_detection(self) -> None:
        """Test terminal state detection."""
        # Create a game state near the end
        game_state = state.initial_state(seed=42)

        # Play until terminal
        while not engine.is_terminal(game_state):
            legal = list(engine.legal_actions(game_state, game_state.active_player))
            if not legal:
                break
            action = legal[0]
            game_state = engine.step(game_state, action)

        node = MCTSNode(state=game_state)
        assert node.is_terminal


class TestMCTSNodeExpansion:
    """Test node expansion with policy priors."""

    def test_expand_with_uniform_priors(self) -> None:
        """Test expanding a node with uniform policy priors."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        # Get legal actions and create uniform priors
        legal = list(engine.legal_actions(game_state, 0))
        priors = {action: 1.0 / len(legal) for action in legal}

        # Expand node
        node.expand(priors)

        assert node.is_expanded
        assert len(node.children) == len(legal)

        # Check each child has correct prior
        for _action, child in node.children.items():
            assert child.parent == node
            assert child.prior == pytest.approx(1.0 / len(legal))
            assert child.visit_count == 0
            assert child.value_sum == 0.0

    def test_expand_with_varied_priors(self) -> None:
        """Test expanding with non-uniform priors."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        legal = list(engine.legal_actions(game_state, 0))
        # Create biased priors
        priors = {legal[0]: 0.5}
        remaining = 0.5 / (len(legal) - 1)
        for action in legal[1:]:
            priors[action] = remaining

        node.expand(priors)

        assert node.children[legal[0]].prior == pytest.approx(0.5)
        for action in legal[1:]:
            assert node.children[action].prior == pytest.approx(remaining)

    def test_expand_terminal_node(self) -> None:
        """Test expanding a terminal node does nothing."""
        # Create terminal state
        game_state = state.initial_state(seed=42)
        while not engine.is_terminal(game_state):
            legal = list(engine.legal_actions(game_state, game_state.active_player))
            if not legal:
                break
            game_state = engine.step(game_state, legal[0])

        node = MCTSNode(state=game_state)
        node.expand({})  # Empty priors for terminal state

        assert not node.is_expanded  # Terminal nodes don't expand
        assert len(node.children) == 0

    def test_expand_already_expanded_raises_error(self) -> None:
        """Test that expanding an already expanded node raises error."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        legal = list(engine.legal_actions(game_state, 0))
        priors = {action: 1.0 / len(legal) for action in legal}
        node.expand(priors)

        with pytest.raises(ValueError, match="already expanded"):
            node.expand(priors)

    def test_expand_with_mismatched_actions(self) -> None:
        """Test that expanding with wrong actions raises error."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        # Create priors for wrong actions
        wrong_priors = {actions.Action(hand_index=999): 1.0}

        with pytest.raises(ValueError, match="don't match legal actions"):
            node.expand(wrong_priors)


class TestMCTSNodeUCB:
    """Test UCB score calculations."""

    def test_ucb_unvisited_node(self) -> None:
        """Test UCB score for unvisited node."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state, prior=0.25)

        parent_visits = 100
        c_puct = 1.414

        score = node.ucb_score(parent_visits, c_puct)

        # Unvisited: value + c_puct * prior * sqrt(parent_visits)
        expected = 0.0 + 1.414 * 0.25 * (100**0.5)
        assert score == pytest.approx(expected)

    def test_ucb_visited_node(self) -> None:
        """Test UCB score for visited node."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state, prior=0.25)

        # Simulate visits
        node.visit_count = 10
        node.value_sum = 5.0  # Average value = 0.5

        parent_visits = 100
        c_puct = 1.414

        score = node.ucb_score(parent_visits, c_puct)

        # Visited: value + c_puct * prior * sqrt(parent) / (1 + visits)
        expected = 0.5 + 1.414 * 0.25 * (100**0.5) / (1 + 10)
        assert score == pytest.approx(expected)

    def test_ucb_high_exploration(self) -> None:
        """Test UCB with higher exploration constant."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state, prior=0.25)
        node.visit_count = 10
        node.value_sum = 5.0

        score_low = node.ucb_score(100, c_puct=1.0)
        score_high = node.ucb_score(100, c_puct=2.0)

        assert score_high > score_low


class TestMCTSNodeSelection:
    """Test child node selection."""

    def test_select_best_child(self) -> None:
        """Test selecting child with highest UCB."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        legal = list(engine.legal_actions(game_state, 0))
        priors = {action: 1.0 / len(legal) for action in legal}
        node.expand(priors)

        # Visit parent once so UCB can be calculated
        node.visit_count = 1

        action, child = node.select_child(c_puct=1.414)

        assert action in node.children
        assert child == node.children[action]

    def test_select_unexpanded_raises_error(self) -> None:
        """Test that selecting from unexpanded node raises error."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        with pytest.raises(ValueError, match="unexpanded"):
            node.select_child()

    def test_select_prefers_high_prior(self) -> None:
        """Test that selection prefers higher prior when unvisited."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)
        node.visit_count = 10  # Parent has visits

        legal = list(engine.legal_actions(game_state, 0))
        # Give one action very high prior
        priors = {legal[0]: 0.8}
        remaining = 0.2 / (len(legal) - 1)
        for action in legal[1:]:
            priors[action] = remaining

        node.expand(priors)

        action, child = node.select_child(c_puct=1.414)

        # Should select high prior action
        assert action == legal[0]
        assert child == node.children[legal[0]]


class TestMCTSNodeBackup:
    """Test value backpropagation."""

    def test_backup_single_node(self) -> None:
        """Test backing up value to a single node."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        node.backup(0.5)

        assert node.visit_count == 1
        assert node.value_sum == pytest.approx(0.5)
        assert node.value == pytest.approx(0.5)

    def test_backup_multiple_times(self) -> None:
        """Test backing up multiple values."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        node.backup(0.5)
        node.backup(1.0)
        node.backup(-0.5)

        assert node.visit_count == 3
        assert node.value_sum == pytest.approx(1.0)
        assert node.value == pytest.approx(1.0 / 3.0)

    def test_backup_propagates_to_parent(self) -> None:
        """Test backup propagates up tree with negation."""
        game_state = state.initial_state(seed=42)
        root = MCTSNode(state=game_state)

        legal = list(engine.legal_actions(game_state, 0))
        priors = {action: 1.0 / len(legal) for action in legal}
        root.expand(priors)

        child = next(iter(root.children.values()))
        child.backup(1.0)

        # Child gets positive value
        assert child.visit_count == 1
        assert child.value_sum == pytest.approx(1.0)

        # Parent gets negated value (opponent's perspective)
        assert root.visit_count == 1
        assert root.value_sum == pytest.approx(-1.0)

    def test_backup_propagates_full_path(self) -> None:
        """Test backup propagates through multiple levels."""
        game_state = state.initial_state(seed=42)
        root = MCTSNode(state=game_state)

        # Expand root
        legal = list(engine.legal_actions(game_state, 0))
        priors = {action: 1.0 / len(legal) for action in legal}
        root.expand(priors)

        # Get first child and expand it
        child1 = next(iter(root.children.values()))
        legal2 = list(engine.legal_actions(child1.state, child1.state.active_player))
        priors2 = {action: 1.0 / len(legal2) for action in legal2}
        child1.expand(priors2)

        # Get grandchild and backup
        child2 = next(iter(child1.children.values()))
        child2.backup(1.0)

        # Check propagation with alternating signs
        assert child2.visit_count == 1
        assert child2.value_sum == pytest.approx(1.0)
        assert child1.visit_count == 1
        assert child1.value_sum == pytest.approx(-1.0)
        assert root.visit_count == 1
        assert root.value_sum == pytest.approx(1.0)  # Same player as child2


class TestMCTSNodeBestChild:
    """Test best child selection by visits."""

    def test_best_child_by_visits(self) -> None:
        """Test getting child with most visits."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        legal = list(engine.legal_actions(game_state, 0))
        priors = {action: 1.0 / len(legal) for action in legal}
        node.expand(priors)

        # Visit children different amounts
        children = list(node.children.items())
        children[0][1].visit_count = 5
        children[1][1].visit_count = 10
        children[2][1].visit_count = 3

        result = node.best_child_by_visits()
        assert result is not None
        _action, child = result
        assert child == children[1][1]
        assert child.visit_count == 10

    def test_best_child_unexpanded_returns_none(self) -> None:
        """Test that unexpanded node returns None."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        result = node.best_child_by_visits()
        assert result is None


class TestMCTSNodeValue:
    """Test value property calculations."""

    def test_value_unvisited(self) -> None:
        """Test value of unvisited node."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        assert node.value == 0.0

    def test_value_after_visits(self) -> None:
        """Test value calculation after visits."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        node.visit_count = 10
        node.value_sum = 7.5

        assert node.value == pytest.approx(0.75)

    def test_value_updates_with_backup(self) -> None:
        """Test that value updates correctly with backups."""
        game_state = state.initial_state(seed=42)
        node = MCTSNode(state=game_state)

        node.backup(1.0)
        assert node.value == pytest.approx(1.0)

        node.backup(0.5)
        assert node.value == pytest.approx(0.75)

        node.backup(0.0)
        assert node.value == pytest.approx(0.5)
