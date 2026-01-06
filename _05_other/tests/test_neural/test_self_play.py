"""
Test suite for self-play game generation.

Tests the generation of training trajectories through self-play,
including parallel game execution and trajectory formatting.
"""

import pytest
import torch

from _02_agents.neural.network import create_network
from _02_agents.neural.utils import NetworkConfig
from _03_training.self_play import (
    GameTrajectory,
    compute_stats,
    generate_games,
    play_game,
    trajectories_to_transitions,
)


@pytest.fixture
def test_network():
    """Create small network for testing."""
    config = NetworkConfig(hidden_dim=32, num_heads=2, num_layers=1, dropout=0.0)
    return create_network(config)


def test_play_game_completes(test_network):
    """Verify self-play games complete successfully."""
    # Play a single game
    trajectory = play_game(
        network=test_network,
        seed=42,
        temperature=1.0,
        device=torch.device("cpu"),
    )

    # Verify trajectory is valid
    assert isinstance(trajectory, GameTrajectory)
    assert trajectory.game_length > 0, "Game should have at least one step"
    assert trajectory.total_steps() > 0, "Should have recorded steps"

    # Verify winner is valid (0, 1, or None for draw)
    assert trajectory.winner in [0, 1, None], f"Invalid winner: {trajectory.winner}"

    # Verify both players have steps
    assert len(trajectory.steps_p0) > 0, "Player 0 should have steps"
    assert len(trajectory.steps_p1) > 0, "Player 1 should have steps"

    # Verify final scores
    assert len(trajectory.final_scores) == 2, "Should have 2 final scores"
    assert all(isinstance(s, int) for s in trajectory.final_scores), "Scores should be integers"


def test_trajectory_structure(test_network):
    """Verify trajectory data has correct format."""
    trajectory = play_game(test_network, seed=123, temperature=1.0)

    # Check player 0 steps
    for step in trajectory.steps_p0:
        assert step.observation.shape[0] > 0, "Observation should have data"
        assert step.action_mask.shape[0] > 0, "Action mask should have data"
        assert isinstance(step.action, int), "Action should be int"
        assert 0.0 <= step.action_prob <= 1.0, "Action prob should be in [0, 1]"
        assert isinstance(step.value, float), "Value should be float"
        assert isinstance(step.reward, float), "Reward should be float"
        assert isinstance(step.done, bool), "Done should be bool"

    # Check player 1 steps
    for step in trajectory.steps_p1:
        assert step.observation.shape[0] > 0, "Observation should have data"
        assert step.action_mask.shape[0] > 0, "Action mask should have data"
        assert isinstance(step.action, int), "Action should be int"
        assert 0.0 <= step.action_prob <= 1.0, "Action prob should be in [0, 1]"
        assert isinstance(step.value, float), "Value should be float"
        assert isinstance(step.reward, float), "Reward should be float"
        assert isinstance(step.done, bool), "Done should be bool"

    # Last step should have done=True
    if len(trajectory.steps_p0) > 0:
        assert trajectory.steps_p0[-1].done, "Last step of p0 should be terminal"
    if len(trajectory.steps_p1) > 0:
        assert trajectory.steps_p1[-1].done, "Last step of p1 should be terminal"


def test_trajectory_rewards(test_network):
    """Verify trajectory rewards are calculated correctly."""
    trajectory = play_game(test_network, seed=456, temperature=1.0)

    # Get final rewards for each player
    if len(trajectory.steps_p0) > 0:
        final_reward_p0 = trajectory.steps_p0[-1].reward
        # Should be +1 (win), -1 (loss), or 0 (draw)
        assert final_reward_p0 in [-1.0, 0.0, 1.0], \
            f"Invalid final reward for p0: {final_reward_p0}"

    if len(trajectory.steps_p1) > 0:
        final_reward_p1 = trajectory.steps_p1[-1].reward
        assert final_reward_p1 in [-1.0, 0.0, 1.0], \
            f"Invalid final reward for p1: {final_reward_p1}"

    # Rewards should be opposite (zero-sum)
    if len(trajectory.steps_p0) > 0 and len(trajectory.steps_p1) > 0:
        final_reward_p0 = trajectory.steps_p0[-1].reward
        final_reward_p1 = trajectory.steps_p1[-1].reward

        if trajectory.winner == 0:
            assert final_reward_p0 == 1.0, "Winner should get +1"
            assert final_reward_p1 == -1.0, "Loser should get -1"
        elif trajectory.winner == 1:
            assert final_reward_p0 == -1.0, "Loser should get -1"
            assert final_reward_p1 == 1.0, "Winner should get +1"
        elif trajectory.winner is None:
            assert final_reward_p0 == 0.0, "Draw should give 0"
            assert final_reward_p1 == 0.0, "Draw should give 0"


def test_parallel_generation(test_network):
    """Verify parallel self-play execution (currently sequential)."""
    num_games = 5

    # Generate batch of games
    trajectories = generate_games(
        network=test_network,
        num_games=num_games,
        temperature=1.0,
        device=torch.device("cpu"),
        seeds=[100 + i for i in range(num_games)],
    )

    # Verify we got the right number of games
    assert len(trajectories) == num_games, \
        f"Expected {num_games} games, got {len(trajectories)}"

    # Verify each trajectory is valid
    for i, traj in enumerate(trajectories):
        assert isinstance(traj, GameTrajectory), f"Game {i} is not a GameTrajectory"
        assert traj.game_length > 0, f"Game {i} has zero length"
        assert traj.total_steps() > 0, f"Game {i} has no steps"

    # Verify stats computation
    stats = compute_stats(trajectories)
    assert stats.games_played == num_games
    assert stats.p0_wins + stats.p1_wins + stats.draws == num_games
    assert stats.total_steps > 0, "Should have collected steps"

    # Convert to transitions
    transitions = trajectories_to_transitions(trajectories)
    assert len(transitions) > 0, "Should have transitions"
    assert len(transitions) == sum(t.total_steps() for t in trajectories), \
        "Transition count should match total steps"
