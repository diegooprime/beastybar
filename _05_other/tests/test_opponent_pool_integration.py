"""Integration tests for the opponent pool system.

Tests the components working together:
- GameGenerator with new opponent types
- Trainer initialization with new config
- Short training loops
- Edge cases with real components
"""

import tempfile
from pathlib import Path

import pytest
import torch


# ============================================================================
# Integration Tests: GameGenerator with opponent types
# ============================================================================

class TestGameGeneratorIntegration:
    """Test GameGenerator with various opponent types."""

    @pytest.fixture
    def network(self):
        """Create a test network."""
        from _02_agents.neural.network import BeastyBarNetwork
        from _02_agents.neural.utils import NetworkConfig

        config = NetworkConfig()
        return BeastyBarNetwork(config)

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    def test_generate_games_with_heuristic_opponent(self, network, device):
        """Generate games against heuristic opponent."""
        from _03_training.game_generator import GameGenerator
        from _03_training.opponent_pool import OpponentConfig, OpponentPool

        # Create pool with heuristic only
        config = OpponentConfig(
            current_weight=0.0,
            checkpoint_weight=0.0,
            random_weight=0.0,
            heuristic_weight=1.0,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)

        generator = GameGenerator(
            network=network,
            device=device,
            opponent_pool=pool,
            temperature=1.0,
        )

        # Generate a few games
        transitions, trajectories, opponent_name, win_rate = generator.generate_games(
            num_games=4,
            shaped_rewards=False,
        )

        assert opponent_name == "heuristic"
        assert len(transitions) > 0
        assert len(trajectories) > 0
        assert 0.0 <= win_rate <= 1.0

    def test_generate_games_with_stats_tracker(self, network, device):
        """Generate games and verify stats tracker is updated."""
        from _03_training.game_generator import GameGenerator
        from _03_training.opponent_pool import OpponentConfig, OpponentPool
        from _03_training.opponent_statistics import OpponentStatsTracker

        config = OpponentConfig(
            current_weight=0.0,
            checkpoint_weight=0.0,
            random_weight=1.0,  # Random opponent
            heuristic_weight=0.0,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)
        stats_tracker = OpponentStatsTracker()

        generator = GameGenerator(
            network=network,
            device=device,
            opponent_pool=pool,
            stats_tracker=stats_tracker,
            temperature=1.0,
        )

        # Generate games
        transitions, trajectories, opponent_name, win_rate = generator.generate_games(
            num_games=4,
            iteration=1,
        )

        assert opponent_name == "random"

        # Stats should be updated
        stats = stats_tracker.get_stats("random")
        assert stats.games == 4

    def test_generate_games_self_play_returns_half_win_rate(self, network, device):
        """Self-play should return 0.5 win rate."""
        from _03_training.game_generator import GameGenerator
        from _03_training.opponent_pool import OpponentConfig, OpponentPool

        config = OpponentConfig(
            current_weight=1.0,
            checkpoint_weight=0.0,
            random_weight=0.0,
            heuristic_weight=0.0,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)

        generator = GameGenerator(
            network=network,
            device=device,
            opponent_pool=pool,
            temperature=1.0,
        )

        transitions, trajectories, opponent_name, win_rate = generator.generate_games(
            num_games=4,
        )

        assert opponent_name == "current"
        # Self-play always returns 0.5
        assert win_rate == 0.5

    def test_generate_games_with_checkpoint_opponent(self, network, device):
        """Generate games against a checkpoint opponent."""
        from _03_training.game_generator import GameGenerator
        from _03_training.opponent_pool import OpponentConfig, OpponentPool
        from _02_agents.neural.utils import NetworkConfig

        config = OpponentConfig(
            current_weight=0.0,
            checkpoint_weight=1.0,
            random_weight=0.0,
            heuristic_weight=0.0,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)

        # Add a checkpoint
        pool.add_checkpoint(
            state_dict=network.state_dict(),
            iteration=100,
        )

        network_config = NetworkConfig()
        generator = GameGenerator(
            network=network,
            device=device,
            opponent_pool=pool,
            network_config=network_config,
            temperature=1.0,
        )

        transitions, trajectories, opponent_name, win_rate = generator.generate_games(
            num_games=4,
        )

        assert "checkpoint" in opponent_name
        assert len(transitions) > 0


class TestTrainerIntegration:
    """Test Trainer initialization with new config options."""

    def test_trainer_creates_without_errors(self):
        """Trainer creates without errors with new config options."""
        from _03_training.trainer import Trainer, TrainingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                total_iterations=2,
                games_per_iteration=4,
                checkpoint_dir=tmpdir,
            )

            # Should create without errors
            trainer = Trainer(config)
            assert trainer is not None
            assert trainer.network is not None

    def test_trainer_with_opponent_pool(self):
        """Trainer initializes opponent pool correctly."""
        from _03_training.trainer import Trainer, TrainingConfig
        from _03_training.opponent_pool import OpponentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            opponent_config = OpponentConfig(
                current_weight=0.6,
                checkpoint_weight=0.2,
                random_weight=0.1,
                heuristic_weight=0.1,
                mcts_weight=0.0,
            )

            config = TrainingConfig(
                total_iterations=2,
                games_per_iteration=4,
                checkpoint_dir=tmpdir,
                opponent_config=opponent_config,
            )

            trainer = Trainer(config)

            assert trainer.opponent_pool is not None
            assert trainer.opponent_pool.config.current_weight == 0.6


class TestShortTrainingLoop:
    """Test short training loops to verify no crashes."""

    def test_2_iterations_no_crash(self):
        """Run 2 iterations of training without crashes."""
        from _03_training.trainer import Trainer, TrainingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                total_iterations=2,
                games_per_iteration=4,
                checkpoint_dir=tmpdir,
                checkpoint_frequency=10,  # Don't checkpoint during short test
            )

            trainer = Trainer(config)
            trainer.train()

            assert trainer._iteration >= 2
            assert trainer._total_games_played >= 8

    def test_training_with_opponent_pool_no_crash(self):
        """Training with opponent pool doesn't crash."""
        from _03_training.trainer import Trainer, TrainingConfig
        from _03_training.opponent_pool import OpponentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            opponent_config = OpponentConfig(
                current_weight=0.5,
                checkpoint_weight=0.0,  # No checkpoints yet
                random_weight=0.25,
                heuristic_weight=0.25,
                mcts_weight=0.0,
            )

            config = TrainingConfig(
                total_iterations=3,
                games_per_iteration=4,
                checkpoint_dir=tmpdir,
                opponent_config=opponent_config,
            )

            trainer = Trainer(config)
            trainer.train()

            # Should complete without errors
            assert trainer._iteration >= 3


class TestOpponentPoolSampling:
    """Test that opponent pool actually samples different opponents."""

    def test_samples_different_opponent_types(self):
        """Pool samples different opponent types according to weights."""
        from _03_training.opponent_pool import OpponentConfig, OpponentPool, OpponentType

        config = OpponentConfig(
            current_weight=0.33,
            checkpoint_weight=0.0,
            random_weight=0.33,
            heuristic_weight=0.34,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)

        # Sample many times
        type_counts = {t: 0 for t in OpponentType}
        for _ in range(300):
            sampled = pool.sample_opponent()
            type_counts[sampled.opponent_type] += 1

        # Should have sampled all three types
        assert type_counts[OpponentType.CURRENT] > 0
        assert type_counts[OpponentType.RANDOM] > 0
        assert type_counts[OpponentType.HEURISTIC] > 0

        # Checkpoint and MCTS should be zero
        assert type_counts[OpponentType.CHECKPOINT] == 0
        assert type_counts[OpponentType.MCTS] == 0


class TestHeuristicVariantsInPool:
    """Test heuristic variants work in opponent pool context."""

    def test_heuristic_variants_are_playable(self):
        """Heuristic variants can play games."""
        from _01_simulator import state, engine
        from _02_agents.heuristic import create_heuristic_variants

        variants = create_heuristic_variants()
        game_state = state.initial_state(seed=42)

        for variant in variants:
            # Get legal actions and select one
            legal = list(engine.legal_actions(game_state, player=0))
            action = variant.select_action(game_state, legal)

            assert action is not None
            assert action in legal


class TestStatsTrackerWithRealGames:
    """Test stats tracker with real game outcomes."""

    def test_tracker_records_real_game_results(self):
        """Stats tracker correctly records results from real games."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker(window_size=100)

        # Simulate some game results
        game_results = [
            ("heuristic", "win"),
            ("heuristic", "loss"),
            ("heuristic", "win"),
            ("random", "win"),
            ("random", "win"),
            ("checkpoint_50", "loss"),
        ]

        tracker.update_batch(game_results, iteration=1)

        heuristic_stats = tracker.get_stats("heuristic")
        assert heuristic_stats.wins == 2
        assert heuristic_stats.losses == 1
        assert abs(heuristic_stats.win_rate - 2 / 3) < 0.01

        random_stats = tracker.get_stats("random")
        assert random_stats.wins == 2
        assert random_stats.win_rate == 1.0


# ============================================================================
# Edge Cases with Real Components
# ============================================================================

class TestEdgeCasesWithRealComponents:
    """Test edge cases using real components."""

    def test_mcts_without_network_raises_error(self):
        """Sampling MCTS before set_mcts_network() raises error."""
        from _03_training.opponent_pool import (
            MCTSOpponentConfig,
            OpponentConfig,
            OpponentPool,
            OpponentType,
        )

        mcts_configs = [MCTSOpponentConfig(name="test_mcts")]
        config = OpponentConfig(
            current_weight=0.0,
            checkpoint_weight=0.0,
            random_weight=0.0,
            heuristic_weight=0.0,
            mcts_weight=1.0,
            mcts_configs=mcts_configs,
        )
        pool = OpponentPool(config=config, seed=42)

        # Sampling MCTS without setting network should raise error
        # because sample_opponent() accesses mcts_agents property
        with pytest.raises(RuntimeError, match="MCTS network not set"):
            pool.sample_opponent()

    def test_mcts_with_network_works(self):
        """MCTS sampling works after set_mcts_network()."""
        from _02_agents.neural.network import BeastyBarNetwork
        from _02_agents.neural.utils import NetworkConfig
        from _03_training.opponent_pool import (
            MCTSOpponentConfig,
            OpponentConfig,
            OpponentPool,
            OpponentType,
        )

        mcts_configs = [MCTSOpponentConfig(name="test_mcts", num_simulations=10)]
        config = OpponentConfig(
            current_weight=0.0,
            checkpoint_weight=0.0,
            random_weight=0.0,
            heuristic_weight=0.0,
            mcts_weight=1.0,
            mcts_configs=mcts_configs,
        )
        pool = OpponentPool(config=config, seed=42)

        # Set the network
        network = BeastyBarNetwork(NetworkConfig())
        pool.set_mcts_network(network)

        # Now sampling should work
        sampled = pool.sample_opponent()
        assert sampled.opponent_type == OpponentType.MCTS
        assert sampled.mcts_config_name == "test_mcts"

        # Should be able to access agents
        agents = pool.mcts_agents
        assert "test_mcts" in agents

    def test_checkpoint_max_limit_enforced(self):
        """Checkpoint pool enforces max_checkpoints limit."""
        from _03_training.opponent_pool import OpponentConfig, OpponentPool

        config = OpponentConfig(max_checkpoints=3)
        pool = OpponentPool(config=config)

        # Add 5 checkpoints
        for i in range(5):
            pool.add_checkpoint(state_dict={"iter": i}, iteration=i * 10)

        # Should only have 3
        assert len(pool.checkpoints) == 3

        # Should be the 3 most recent
        iterations = [cp.iteration for cp in pool.checkpoints]
        assert iterations == [20, 30, 40]


class TestExploitPatchManagerIntegration:
    """Test ExploitPatchManager with real components."""

    def test_manager_initializes_correctly(self):
        """Manager initializes correctly with real config."""
        from _03_training.exploit_patch_cycle import CycleConfig, ExploitPatchManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CycleConfig(
                cycle_interval=100,
                plateau_window=20,
            )
            manager = ExploitPatchManager(
                config=config,
                checkpoints_dir=Path(tmpdir),
            )

            assert manager.config.cycle_interval == 100
            assert (Path(tmpdir) / "exploiters").exists()

    def test_plateau_detection_with_varying_data(self):
        """Plateau detection works with varying win rate data."""
        from _03_training.exploit_patch_cycle import CycleConfig, ExploitPatchManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CycleConfig(plateau_window=20, plateau_threshold=0.03)
            manager = ExploitPatchManager(config=config, checkpoints_dir=Path(tmpdir))

            # First phase: rapid improvement (no plateau)
            for i in range(10):
                manager.record_win_rate(0.5 + 0.02 * i, iteration=i)

            # Second phase: stagnation (plateau)
            for i in range(10, 30):
                manager.record_win_rate(0.70 + 0.001 * (i % 5), iteration=i)

            # Should detect plateau after stagnation phase
            assert manager.detect_plateau() is True


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
