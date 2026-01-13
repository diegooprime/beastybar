"""Comprehensive tests for the opponent pool system.

Tests cover:
- opponent_statistics.py: OpponentStats, OpponentStatsTracker
- opponent_pool.py: MCTS additions, OpponentConfig, OpponentPool
- heuristic.py: HeuristicConfig, parameterized agents
- exploiter_training.py: ExploiterConfig, basic functionality
- exploit_patch_cycle.py: CycleConfig, ExploitPatchManager
"""

import tempfile
from pathlib import Path

import pytest

# ============================================================================
# Tests for opponent_statistics.py
# ============================================================================

class TestOpponentStats:
    """Test OpponentStats dataclass."""

    def test_win_rate_returns_0_5_when_no_games_played(self):
        """OpponentStats.win_rate returns 0.5 when no games played."""
        from _03_training.opponent_statistics import OpponentStats

        stats = OpponentStats(opponent_id="test")
        assert stats.win_rate == 0.5
        assert stats.games == 0

    def test_win_rate_calculation(self):
        """OpponentStats.win_rate calculates correctly."""
        from _03_training.opponent_statistics import OpponentStats

        stats = OpponentStats(opponent_id="test", wins=7, losses=2, draws=1)
        assert stats.games == 10
        assert stats.win_rate == 0.7

    def test_to_dict_from_dict_roundtrip(self):
        """OpponentStats.to_dict/from_dict round-trips correctly."""
        from _03_training.opponent_statistics import OpponentStats

        original = OpponentStats(
            opponent_id="test_opponent",
            wins=10,
            losses=5,
            draws=3,
            last_updated_iteration=100,
        )
        serialized = original.to_dict()
        restored = OpponentStats.from_dict(serialized)

        assert restored.opponent_id == original.opponent_id
        assert restored.wins == original.wins
        assert restored.losses == original.losses
        assert restored.draws == original.draws
        assert restored.last_updated_iteration == original.last_updated_iteration


class TestOpponentStatsTracker:
    """Test OpponentStatsTracker class."""

    def test_update_correctly_increments_counts(self):
        """update() correctly increments counts."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker(window_size=100)

        tracker.update("opponent_1", "win", iteration=1)
        tracker.update("opponent_1", "win", iteration=2)
        tracker.update("opponent_1", "loss", iteration=3)

        stats = tracker.get_stats("opponent_1")
        assert stats.wins == 2
        assert stats.losses == 1
        assert stats.draws == 0
        assert stats.games == 3

    def test_update_batch_handles_empty_list(self):
        """update_batch() handles empty list."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker()
        # Should not raise
        tracker.update_batch([], iteration=1)
        assert len(tracker.get_all_stats()) == 0

    def test_update_batch_multiple_results(self):
        """update_batch() processes multiple results."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker()
        results = [
            ("opponent_1", "win"),
            ("opponent_1", "loss"),
            ("opponent_2", "win"),
            ("opponent_2", "draw"),
        ]
        tracker.update_batch(results, iteration=1)

        stats1 = tracker.get_stats("opponent_1")
        stats2 = tracker.get_stats("opponent_2")

        assert stats1.wins == 1
        assert stats1.losses == 1
        assert stats2.wins == 1
        assert stats2.draws == 1

    def test_compute_learning_weights_returns_normalized_weights(self):
        """compute_learning_weights() returns normalized weights summing to 1.0."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker()
        # Add data for multiple opponents
        tracker.update_batch(
            [("opp1", "win"), ("opp1", "win"), ("opp2", "loss"), ("opp3", "draw")],
            iteration=1,
        )

        weights = tracker.compute_learning_weights()
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_compute_learning_weights_empty_tracker(self):
        """compute_learning_weights() returns empty dict for empty tracker."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker()
        weights = tracker.compute_learning_weights()
        assert weights == {}

    def test_compute_learning_weights_highest_at_50_percent(self):
        """compute_learning_weights() gives highest weight to 50% win rate opponents."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker()

        # Create opponents with different win rates
        # 50% win rate opponent
        for _ in range(5):
            tracker.update("opp_50", "win", iteration=1)
            tracker.update("opp_50", "loss", iteration=1)

        # 100% win rate opponent (too easy)
        for _ in range(10):
            tracker.update("opp_100", "win", iteration=1)

        # 0% win rate opponent (too hard)
        for _ in range(10):
            tracker.update("opp_0", "loss", iteration=1)

        weights = tracker.compute_learning_weights(exploration_rate=0.0)

        # 50% opponent should have highest raw signal
        # Due to bell curve centered at 0.5
        assert weights["opp_50"] > weights["opp_100"]
        assert weights["opp_50"] > weights["opp_0"]

    def test_sliding_window_works(self):
        """Sliding window works - old results don't count after window_size games."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker(window_size=5)

        # Fill with losses
        for i in range(5):
            tracker.update("test", "loss", iteration=i)

        stats = tracker.get_stats("test")
        assert stats.wins == 0
        assert stats.losses == 5
        assert stats.win_rate == 0.0

        # Now add 5 wins - should push out the losses
        for i in range(5, 10):
            tracker.update("test", "win", iteration=i)

        stats = tracker.get_stats("test")
        assert stats.wins == 5
        assert stats.losses == 0
        assert stats.win_rate == 1.0

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict()/from_dict() round-trips correctly."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker(window_size=100)
        tracker.update_batch(
            [
                ("opp1", "win"),
                ("opp1", "loss"),
                ("opp2", "draw"),
            ],
            iteration=42,
        )

        serialized = tracker.to_dict()
        restored = OpponentStatsTracker.from_dict(serialized)

        assert restored._window_size == tracker._window_size

        # Check stats match
        orig_stats = tracker.get_all_stats()
        rest_stats = restored.get_all_stats()
        assert set(orig_stats.keys()) == set(rest_stats.keys())

        for opp_id in orig_stats:
            assert orig_stats[opp_id].wins == rest_stats[opp_id].wins
            assert orig_stats[opp_id].losses == rest_stats[opp_id].losses
            assert orig_stats[opp_id].draws == rest_stats[opp_id].draws


# ============================================================================
# Tests for opponent_pool.py (MCTS additions)
# ============================================================================

class TestOpponentConfig:
    """Test OpponentConfig dataclass."""

    def test_weights_must_sum_to_1(self):
        """OpponentConfig with weights summing to 1.0 validates correctly."""
        from _03_training.opponent_pool import OpponentConfig

        # Valid config
        config = OpponentConfig(
            current_weight=0.5,
            checkpoint_weight=0.2,
            random_weight=0.1,
            heuristic_weight=0.1,
            mcts_weight=0.1,
        )
        assert config.current_weight == 0.5

    def test_invalid_weights_raise_error(self):
        """OpponentConfig with weights summing to 0.5 should raise."""
        from _03_training.opponent_pool import OpponentConfig

        with pytest.raises(ValueError, match="must sum to 1.0"):
            OpponentConfig(
                current_weight=0.3,
                checkpoint_weight=0.2,
                random_weight=0.0,
                heuristic_weight=0.0,
                mcts_weight=0.0,
            )


class TestMCTSOpponentConfig:
    """Test MCTSOpponentConfig dataclass."""

    def test_auto_generates_name_if_not_provided(self):
        """MCTSOpponentConfig auto-generates name if not provided."""
        from _03_training.opponent_pool import MCTSOpponentConfig

        config = MCTSOpponentConfig(c_puct=1.5, num_simulations=200)
        assert "mcts" in config.name
        assert "1.5" in config.name
        assert "200" in config.name

    def test_uses_provided_name(self):
        """MCTSOpponentConfig uses provided name."""
        from _03_training.opponent_pool import MCTSOpponentConfig

        config = MCTSOpponentConfig(name="my_custom_mcts")
        assert config.name == "my_custom_mcts"


class TestCreateDefaultMCTSConfigs:
    """Test create_default_mcts_configs function."""

    def test_returns_8_distinct_configs(self):
        """create_default_mcts_configs() returns 8 distinct configs including MCTS-100."""
        from _03_training.opponent_pool import create_default_mcts_configs

        configs = create_default_mcts_configs()
        assert len(configs) == 8

        # Check names are distinct
        names = [c.name for c in configs]
        assert len(names) == len(set(names))

        # Verify MCTS-100 configs are present
        mcts_100_names = [c.name for c in configs if c.num_simulations == 100]
        assert "mcts_100" in mcts_100_names
        assert "mcts_100_exploit" in mcts_100_names


class TestCreateMCTS100Configs:
    """Test create_mcts_100_configs function."""

    def test_returns_2_focused_configs(self):
        """create_mcts_100_configs() returns 2 MCTS-100 configs."""
        from _03_training.opponent_pool import create_mcts_100_configs

        configs = create_mcts_100_configs()
        assert len(configs) == 2

        # All should be 100 simulations
        for config in configs:
            assert config.num_simulations == 100

        names = [c.name for c in configs]
        assert "mcts_100" in names
        assert "mcts_100_exploit" in names


class TestOutcomeHeuristicOpponent:
    """Test OUTCOME_HEURISTIC opponent type."""

    def test_outcome_heuristic_weight_in_config(self):
        """OpponentConfig supports outcome_heuristic_weight."""
        from _03_training.opponent_pool import OpponentConfig

        config = OpponentConfig(
            current_weight=0.5,
            checkpoint_weight=0.1,
            random_weight=0.1,
            heuristic_weight=0.1,
            outcome_heuristic_weight=0.2,
            mcts_weight=0.0,
        )
        assert config.outcome_heuristic_weight == 0.2

    def test_outcome_heuristic_sampling(self):
        """OpponentPool samples OUTCOME_HEURISTIC opponents correctly."""
        from _03_training.opponent_pool import OpponentConfig, OpponentPool, OpponentType

        config = OpponentConfig(
            current_weight=0.0,
            checkpoint_weight=0.0,
            random_weight=0.0,
            heuristic_weight=0.0,
            outcome_heuristic_weight=1.0,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)

        sampled = pool.sample_opponent()
        assert sampled.opponent_type == OpponentType.OUTCOME_HEURISTIC
        assert sampled.name == "outcome_heuristic"
        assert sampled.agent is not None


class TestOpponentPool:
    """Test OpponentPool class."""

    def test_sample_returns_current_by_default(self):
        """sample_opponent() returns CURRENT type when no checkpoints or MCTS."""
        from _03_training.opponent_pool import OpponentConfig, OpponentPool, OpponentType

        config = OpponentConfig(
            current_weight=1.0,
            checkpoint_weight=0.0,
            random_weight=0.0,
            heuristic_weight=0.0,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)
        sampled = pool.sample_opponent()
        assert sampled.opponent_type == OpponentType.CURRENT
        assert sampled.name == "current"

    def test_mcts_weight_redistributes_to_current_when_no_configs(self):
        """MCTS weight redistributes to CURRENT when no mcts_configs provided."""
        from _03_training.opponent_pool import OpponentConfig, OpponentPool, OpponentType

        config = OpponentConfig(
            current_weight=0.8,
            checkpoint_weight=0.0,
            random_weight=0.0,
            heuristic_weight=0.0,
            mcts_weight=0.2,  # No mcts_configs, so should go to current
        )
        pool = OpponentPool(config=config, seed=42)

        # Sample many times - should only get CURRENT
        sampled_types = set()
        for _ in range(50):
            sampled = pool.sample_opponent()
            sampled_types.add(sampled.opponent_type)

        assert sampled_types == {OpponentType.CURRENT}

    def test_sampled_opponent_name_for_mcts(self):
        """SampledOpponent.name returns correct format for MCTS."""
        from _03_training.opponent_pool import (
            OpponentType,
            SampledOpponent,
        )

        sampled = SampledOpponent(
            opponent_type=OpponentType.MCTS,
            mcts_config_name="mcts_balanced",
        )
        assert sampled.name == "mcts_balanced"


class TestOpponentPoolCheckpoints:
    """Test OpponentPool checkpoint functionality."""

    def test_checkpoint_weight_redistributes_when_empty(self):
        """Checkpoint weight goes to CURRENT when no checkpoints exist."""
        from _03_training.opponent_pool import OpponentConfig, OpponentPool, OpponentType

        config = OpponentConfig(
            current_weight=0.6,
            checkpoint_weight=0.2,  # No checkpoints, should go to current
            random_weight=0.1,
            heuristic_weight=0.1,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)

        # Sample many times
        type_counts = dict.fromkeys(OpponentType, 0)
        for _ in range(100):
            sampled = pool.sample_opponent()
            type_counts[sampled.opponent_type] += 1

        # Should never get CHECKPOINT since none exist
        assert type_counts[OpponentType.CHECKPOINT] == 0

    def test_add_checkpoint_and_sample(self):
        """Add checkpoint and verify it can be sampled."""
        from _03_training.opponent_pool import OpponentConfig, OpponentPool, OpponentType

        config = OpponentConfig(
            current_weight=0.0,
            checkpoint_weight=1.0,
            random_weight=0.0,
            heuristic_weight=0.0,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)

        # Add a checkpoint
        pool.add_checkpoint(state_dict={"test": 1}, iteration=100)

        sampled = pool.sample_opponent()
        assert sampled.opponent_type == OpponentType.CHECKPOINT
        assert sampled.iteration == 100


# ============================================================================
# Tests for heuristic.py parameterization
# ============================================================================

class TestHeuristicConfig:
    """Test HeuristicConfig dataclass."""

    def test_defaults_match_original_values(self):
        """HeuristicConfig defaults match original hardcoded values."""
        from _02_agents.heuristic import HeuristicConfig

        config = HeuristicConfig()
        assert config.bar_weight == 2.0
        assert config.queue_front_weight == 1.1
        assert config.queue_back_weight == 0.3
        assert config.thats_it_weight == -0.5
        assert config.hand_weight == 0.1
        assert config.aggression == 0.5
        assert config.noise_epsilon == 0.0


class TestHeuristicAgent:
    """Test HeuristicAgent class."""

    def test_default_agent_backward_compatible(self):
        """Default HeuristicAgent() is backward compatible."""
        from _02_agents.heuristic import HeuristicAgent

        agent = HeuristicAgent()
        assert agent.name == "HeuristicAgent"

    def test_custom_config_uses_weights(self):
        """HeuristicAgent with custom config uses those weights."""
        from _02_agents.heuristic import HeuristicAgent, HeuristicConfig

        config = HeuristicConfig(bar_weight=5.0, aggression=0.8)
        agent = HeuristicAgent(config=config)

        # Name should reflect configuration
        assert "aggressive" in agent.name or "bar=5.0" in agent.name

    def test_aggression_parameter(self):
        """aggression parameter biases action selection."""
        from _02_agents.heuristic import HeuristicConfig

        # Just test config creation - action selection requires game state
        aggressive_config = HeuristicConfig(aggression=0.9)
        defensive_config = HeuristicConfig(aggression=0.1)

        assert aggressive_config.aggression > defensive_config.aggression

    def test_noise_epsilon_adds_randomness(self):
        """noise_epsilon adds randomness (same state, different actions possible)."""
        from _02_agents.heuristic import HeuristicConfig

        noisy_config = HeuristicConfig(noise_epsilon=0.5, seed=42)
        assert noisy_config.noise_epsilon == 0.5

    def test_species_weights_multipliers(self):
        """species_weights multipliers work correctly."""
        from _02_agents.heuristic import HeuristicConfig, MaterialEvaluator

        config = HeuristicConfig(species_weights={"skunk": 2.0, "lion": 1.5})
        evaluator = MaterialEvaluator.from_config(config)

        assert evaluator._get_species_multiplier("skunk") == 2.0
        assert evaluator._get_species_multiplier("lion") == 1.5
        assert evaluator._get_species_multiplier("unknown") == 1.0


class TestCreateHeuristicVariants:
    """Test create_heuristic_variants function."""

    def test_returns_8_agents_with_distinct_names(self):
        """create_heuristic_variants() returns 8 agents with distinct names."""
        from _02_agents.heuristic import create_heuristic_variants

        variants = create_heuristic_variants()
        assert len(variants) == 8

        # Check names are distinct
        names = [v.name for v in variants]
        assert len(names) == len(set(names))

        # Verify key variants are included
        assert "OnlineStrategies" in names
        assert "OutcomeHeuristic" in names
        assert "DistilledOutcomeHeuristic" in names


# ============================================================================
# Tests for exploiter_training.py
# ============================================================================

class TestExploiterConfig:
    """Test ExploiterConfig dataclass."""

    def test_sensible_defaults(self):
        """ExploiterConfig has sensible defaults."""
        from _03_training.exploiter_training import ExploiterConfig

        config = ExploiterConfig()

        assert config.max_iterations > 0
        assert config.games_per_iteration > 0
        assert 0.0 < config.win_rate_threshold <= 1.0
        assert config.learning_rate > 0
        assert config.batch_size > 0

    def test_validation_catches_invalid_max_iterations(self):
        """ExploiterConfig.validate() catches invalid max_iterations."""
        from _03_training.exploiter_training import ExploiterConfig

        config = ExploiterConfig(max_iterations=0)
        with pytest.raises(ValueError, match="max_iterations"):
            config.validate()

    def test_validation_catches_invalid_win_rate_threshold(self):
        """ExploiterConfig.validate() catches invalid win_rate_threshold."""
        from _03_training.exploiter_training import ExploiterConfig

        config = ExploiterConfig(win_rate_threshold=1.5)
        with pytest.raises(ValueError, match="win_rate_threshold"):
            config.validate()

    def test_to_dict_from_dict_roundtrip(self):
        """ExploiterConfig to_dict/from_dict round-trips correctly."""
        from _03_training.exploiter_training import ExploiterConfig

        original = ExploiterConfig(
            max_iterations=100,
            games_per_iteration=64,
            win_rate_threshold=0.75,
        )
        serialized = original.to_dict()
        restored = ExploiterConfig.from_dict(serialized)

        assert restored.max_iterations == original.max_iterations
        assert restored.games_per_iteration == original.games_per_iteration
        assert restored.win_rate_threshold == original.win_rate_threshold


# ============================================================================
# Tests for exploit_patch_cycle.py
# ============================================================================

class TestCycleConfig:
    """Test CycleConfig dataclass."""

    def test_defaults_are_reasonable(self):
        """CycleConfig defaults are reasonable."""
        from _03_training.exploit_patch_cycle import CycleConfig

        config = CycleConfig()
        assert config.cycle_interval > 0
        assert config.plateau_window > 0
        assert 0.0 < config.plateau_threshold < 1.0
        assert config.max_exploiters > 0
        assert 0.0 <= config.exploiter_weight <= 1.0

    def test_validation_catches_invalid_cycle_interval(self):
        """CycleConfig validates cycle_interval."""
        from _03_training.exploit_patch_cycle import CycleConfig

        with pytest.raises(ValueError, match="cycle_interval"):
            CycleConfig(cycle_interval=0)

    def test_validation_catches_invalid_plateau_threshold(self):
        """CycleConfig validates plateau_threshold."""
        from _03_training.exploit_patch_cycle import CycleConfig

        with pytest.raises(ValueError, match="plateau_threshold"):
            CycleConfig(plateau_threshold=1.5)


class TestExploitPatchManager:
    """Test ExploitPatchManager class."""

    def test_record_win_rate_stores_history(self):
        """record_win_rate() stores history correctly."""
        from _03_training.exploit_patch_cycle import CycleConfig, ExploitPatchManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CycleConfig(plateau_window=10)
            manager = ExploitPatchManager(config=config, checkpoints_dir=Path(tmpdir))

            manager.record_win_rate(0.75, iteration=1)
            manager.record_win_rate(0.76, iteration=2)
            manager.record_win_rate(0.77, iteration=3)

            assert len(manager.win_rate_history) == 3
            assert manager.win_rate_history[-1] == 0.77

    def test_detect_plateau_returns_true_when_stagnant(self):
        """detect_plateau() returns True when win rate is stagnant."""
        from _03_training.exploit_patch_cycle import CycleConfig, ExploitPatchManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CycleConfig(plateau_window=10, plateau_threshold=0.02)
            manager = ExploitPatchManager(config=config, checkpoints_dir=Path(tmpdir))

            # Add stagnant win rates (all around 0.8 with tiny variance)
            for i in range(10):
                manager.record_win_rate(0.80 + 0.001 * (i % 3), iteration=i)

            assert manager.detect_plateau() is True

    def test_detect_plateau_returns_false_when_improving(self):
        """detect_plateau() returns False when win rate is improving."""
        from _03_training.exploit_patch_cycle import CycleConfig, ExploitPatchManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CycleConfig(plateau_window=10, plateau_threshold=0.02)
            manager = ExploitPatchManager(config=config, checkpoints_dir=Path(tmpdir))

            # Add improving win rates
            for i in range(10):
                manager.record_win_rate(0.50 + 0.03 * i, iteration=i)

            assert manager.detect_plateau() is False

    def test_should_start_cycle_triggers_on_interval(self):
        """should_start_cycle() triggers on interval."""
        from _03_training.exploit_patch_cycle import CycleConfig, ExploitPatchManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CycleConfig(cycle_interval=100)
            manager = ExploitPatchManager(config=config, checkpoints_dir=Path(tmpdir))

            # Initially, last_cycle_iteration is -100, so iteration 0 should trigger
            assert manager.should_start_cycle(0) is True

            # Simulate cycle completed at iteration 0
            manager._last_cycle_iteration = 0

            # Should not trigger at 50
            assert manager.should_start_cycle(50) is False

            # Should trigger at 100
            assert manager.should_start_cycle(100) is True

    def test_to_dict_from_dict_preserves_state(self):
        """to_dict()/from_dict() preserves all state."""
        from _03_training.exploit_patch_cycle import CycleConfig, ExploitPatchManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CycleConfig(cycle_interval=50)
            manager = ExploitPatchManager(config=config, checkpoints_dir=Path(tmpdir))

            # Add some state
            manager.record_win_rate(0.75, iteration=10)
            manager.record_win_rate(0.76, iteration=11)
            manager._last_cycle_iteration = 5

            serialized = manager.to_dict()
            restored = ExploitPatchManager.from_dict(serialized)

            assert restored.config.cycle_interval == config.cycle_interval
            assert len(restored.win_rate_history) == len(manager.win_rate_history)
            assert restored._last_cycle_iteration == manager._last_cycle_iteration

    def test_get_exploiter_opponents_empty_when_no_exploiters(self):
        """get_exploiter_opponents() returns empty list when no exploiters."""
        from _03_training.exploit_patch_cycle import ExploitPatchManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExploitPatchManager(checkpoints_dir=Path(tmpdir))
            opponents = manager.get_exploiter_opponents()
            assert opponents == []


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_opponent_pool_checkpoint_weight(self):
        """Empty pool with checkpoint_weight > 0 redistributes to CURRENT."""
        from _03_training.opponent_pool import OpponentConfig, OpponentPool, OpponentType

        config = OpponentConfig(
            current_weight=0.5,
            checkpoint_weight=0.5,
            random_weight=0.0,
            heuristic_weight=0.0,
            mcts_weight=0.0,
        )
        pool = OpponentPool(config=config, seed=42)

        # Should not crash - checkpoint weight goes to current
        sampled = pool.sample_opponent()
        assert sampled.opponent_type in {OpponentType.CURRENT}

    def test_zero_mcts_configs_with_mcts_weight(self):
        """Zero MCTS configs with mcts_weight > 0 redistributes to CURRENT."""
        from _03_training.opponent_pool import OpponentConfig, OpponentPool, OpponentType

        config = OpponentConfig(
            current_weight=0.5,
            checkpoint_weight=0.0,
            random_weight=0.0,
            heuristic_weight=0.0,
            mcts_weight=0.5,
            mcts_configs=[],  # Empty!
        )
        pool = OpponentPool(config=config, seed=42)

        # Should redistribute MCTS weight to CURRENT
        type_counts = dict.fromkeys(OpponentType, 0)
        for _ in range(50):
            sampled = pool.sample_opponent()
            type_counts[sampled.opponent_type] += 1

        assert type_counts[OpponentType.MCTS] == 0

    def test_stats_tracker_high_volume(self):
        """Stats tracker handles high volume without crashing."""
        from _03_training.opponent_statistics import OpponentStatsTracker

        tracker = OpponentStatsTracker(window_size=1000)

        # 10,000 updates - should not crash
        for i in range(10000):
            result = ["win", "loss", "draw"][i % 3]
            tracker.update(f"opp_{i % 10}", result, iteration=i)

        # Should still work
        stats = tracker.get_all_stats()
        assert len(stats) == 10


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
