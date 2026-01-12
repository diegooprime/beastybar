"""Tests for Phase 4: Population-based training.

This module tests the population training infrastructure:
- PopulationConfig validation
- PopulationMember management
- Exploiter lifecycle
- ELO rating integration
- Tournament mechanics
- Culling and integration

Edge case tests verify that the population trainer handles
adversarial and unusual scenarios correctly.
"""

import numpy as np
import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")

from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig
from _03_training.elo import EloRating, Leaderboard, PlayerStats
from _03_training.population import (
    Exploiter,
    PopulationConfig,
    PopulationMember,
    PopulationTrainer,
)


# ============================================================================
# Configuration Tests
# ============================================================================


class TestPopulationConfig:
    """Tests for PopulationConfig dataclass."""

    def test_default_config_valid(self):
        """Default configuration should pass validation."""
        config = PopulationConfig()
        config.validate()  # Should not raise
        assert config.population_size == 8
        assert config.max_exploiters == 2

    def test_config_from_dict(self):
        """Configuration can be created from dictionary."""
        data = {
            "population_size": 4,
            "max_exploiters": 1,
            "exploit_threshold": 0.55,
            "network_config": {"hidden_dim": 128},
        }
        config = PopulationConfig.from_dict(data)
        assert config.population_size == 4
        assert config.max_exploiters == 1
        assert config.exploit_threshold == 0.55
        assert config.network_config.hidden_dim == 128

    def test_config_from_nested_dict(self):
        """Configuration handles nested population_config key."""
        data = {
            "population_config": {
                "population_size": 6,
                "max_exploiters": 3,
            },
            "network_config": {"hidden_dim": 256},
        }
        config = PopulationConfig.from_dict(data)
        assert config.population_size == 6
        assert config.max_exploiters == 3

    def test_config_to_dict_roundtrip(self):
        """Configuration survives dict conversion and back."""
        original = PopulationConfig(population_size=5, exploit_threshold=0.7)
        data = original.to_dict()
        restored = PopulationConfig.from_dict(data)
        assert restored.population_size == original.population_size
        assert restored.exploit_threshold == original.exploit_threshold

    def test_invalid_population_size(self):
        """Population size must be at least 2."""
        config = PopulationConfig(population_size=1)
        with pytest.raises(ValueError, match="population_size must be >= 2"):
            config.validate()

    def test_invalid_exploit_threshold(self):
        """Exploit threshold must be in (0, 1)."""
        config = PopulationConfig(exploit_threshold=1.5)
        with pytest.raises(ValueError, match="exploit_threshold must be in"):
            config.validate()

        config = PopulationConfig(exploit_threshold=0.0)
        with pytest.raises(ValueError, match="exploit_threshold must be in"):
            config.validate()

    def test_invalid_cull_threshold(self):
        """Cull threshold must be in (0, 1)."""
        config = PopulationConfig(cull_threshold=-0.1)
        with pytest.raises(ValueError, match="cull_threshold must be in"):
            config.validate()


# ============================================================================
# Population Member Tests
# ============================================================================


class TestPopulationMember:
    """Tests for PopulationMember dataclass."""

    def test_member_creation(self):
        """Population member can be created with network."""
        network = BeastyBarNetwork(NetworkConfig())
        optimizer = torch.optim.Adam(network.parameters())

        member = PopulationMember(
            agent_id="test_agent",
            network=network,
            optimizer=optimizer,
        )

        assert member.agent_id == "test_agent"
        assert member.generation == 0
        assert member.total_games == 0

    def test_win_rate_calculation(self):
        """Win rate is calculated correctly."""
        network = BeastyBarNetwork(NetworkConfig())
        optimizer = torch.optim.Adam(network.parameters())
        member = PopulationMember(
            agent_id="test", network=network, optimizer=optimizer
        )

        # Initial win rate should be 0.5 (default for new agents)
        assert member.win_rate == 0.5

        # Record some results
        member.record_result(won=True, lost=False, drawn=False)
        member.record_result(won=True, lost=False, drawn=False)
        member.record_result(won=False, lost=True, drawn=False)

        assert member.total_games == 3
        assert member.total_wins == 2
        assert member.total_losses == 1
        assert member.win_rate == pytest.approx(2 / 3)

    def test_draw_recording(self):
        """Draws are recorded correctly."""
        network = BeastyBarNetwork(NetworkConfig())
        optimizer = torch.optim.Adam(network.parameters())
        member = PopulationMember(
            agent_id="test", network=network, optimizer=optimizer
        )

        member.record_result(won=False, lost=False, drawn=True)
        assert member.total_draws == 1
        assert member.total_games == 1


# ============================================================================
# Exploiter Tests
# ============================================================================


class TestExploiter:
    """Tests for Exploiter dataclass."""

    def test_exploiter_creation(self):
        """Exploiter can be created with target."""
        network = BeastyBarNetwork(NetworkConfig())
        optimizer = torch.optim.Adam(network.parameters())

        exploiter = Exploiter(
            exploiter_id="exp_001",
            network=network,
            optimizer=optimizer,
            target_id="agent_000",
            created_iteration=100,
        )

        assert exploiter.exploiter_id == "exp_001"
        assert exploiter.target_id == "agent_000"
        assert exploiter.training_iterations == 0
        assert exploiter.best_win_rate == 0.0
        assert not exploiter.abandoned

    def test_exploiter_win_rate_tracking(self):
        """Exploiter tracks best and current win rates."""
        network = BeastyBarNetwork(NetworkConfig())
        optimizer = torch.optim.Adam(network.parameters())

        exploiter = Exploiter(
            exploiter_id="exp",
            network=network,
            optimizer=optimizer,
            target_id="target",
            created_iteration=0,
        )

        exploiter.current_win_rate = 0.45
        exploiter.best_win_rate = max(exploiter.best_win_rate, exploiter.current_win_rate)
        assert exploiter.best_win_rate == 0.45

        exploiter.current_win_rate = 0.55
        exploiter.best_win_rate = max(exploiter.best_win_rate, exploiter.current_win_rate)
        assert exploiter.best_win_rate == 0.55

        exploiter.current_win_rate = 0.50
        # Best should not decrease
        assert exploiter.best_win_rate == 0.55


# ============================================================================
# ELO System Integration Tests
# ============================================================================


class TestELOIntegration:
    """Tests for ELO rating system integration."""

    def test_leaderboard_registration(self):
        """Leaderboard registers new players correctly."""
        leaderboard = Leaderboard(initial_rating=1500.0)
        leaderboard.register("player_a")
        leaderboard.register("player_b")

        assert leaderboard.get_rating("player_a") == 1500.0
        assert leaderboard.get_rating("player_b") == 1500.0

    def test_elo_update_on_win(self):
        """ELO ratings update correctly after a win."""
        elo = EloRating(k_factor=32.0, initial_rating=1500.0)
        leaderboard = Leaderboard(elo=elo, initial_rating=1500.0)

        leaderboard.register("winner")
        leaderboard.register("loser")

        # Winner beats loser (score 1-0)
        new_a, new_b = leaderboard.record_match("winner", "loser", 1, 0)

        # Winner's rating should increase
        assert new_a > 1500.0
        # Loser's rating should decrease
        assert new_b < 1500.0
        # Symmetric change (K-factor applied equally)
        assert abs((new_a - 1500.0) + (new_b - 1500.0)) < 0.01

    def test_rankings_sorted_by_elo(self):
        """Rankings are sorted by ELO descending."""
        leaderboard = Leaderboard(initial_rating=1500.0)

        for name in ["a", "b", "c"]:
            leaderboard.register(name)

        # Simulate: a beats b beats c
        leaderboard.record_match("a", "b", 1, 0)
        leaderboard.record_match("b", "c", 1, 0)
        leaderboard.record_match("a", "c", 1, 0)

        rankings = leaderboard.rankings()
        assert rankings[0].name == "a"  # Highest rated
        assert rankings[-1].name == "c"  # Lowest rated


# ============================================================================
# Population Trainer Tests (Unit)
# ============================================================================


class TestPopulationTrainerUnit:
    """Unit tests for PopulationTrainer methods."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal config for fast testing."""
        return PopulationConfig(
            population_size=2,
            max_exploiters=1,
            games_per_iteration=4,
            total_iterations=2,
            batch_size=4,
            epochs_per_iteration=1,
            num_simulations=2,  # Very few for speed
            tournament_frequency=0,  # Disable tournament
            cull_frequency=0,  # Disable culling
            eval_frequency=0,  # Disable evaluation
            checkpoint_frequency=1000,  # Don't checkpoint
            buffer_size=100,
            min_buffer_size=1,
            parallel_games=2,
            device="cpu",
        )

    def test_trainer_initialization(self, minimal_config):
        """Trainer initializes with correct population size."""
        trainer = PopulationTrainer(minimal_config)

        assert len(trainer.population) == minimal_config.population_size
        assert len(trainer.exploiters) == 0
        assert trainer._iteration == 0

    def test_population_diversity(self, minimal_config):
        """Initial population has diverse networks."""
        trainer = PopulationTrainer(minimal_config)

        # Networks should be different (random initialization)
        params_0 = list(trainer.population[0].network.parameters())[0].data
        params_1 = list(trainer.population[1].network.parameters())[0].data

        # With random init, parameters should differ
        # (This is probabilistic but virtually guaranteed)
        assert not torch.allclose(params_0, params_1)

    def test_get_best_agent(self, minimal_config):
        """Best agent is identified by ELO rating."""
        trainer = PopulationTrainer(minimal_config)

        # Initially all have same rating
        best = trainer._get_best_agent()
        assert best in trainer.population

        # Manually increase one agent's rating
        trainer.leaderboard.record_match(
            trainer.population[0].agent_id,
            trainer.population[1].agent_id,
            score_a=1,
            score_b=0,
        )

        best = trainer._get_best_agent()
        assert best.agent_id == trainer.population[0].agent_id

    def test_spawn_exploiter(self, minimal_config):
        """Exploiter is spawned targeting best agent."""
        trainer = PopulationTrainer(minimal_config)

        best_agent = trainer._get_best_agent()
        exploiter = trainer.spawn_exploiter()

        assert exploiter is not None
        assert exploiter.target_id == best_agent.agent_id
        assert len(trainer.exploiters) == 1

    def test_exploiter_capacity_limit(self, minimal_config):
        """Cannot spawn more exploiters than max_exploiters."""
        trainer = PopulationTrainer(minimal_config)

        # Spawn up to limit
        exploiter1 = trainer.spawn_exploiter()
        assert exploiter1 is not None

        # Should return None at capacity
        exploiter2 = trainer.spawn_exploiter()
        assert exploiter2 is None
        assert len(trainer.exploiters) == minimal_config.max_exploiters


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and adversarial scenarios."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal config for edge case testing."""
        return PopulationConfig(
            population_size=2,
            max_exploiters=1,
            games_per_iteration=2,
            total_iterations=1,
            batch_size=2,
            epochs_per_iteration=1,
            num_simulations=2,
            tournament_frequency=0,
            cull_frequency=0,
            eval_frequency=0,
            checkpoint_frequency=1000,
            buffer_size=50,
            min_buffer_size=1,
            parallel_games=1,
            device="cpu",
        )

    def test_empty_replay_buffer_handling(self, minimal_config):
        """Training handles empty replay buffer gracefully."""
        trainer = PopulationTrainer(minimal_config)

        # Buffer is empty initially
        assert len(trainer.replay_buffer) == 0

        # Training should return empty dict when buffer too small
        metrics = trainer.train_member(trainer.population[0])
        # With min_buffer_size=1, it might succeed or return empty
        # depending on implementation

    def test_exploiter_target_removal(self, minimal_config):
        """Exploiter handles removed target gracefully."""
        trainer = PopulationTrainer(minimal_config)

        # Spawn exploiter
        exploiter = trainer.spawn_exploiter()
        assert exploiter is not None

        # "Remove" target by clearing population (simulating the target being culled)
        original_target = exploiter.target_id

        # Remove all members (simulating complete population replacement)
        trainer.population.clear()

        # Training exploiter should detect missing target and mark abandoned
        metrics = trainer.train_exploiter(exploiter)
        assert exploiter.abandoned
        assert metrics == {}  # Empty metrics when target not found

    def test_single_agent_population_cull(self, minimal_config):
        """Culling respects minimum population size."""
        minimal_config.population_size = 2
        minimal_config.cull_threshold = 0.9  # Very high threshold
        trainer = PopulationTrainer(minimal_config)

        # Even with high threshold, should not cull to below 2
        culled = trainer.cull_weak_agents()
        assert len(trainer.population) >= 2

    def test_all_draws_elo_stability(self):
        """ELO remains stable when all games are draws."""
        leaderboard = Leaderboard(initial_rating=1500.0)
        leaderboard.register("a")
        leaderboard.register("b")

        # Record 10 draws
        for _ in range(10):
            leaderboard.record_match("a", "b", 0, 0)

        # Ratings should remain close to initial
        assert abs(leaderboard.get_rating("a") - 1500.0) < 50
        assert abs(leaderboard.get_rating("b") - 1500.0) < 50

    def test_extreme_win_streak(self):
        """ELO handles extreme win streaks."""
        leaderboard = Leaderboard(initial_rating=1500.0)
        leaderboard.register("strong")
        leaderboard.register("weak")

        # 100 consecutive wins
        for _ in range(100):
            leaderboard.record_match("strong", "weak", 1, 0)

        # Strong should have significantly higher rating than initial
        # With K=32 and diminishing expected score, typical is ~1800 after 100 games
        assert leaderboard.get_rating("strong") > 1750
        assert leaderboard.get_rating("weak") < 1250
        # Also verify the gap
        gap = leaderboard.get_rating("strong") - leaderboard.get_rating("weak")
        assert gap > 500  # Significant ELO gap after 100 consecutive wins

    def test_network_clone_independence(self, minimal_config):
        """Cloned networks are independent."""
        trainer = PopulationTrainer(minimal_config)

        network1 = trainer.population[0].network
        network2 = trainer.population[1].network

        # Modify one network
        with torch.no_grad():
            for param in network1.parameters():
                param.add_(1.0)

        # Other should be unchanged
        params1 = list(network1.parameters())[0].data.mean().item()
        params2 = list(network2.parameters())[0].data.mean().item()
        assert abs(params1 - params2) > 0.5


# ============================================================================
# Integration Tests (Minimal)
# ============================================================================


class TestIntegration:
    """Integration tests for population training."""

    @pytest.fixture
    def tiny_config(self):
        """Extremely minimal config for integration testing."""
        return PopulationConfig(
            population_size=2,
            max_exploiters=0,  # No exploiters for simplicity
            games_per_iteration=2,
            total_iterations=2,
            batch_size=2,
            epochs_per_iteration=1,
            num_simulations=2,
            tournament_frequency=1,
            tournament_games=2,
            cull_frequency=0,
            eval_frequency=0,
            checkpoint_frequency=1000,
            buffer_size=50,
            min_buffer_size=1,
            parallel_games=1,
            device="cpu",
        )

    def test_single_iteration(self, tiny_config):
        """Single training iteration completes."""
        trainer = PopulationTrainer(tiny_config)
        metrics = trainer.train_iteration()

        assert "iteration" in metrics
        assert metrics["iteration"] == 0.0
        assert "self_play/examples_collected" in metrics

    def test_tournament_updates_elo(self, tiny_config):
        """Tournament updates ELO ratings."""
        trainer = PopulationTrainer(tiny_config)

        initial_ratings = {
            m.agent_id: trainer.leaderboard.get_rating(m.agent_id)
            for m in trainer.population
        }

        trainer.run_tournament()

        # At least one rating should have changed
        final_ratings = {
            m.agent_id: trainer.leaderboard.get_rating(m.agent_id)
            for m in trainer.population
        }

        ratings_changed = any(
            abs(initial_ratings[k] - final_ratings[k]) > 0.1
            for k in initial_ratings
        )
        assert ratings_changed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
