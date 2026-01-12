"""Tests for curriculum learning module.

Tests cover:
- Species curriculum: restricted species games, level advancement
- Opponent curriculum: mix sampling, historical pool management
- Persistence: save/load of curriculum state
- Integration: scheduler with actual game states
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from _01_simulator import rules
from _01_simulator.state import State
from _03_training.curriculum import (
    CURRICULUM_LEVELS,
    MIN_SPECIES_FOR_GAME,
    OPPONENT_STAGES,
    CurriculumConfig,
    CurriculumScheduler,
    HistoricalPool,
    OpponentMix,
    create_curriculum_game,
    get_curriculum_level_for_species_count,
    get_opponent_type_name,
    load_curriculum_state,
    load_historical_pool,
    save_curriculum_state,
    save_historical_pool,
    validate_species_whitelist,
)

# ============================================================================
# Species Curriculum Tests (7.1)
# ============================================================================


class TestCurriculumLevels:
    """Test curriculum level definitions."""

    def test_curriculum_levels_are_ordered(self) -> None:
        """Levels should have increasing species count."""
        prev_count = 0
        for level in sorted(CURRICULUM_LEVELS.keys()):
            current_count = len(CURRICULUM_LEVELS[level])
            assert current_count >= prev_count, f"Level {level} has fewer species than previous"
            prev_count = current_count

    def test_curriculum_levels_contain_valid_species(self) -> None:
        """All species in levels should be valid."""
        for level, species_list in CURRICULUM_LEVELS.items():
            for species in species_list:
                assert species in rules.SPECIES, f"Unknown species '{species}' in level {level}"
                assert species != "unknown", "Cannot include 'unknown' species"

    def test_level_4_is_full_game(self) -> None:
        """Level 4 should contain all base game species."""
        level_4_species = set(CURRICULUM_LEVELS[4])
        base_species = set(rules.BASE_DECK)
        assert level_4_species == base_species, "Level 4 should contain all base species"

    def test_each_level_has_minimum_species(self) -> None:
        """Each level should have at least MIN_SPECIES_FOR_GAME species."""
        for level, species_list in CURRICULUM_LEVELS.items():
            assert len(species_list) >= MIN_SPECIES_FOR_GAME, (
                f"Level {level} has only {len(species_list)} species, "
                f"minimum is {MIN_SPECIES_FOR_GAME}"
            )


class TestValidateSpeciesWhitelist:
    """Test species whitelist validation."""

    def test_valid_whitelist_accepted(self) -> None:
        """Valid whitelist should not raise."""
        validate_species_whitelist(["lion", "hippo", "crocodile", "snake"])

    def test_whitelist_too_small_rejected(self) -> None:
        """Whitelist with too few species should raise."""
        with pytest.raises(ValueError, match="at least"):
            validate_species_whitelist(["lion", "hippo"])

    def test_unknown_species_rejected(self) -> None:
        """Whitelist with unknown species should raise."""
        with pytest.raises(ValueError, match="Unknown species"):
            validate_species_whitelist(["lion", "hippo", "crocodile", "dragon"])

    def test_unknown_species_in_whitelist_rejected(self) -> None:
        """Cannot include 'unknown' (masking species) in whitelist."""
        with pytest.raises(ValueError, match="unknown"):
            validate_species_whitelist(["lion", "hippo", "crocodile", "unknown"])


class TestCreateCurriculumGame:
    """Test curriculum game initialization."""

    def test_creates_valid_game_state(self) -> None:
        """Created state should be a valid game state."""
        species = ["lion", "hippo", "crocodile", "snake"]
        state = create_curriculum_game(seed=42, species_whitelist=species)

        assert isinstance(state, State)
        assert state.turn == 0
        assert state.active_player in (0, 1)

    def test_respects_species_whitelist(self) -> None:
        """All cards should be from the whitelist."""
        species = ["lion", "hippo", "crocodile", "snake"]
        state = create_curriculum_game(seed=42, species_whitelist=species)

        all_species = set()

        # Check both players' hands and decks
        for player_state in state.players:
            for card in player_state.hand:
                all_species.add(card.species)
            for card in player_state.deck:
                all_species.add(card.species)

        assert all_species == set(species), f"Found species {all_species}, expected {species}"

    def test_each_player_has_all_whitelisted_species(self) -> None:
        """Each player should have one of each whitelisted species."""
        species = ["lion", "hippo", "crocodile", "snake"]
        state = create_curriculum_game(seed=42, species_whitelist=species)

        for i, player_state in enumerate(state.players):
            player_species = [c.species for c in player_state.hand]
            player_species.extend(c.species for c in player_state.deck)
            assert sorted(player_species) == sorted(species), (
                f"Player {i} species {player_species} != expected {species}"
            )

    def test_respects_starting_player(self) -> None:
        """Starting player should be set correctly."""
        species = ["lion", "hippo", "crocodile", "snake"]

        state_0 = create_curriculum_game(seed=42, species_whitelist=species, starting_player=0)
        assert state_0.active_player == 0

        state_1 = create_curriculum_game(seed=42, species_whitelist=species, starting_player=1)
        assert state_1.active_player == 1

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce identical states."""
        species = ["lion", "hippo", "crocodile", "snake"]

        state1 = create_curriculum_game(seed=123, species_whitelist=species)
        state2 = create_curriculum_game(seed=123, species_whitelist=species)

        # Compare hands (should be identical order)
        assert state1.players[0].hand == state2.players[0].hand
        assert state1.players[1].hand == state2.players[1].hand
        assert state1.players[0].deck == state2.players[0].deck
        assert state1.players[1].deck == state2.players[1].deck

    def test_different_seeds_produce_different_states(self) -> None:
        """Different seeds should (usually) produce different states."""
        species = ["lion", "hippo", "crocodile", "snake"]

        state1 = create_curriculum_game(seed=42, species_whitelist=species)
        state2 = create_curriculum_game(seed=999, species_whitelist=species)

        # At least one player's hand should differ
        hands_differ = (
            state1.players[0].hand != state2.players[0].hand
            or state1.players[1].hand != state2.players[1].hand
        )
        assert hands_differ, "Different seeds should produce different shuffles"

    def test_invalid_starting_player_rejected(self) -> None:
        """Invalid starting player should raise."""
        species = ["lion", "hippo", "crocodile", "snake"]

        with pytest.raises(ValueError, match="Invalid starting player"):
            create_curriculum_game(seed=42, species_whitelist=species, starting_player=2)

    def test_full_game_species(self) -> None:
        """Level 4 species should create a full game."""
        species = CURRICULUM_LEVELS[4]
        state = create_curriculum_game(seed=42, species_whitelist=species)

        # Should have full deck size per player
        for player_state in state.players:
            total_cards = len(player_state.hand) + len(player_state.deck)
            assert total_cards == rules.DECK_SIZE


class TestGetCurriculumLevelForSpeciesCount:
    """Test level determination from species count."""

    def test_returns_appropriate_level(self) -> None:
        """Should return level with at least the requested species count."""
        # Level 1 has 4 species
        assert get_curriculum_level_for_species_count(4) == 1

        # Level 2 has 7 species
        assert get_curriculum_level_for_species_count(6) == 2
        assert get_curriculum_level_for_species_count(7) == 2

        # Level 4 has 12 species (full game)
        assert get_curriculum_level_for_species_count(12) == 4

    def test_large_count_returns_max_level(self) -> None:
        """Requesting more species than available returns max level."""
        assert get_curriculum_level_for_species_count(100) == max(CURRICULUM_LEVELS.keys())


# ============================================================================
# Curriculum Scheduler Tests
# ============================================================================


class TestCurriculumConfig:
    """Test curriculum configuration."""

    def test_default_config_valid(self) -> None:
        """Default config should be valid."""
        config = CurriculumConfig()
        assert config.initial_level == 1
        assert 0 < config.advance_threshold <= 1

    def test_invalid_initial_level_rejected(self) -> None:
        """Invalid initial level should raise."""
        with pytest.raises(ValueError):
            CurriculumConfig(initial_level=0)
        with pytest.raises(ValueError):
            CurriculumConfig(initial_level=10)

    def test_invalid_threshold_rejected(self) -> None:
        """Invalid threshold should raise."""
        with pytest.raises(ValueError):
            CurriculumConfig(advance_threshold=0)
        with pytest.raises(ValueError):
            CurriculumConfig(advance_threshold=1.5)

    def test_to_dict_and_back(self) -> None:
        """Config should survive serialization round-trip."""
        config = CurriculumConfig(
            initial_level=2,
            advance_threshold=0.75,
            min_games_per_level=1000,
        )
        data = config.to_dict()
        restored = CurriculumConfig.from_dict(data)

        assert restored.initial_level == config.initial_level
        assert restored.advance_threshold == config.advance_threshold
        assert restored.min_games_per_level == config.min_games_per_level


class TestCurriculumScheduler:
    """Test curriculum scheduler."""

    def test_initialization(self) -> None:
        """Scheduler should initialize with correct defaults."""
        scheduler = CurriculumScheduler()
        assert scheduler.current_level == 1
        assert scheduler.games_at_level == 0
        assert scheduler.win_rate == 0.0

    def test_initialization_with_config(self) -> None:
        """Scheduler should respect config."""
        config = CurriculumConfig(initial_level=2)
        scheduler = CurriculumScheduler(config)
        assert scheduler.current_level == 2

    def test_get_species_whitelist(self) -> None:
        """Should return correct species for current level."""
        scheduler = CurriculumScheduler()
        whitelist = scheduler.get_species_whitelist()
        assert whitelist == CURRICULUM_LEVELS[1]

    def test_record_game_updates_counters(self) -> None:
        """Recording games should update counters."""
        scheduler = CurriculumScheduler()

        scheduler.record_game(win=True)
        assert scheduler.games_at_level == 1
        assert scheduler.wins_at_level == 1
        assert scheduler.win_rate == 1.0

        scheduler.record_game(win=False)
        assert scheduler.games_at_level == 2
        assert scheduler.wins_at_level == 1
        assert scheduler.win_rate == 0.5

    def test_should_advance_not_enough_games(self) -> None:
        """Should not advance without minimum games."""
        config = CurriculumConfig(min_games_per_level=100, advance_threshold=0.7)
        scheduler = CurriculumScheduler(config)

        # Record some wins but not enough games
        for _ in range(50):
            scheduler.record_game(win=True)

        # Even with 100% win rate, not enough games
        assert not scheduler.should_advance(1.0, 50)

    def test_should_advance_threshold_met(self) -> None:
        """Should advance when threshold is met."""
        config = CurriculumConfig(min_games_per_level=10, advance_threshold=0.7)
        scheduler = CurriculumScheduler(config)

        # Win rate above threshold
        assert scheduler.should_advance(0.75, 100)

        # Win rate below threshold
        assert not scheduler.should_advance(0.5, 100)

    def test_advance_increments_level(self) -> None:
        """Advance should increment level and reset counters."""
        scheduler = CurriculumScheduler()

        # Record some games
        for _ in range(10):
            scheduler.record_game(win=True)

        old_level = scheduler.current_level
        result = scheduler.advance()

        assert result is True
        assert scheduler.current_level == old_level + 1
        assert scheduler.games_at_level == 0
        assert scheduler.wins_at_level == 0

    def test_advance_records_history(self) -> None:
        """Advance should record level history."""
        scheduler = CurriculumScheduler()

        for _ in range(10):
            scheduler.record_game(win=True)

        scheduler.advance()

        assert len(scheduler.level_history) == 1
        level, games, win_rate = scheduler.level_history[0]
        assert level == 1
        assert games == 10
        assert win_rate == 1.0

    def test_cannot_advance_past_max_level(self) -> None:
        """Should not advance past max level."""
        config = CurriculumConfig(initial_level=len(CURRICULUM_LEVELS))
        scheduler = CurriculumScheduler(config)

        result = scheduler.advance()
        assert result is False
        assert scheduler.current_level == scheduler.max_level

    def test_auto_advance_on_threshold(self) -> None:
        """Recording games should auto-advance when criteria met."""
        config = CurriculumConfig(
            min_games_per_level=10,
            max_games_per_level=100,
            advance_threshold=0.7,
        )
        scheduler = CurriculumScheduler(config)

        # Record 10 wins (100% win rate)
        for _ in range(10):
            scheduler.record_game(win=True)

        # Should have auto-advanced
        assert scheduler.current_level == 2

    def test_force_advance_at_max_games(self) -> None:
        """Should force advance after max games even with low win rate."""
        config = CurriculumConfig(
            min_games_per_level=10,
            max_games_per_level=20,
            advance_threshold=0.9,  # High threshold
        )
        scheduler = CurriculumScheduler(config)

        # Record 20 losses (0% win rate but at max games)
        for _ in range(20):
            scheduler.record_game(win=False)

        # Should have force-advanced despite low win rate
        assert scheduler.current_level == 2

    def test_set_level_manual(self) -> None:
        """Should allow manual level setting."""
        scheduler = CurriculumScheduler()
        scheduler.set_level(3)
        assert scheduler.current_level == 3
        assert scheduler.games_at_level == 0

    def test_set_level_invalid_rejected(self) -> None:
        """Invalid manual level should raise."""
        scheduler = CurriculumScheduler()
        with pytest.raises(ValueError):
            scheduler.set_level(0)
        with pytest.raises(ValueError):
            scheduler.set_level(10)

    def test_summary_format(self) -> None:
        """Summary should return readable string."""
        scheduler = CurriculumScheduler()
        summary = scheduler.summary()
        assert "Curriculum Status" in summary
        assert "Species Level" in summary


class TestCurriculumStatePersistence:
    """Test curriculum state save/load."""

    def test_save_and_load_state(self) -> None:
        """State should survive save/load cycle."""
        config = CurriculumConfig(advance_threshold=0.8)
        scheduler = CurriculumScheduler(config)

        # Record some progress
        for _ in range(50):
            scheduler.record_game(win=True)
        scheduler.advance()
        for _ in range(30):
            scheduler.record_game(win=False)

        # Save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name

        try:
            save_curriculum_state(scheduler, save_path)

            # Load
            loaded = load_curriculum_state(save_path)

            # Verify state matches
            assert loaded.current_level == scheduler.current_level
            assert loaded.games_at_level == scheduler.games_at_level
            assert loaded.wins_at_level == scheduler.wins_at_level
            assert loaded.opponent_stage == scheduler.opponent_stage
            assert loaded.level_history == scheduler.level_history
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_load_nonexistent_raises(self) -> None:
        """Loading nonexistent file should raise."""
        with pytest.raises(FileNotFoundError):
            load_curriculum_state("/nonexistent/path.json")


# ============================================================================
# Opponent Curriculum Tests (7.2)
# ============================================================================


class TestOpponentMix:
    """Test opponent mix configuration."""

    def test_valid_mix_accepted(self) -> None:
        """Valid mix should not raise."""
        mix = OpponentMix(self_play_ratio=0.5, random_ratio=0.5)
        assert mix.self_play_ratio == 0.5
        assert mix.random_ratio == 0.5

    def test_ratios_must_sum_to_one(self) -> None:
        """Ratios not summing to 1 should raise."""
        with pytest.raises(ValueError, match="sum to 1"):
            OpponentMix(self_play_ratio=0.5, random_ratio=0.3)

    def test_negative_ratio_rejected(self) -> None:
        """Negative ratios should raise."""
        with pytest.raises(ValueError):
            OpponentMix(self_play_ratio=-0.1, random_ratio=1.1)

    def test_predefined_stages_valid(self) -> None:
        """All predefined stages should be valid."""
        for stage, mix in OPPONENT_STAGES.items():
            total = (
                mix.self_play_ratio
                + mix.random_ratio
                + mix.heuristic_ratio
                + mix.mcts_ratio
                + mix.historical_ratio
            )
            assert abs(total - 1.0) < 0.01, f"Stage {stage} ratios sum to {total}"

    def test_to_dict_and_back(self) -> None:
        """Mix should survive serialization."""
        mix = OpponentMix(self_play_ratio=0.6, heuristic_ratio=0.4)
        data = mix.to_dict()
        restored = OpponentMix.from_dict(data)

        assert restored.self_play_ratio == mix.self_play_ratio
        assert restored.heuristic_ratio == mix.heuristic_ratio


class TestGetOpponentTypeName:
    """Test opponent mix naming."""

    def test_pure_self_play(self) -> None:
        """Pure self-play should be named correctly."""
        mix = OpponentMix(self_play_ratio=1.0)
        name = get_opponent_type_name(mix)
        assert "self-play" in name

    def test_mixed_opponents(self) -> None:
        """Mixed opponents should list all types."""
        mix = OpponentMix(self_play_ratio=0.5, heuristic_ratio=0.3, mcts_ratio=0.2)
        name = get_opponent_type_name(mix)
        assert "self-play" in name
        assert "heuristic" in name
        assert "mcts" in name


# ============================================================================
# Historical Pool Tests
# ============================================================================


class TestHistoricalPool:
    """Test historical checkpoint pool."""

    def test_initialization(self) -> None:
        """Pool should initialize empty."""
        pool = HistoricalPool(max_size=5)
        assert len(pool) == 0
        assert pool.max_size == 5

    def test_add_checkpoint(self) -> None:
        """Adding checkpoint should increase size."""
        pool = HistoricalPool(max_size=5)
        pool.add_checkpoint("/path/to/checkpoint.pt", iteration=100, win_rate=0.6)
        assert len(pool) == 1

    def test_max_size_enforced(self) -> None:
        """Pool should not exceed max size."""
        pool = HistoricalPool(max_size=3)

        for i in range(5):
            pool.add_checkpoint(f"/path/{i}.pt", iteration=i, win_rate=0.5 + i * 0.05)

        assert len(pool) == 3

    def test_eviction_removes_lowest_win_rate(self) -> None:
        """Eviction should remove checkpoint with lowest win rate."""
        pool = HistoricalPool(max_size=2)

        pool.add_checkpoint("/path/low.pt", iteration=1, win_rate=0.3)
        pool.add_checkpoint("/path/mid.pt", iteration=2, win_rate=0.5)
        pool.add_checkpoint("/path/high.pt", iteration=3, win_rate=0.7)

        paths = pool.get_checkpoints()
        assert "/path/low.pt" not in paths
        assert "/path/mid.pt" in paths
        assert "/path/high.pt" in paths

    def test_sample_checkpoint_uniform(self) -> None:
        """Uniform sampling should return checkpoint."""
        pool = HistoricalPool(max_size=5)
        pool.add_checkpoint("/path/1.pt", iteration=1, win_rate=0.5)
        pool.add_checkpoint("/path/2.pt", iteration=2, win_rate=0.6)

        # Should return one of the checkpoints
        path = pool.sample_checkpoint(strategy="uniform")
        assert path in ["/path/1.pt", "/path/2.pt"]

    def test_sample_checkpoint_empty_pool(self) -> None:
        """Sampling from empty pool should return None."""
        pool = HistoricalPool(max_size=5)
        assert pool.sample_checkpoint() is None

    def test_sample_checkpoint_weighted(self) -> None:
        """Weighted sampling should favor higher win rates."""
        pool = HistoricalPool(max_size=5)
        pool.add_checkpoint("/path/low.pt", iteration=1, win_rate=0.1)
        pool.add_checkpoint("/path/high.pt", iteration=2, win_rate=0.9)

        # Sample many times and check distribution
        counts = {"/path/low.pt": 0, "/path/high.pt": 0}
        for _ in range(1000):
            path = pool.sample_checkpoint(strategy="weighted")
            if path:
                counts[path] += 1

        # High win rate should be sampled more often
        assert counts["/path/high.pt"] > counts["/path/low.pt"]

    def test_get_checkpoints_sorted_by_iteration(self) -> None:
        """get_checkpoints should return sorted by iteration."""
        pool = HistoricalPool(max_size=5)
        pool.add_checkpoint("/path/3.pt", iteration=300, win_rate=0.5)
        pool.add_checkpoint("/path/1.pt", iteration=100, win_rate=0.5)
        pool.add_checkpoint("/path/2.pt", iteration=200, win_rate=0.5)

        paths = pool.get_checkpoints()
        assert paths == ["/path/1.pt", "/path/2.pt", "/path/3.pt"]

    def test_clear_pool(self) -> None:
        """Clear should remove all checkpoints."""
        pool = HistoricalPool(max_size=5)
        pool.add_checkpoint("/path/1.pt", iteration=1, win_rate=0.5)
        pool.add_checkpoint("/path/2.pt", iteration=2, win_rate=0.5)

        pool.clear()
        assert len(pool) == 0

    def test_duplicate_path_updates(self) -> None:
        """Adding same path again should update, not duplicate."""
        pool = HistoricalPool(max_size=5)
        pool.add_checkpoint("/path/1.pt", iteration=1, win_rate=0.5)
        pool.add_checkpoint("/path/1.pt", iteration=1, win_rate=0.7)  # Update

        assert len(pool) == 1
        info = pool.get_checkpoint_info()[0]
        assert info.win_rate == 0.7

    def test_state_dict_round_trip(self) -> None:
        """Pool should survive serialization."""
        pool = HistoricalPool(max_size=5)
        pool.add_checkpoint("/path/1.pt", iteration=1, win_rate=0.5)
        pool.add_checkpoint("/path/2.pt", iteration=2, win_rate=0.6)

        state = pool.get_state_dict()
        restored = HistoricalPool.from_state_dict(state)

        assert len(restored) == len(pool)
        assert restored.max_size == pool.max_size
        assert restored.get_checkpoints() == pool.get_checkpoints()


class TestHistoricalPoolPersistence:
    """Test historical pool save/load."""

    def test_save_and_load(self) -> None:
        """Pool should survive file save/load."""
        pool = HistoricalPool(max_size=5)
        pool.add_checkpoint("/path/1.pt", iteration=100, win_rate=0.55)
        pool.add_checkpoint("/path/2.pt", iteration=200, win_rate=0.65)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name

        try:
            save_historical_pool(pool, save_path)
            loaded = load_historical_pool(save_path)

            assert len(loaded) == len(pool)
            assert loaded.get_checkpoints() == pool.get_checkpoints()
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_load_nonexistent_raises(self) -> None:
        """Loading nonexistent file should raise."""
        with pytest.raises(FileNotFoundError):
            load_historical_pool("/nonexistent/pool.json")


# ============================================================================
# Integration Tests
# ============================================================================


class TestCurriculumIntegration:
    """Integration tests for curriculum system."""

    def test_full_training_simulation(self) -> None:
        """Simulate a training run with curriculum progression."""
        config = CurriculumConfig(
            min_games_per_level=20,
            max_games_per_level=50,
            advance_threshold=0.6,
            opponent_min_games=15,
            opponent_advance_threshold=0.55,
        )
        scheduler = CurriculumScheduler(config)

        # Simulate training games with improving win rate
        np.random.seed(42)

        for level_idx in range(3):  # Go through 3 levels
            # Get current level species
            species = scheduler.get_species_whitelist()
            assert len(species) >= MIN_SPECIES_FOR_GAME

            # Simulate games at this level
            games_played = 0
            while scheduler.current_level == level_idx + 1 and games_played < 100:
                # Create a game with current curriculum
                state = create_curriculum_game(
                    seed=np.random.randint(0, 10000),
                    species_whitelist=species,
                )
                assert state is not None

                # Simulate game result (improving win rate as training progresses)
                base_win_prob = 0.4 + (games_played / 100) * 0.4
                won = np.random.random() < base_win_prob
                scheduler.record_game(win=won)
                games_played += 1

        # Should have progressed through at least 2 levels
        assert scheduler.current_level >= 2 or len(scheduler.level_history) >= 1

    def test_opponent_stage_with_species_level(self) -> None:
        """Opponent stage should progress alongside species level."""
        config = CurriculumConfig(
            min_games_per_level=20,
            advance_threshold=0.7,
            opponent_min_games=20,
            opponent_advance_threshold=0.6,
        )
        scheduler = CurriculumScheduler(config)

        # Play games with good win rate
        for _ in range(30):
            scheduler.record_game(win=True)

        # Both should have advanced
        assert scheduler.current_level >= 2 or len(scheduler.level_history) >= 1
        assert scheduler.opponent_stage >= 2 or len(scheduler.stage_history) >= 1

    def test_scheduler_with_historical_pool(self) -> None:
        """Scheduler should work with historical pool integration."""
        scheduler = CurriculumScheduler()
        pool = HistoricalPool(max_size=5)

        # Simulate training iterations
        for iteration in range(100, 600, 100):
            # Add checkpoint to pool
            pool.add_checkpoint(
                f"/checkpoints/iter_{iteration}.pt",
                iteration=iteration,
                win_rate=0.5 + (iteration / 1000),
            )

            # Record games
            for _ in range(10):
                scheduler.record_game(win=np.random.random() > 0.4)

        # Pool should have checkpoints
        assert len(pool) >= 3

        # Should be able to get opponent mix for stage 5 (historical)
        mix = OPPONENT_STAGES[5]
        assert mix.historical_ratio > 0

        # Checkpoints should be available for sampling
        checkpoint = pool.sample_checkpoint()
        assert checkpoint is not None
