"""Self-play game generation for training.

This module provides the GameGenerator class responsible for generating
self-play games using a neural network, optionally with diverse opponents
from an opponent pool.

Example:
    generator = GameGenerator(
        network=network,
        device=device,
        opponent_pool=pool,
        temperature=1.0,
    )
    transitions, trajectories, opponent_name, win_rate = generator.generate_games(
        num_games=256,
        shaped_rewards=False,
    )
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig
from _03_training.opponent_pool import (
    OpponentPool,
    OpponentType,
    SampledOpponent,
    create_opponent_network,
)
from _03_training.opponent_statistics import OpponentStatsTracker
from _03_training.replay_buffer import Transition
from _03_training.self_play import (
    GameTrajectory,
    generate_games,
    trajectory_to_player_transitions,
)

logger = logging.getLogger(__name__)


class GameGenerator:
    """Handles self-play game generation with optional opponent diversity.

    This class encapsulates the logic for generating training games through
    self-play. It supports both standard self-play (same network for both
    players) and opponent diversity training (using past checkpoints, random,
    or heuristic opponents).

    Attributes:
        network: Neural network for policy/value estimation.
        device: Device for network inference.
        opponent_pool: Optional pool of diverse opponents.
        network_config: Network configuration for creating opponent networks.
        temperature: Default temperature for action sampling.
        num_workers: Number of parallel workers for game generation.

    Example:
        generator = GameGenerator(
            network=network,
            device=device,
            opponent_pool=pool,
        )
        transitions, trajectories, opponent = generator.generate_games(256)
    """

    def __init__(
        self,
        network: BeastyBarNetwork,
        device: torch.device,
        opponent_pool: OpponentPool | None = None,
        network_config: NetworkConfig | None = None,
        temperature: float = 1.0,
        num_workers: int = 1,
        stats_tracker: OpponentStatsTracker | None = None,
    ) -> None:
        """Initialize the game generator.

        Args:
            network: Neural network for policy/value estimation.
            device: Device for network inference.
            opponent_pool: Optional pool of diverse opponents.
            network_config: Network configuration for creating opponent networks.
                Required if opponent_pool contains checkpoints.
            temperature: Default temperature for action sampling.
            num_workers: Number of parallel workers for game generation.
            stats_tracker: Optional tracker for per-opponent win rate statistics.
        """
        self.network = network
        self.device = device
        self.opponent_pool = opponent_pool
        self.network_config = network_config or NetworkConfig()
        self.temperature = temperature
        self.num_workers = num_workers
        self.stats_tracker = stats_tracker

    def generate_games(
        self,
        num_games: int,
        shaped_rewards: bool = False,
        temperature: float | None = None,
        iteration: int = 0,
    ) -> tuple[list[Transition], list[list[Transition]], str, float]:
        """Generate self-play games and return transitions with trajectory info.

        Uses the actual self-play module to generate real game trajectories.
        When opponent pool is enabled, samples diverse opponents to prevent
        self-play collapse.

        Args:
            num_games: Number of games to generate.
            shaped_rewards: If True, use score-margin shaped rewards.
            temperature: Temperature for action sampling. Uses default if None.
            iteration: Current training iteration for stats tracking.

        Returns:
            Tuple of (all_transitions, trajectory_list, opponent_name, win_rate) where:
            - all_transitions: All transitions collected for replay buffer
            - trajectory_list: Separate lists per trajectory for GAE computation
            - opponent_name: Name of sampled opponent for logging
            - win_rate: Win rate against the opponent (0.5 for self-play)
        """
        temp = temperature if temperature is not None else self.temperature

        # Sample opponent from pool if enabled
        opponent_agent = None
        opponent_network = None
        opponent_name = "self"
        collect_both_players = True

        if self.opponent_pool is not None:
            sampled = self.opponent_pool.sample_opponent()
            opponent_name = sampled.name

            if sampled.opponent_type == OpponentType.CURRENT:
                # Standard self-play (both players use training network)
                pass
            elif sampled.opponent_type == OpponentType.CHECKPOINT:
                # Use past checkpoint network
                opponent_network = create_opponent_network(
                    sampled=sampled,
                    network_class=BeastyBarNetwork,
                    network_config=self.network_config,
                    device=self.device,
                )
                collect_both_players = False  # Only collect P0 (learning agent)
            elif sampled.opponent_type == OpponentType.MCTS:
                # Use MCTS agent from pool
                opponent_agent = sampled.agent
                collect_both_players = False  # Only collect P0 (learning agent)
            elif sampled.opponent_type in (OpponentType.RANDOM, OpponentType.HEURISTIC, OpponentType.OUTCOME_HEURISTIC):
                opponent_agent = sampled.agent
                collect_both_players = False  # Only collect P0 (learning agent)

        # Generate games with potentially diverse opponents
        trajectories = generate_games(
            network=self.network,
            num_games=num_games,
            temperature=temp,
            device=self.device,
            num_workers=self.num_workers,
            shaped_rewards=shaped_rewards,
            opponent=opponent_agent,
            opponent_network=opponent_network,
        )

        # Convert to transitions, keeping trajectory boundaries for GAE
        all_transitions: list[Transition] = []
        trajectory_list: list[list[Transition]] = []

        for trajectory in trajectories:
            # Determine which players to collect transitions from
            players_to_collect = [0, 1] if collect_both_players else [0]

            for player in players_to_collect:
                player_transitions = trajectory_to_player_transitions(trajectory, player)
                if player_transitions:
                    trajectory_list.append(player_transitions)
                    all_transitions.extend(player_transitions)

        # Compute win rate and update stats tracker if enabled
        win_rate = self._compute_and_track_results(
            trajectories=trajectories,
            opponent_name=opponent_name,
            iteration=iteration,
            is_self_play=collect_both_players,
        )

        return all_transitions, trajectory_list, opponent_name, win_rate

    def _compute_and_track_results(
        self,
        trajectories: list[GameTrajectory],
        opponent_name: str,
        iteration: int,
        is_self_play: bool,
    ) -> float:
        """Compute win rate and optionally update stats tracker.

        Args:
            trajectories: Generated game trajectories.
            opponent_name: Name of the opponent for stats tracking.
            iteration: Current training iteration.
            is_self_play: Whether this was self-play (both players same network).

        Returns:
            Win rate from P0's perspective (0.5 for self-play).
        """
        if is_self_play:
            # Self-play has no meaningful win rate against "self"
            return 0.5

        # Determine results from P0's perspective
        results: list[tuple[str, Literal["win", "loss", "draw"]]] = []
        wins = 0
        losses = 0
        draws = 0

        for trajectory in trajectories:
            # Determine result from P0's perspective using winner attribute
            # In self_play.py, P0 plays as player 0
            if trajectory.winner == 0:
                results.append((opponent_name, "win"))
                wins += 1
            elif trajectory.winner == 1:
                results.append((opponent_name, "loss"))
                losses += 1
            else:
                results.append((opponent_name, "draw"))
                draws += 1

        # Update stats tracker if enabled
        if self.stats_tracker is not None and results:
            self.stats_tracker.update_batch(results, iteration)

        # Compute win rate
        total = len(trajectories)
        if total == 0:
            return 0.5

        return wins / total

    def generate_trajectories(
        self,
        num_games: int,
        shaped_rewards: bool = False,
        temperature: float | None = None,
    ) -> list[GameTrajectory]:
        """Generate self-play games and return raw trajectories.

        A simpler interface that returns just the GameTrajectory objects
        without converting to transitions.

        Args:
            num_games: Number of games to generate.
            shaped_rewards: If True, use score-margin shaped rewards.
            temperature: Temperature for action sampling. Uses default if None.

        Returns:
            List of GameTrajectory objects.
        """
        temp = temperature if temperature is not None else self.temperature

        # Sample opponent from pool if enabled
        opponent_agent = None
        opponent_network = None

        if self.opponent_pool is not None:
            sampled = self.opponent_pool.sample_opponent()

            if sampled.opponent_type == OpponentType.CURRENT:
                pass
            elif sampled.opponent_type == OpponentType.CHECKPOINT:
                opponent_network = create_opponent_network(
                    sampled=sampled,
                    network_class=BeastyBarNetwork,
                    network_config=self.network_config,
                    device=self.device,
                )
            elif sampled.opponent_type == OpponentType.MCTS:
                # Use MCTS agent from pool
                opponent_agent = sampled.agent
            elif sampled.opponent_type in (OpponentType.RANDOM, OpponentType.HEURISTIC, OpponentType.OUTCOME_HEURISTIC):
                opponent_agent = sampled.agent

        return generate_games(
            network=self.network,
            num_games=num_games,
            temperature=temp,
            device=self.device,
            num_workers=self.num_workers,
            shaped_rewards=shaped_rewards,
            opponent=opponent_agent,
            opponent_network=opponent_network,
        )


def generate_games_batched_by_opponent(
    network: BeastyBarNetwork,
    num_games: int,
    opponent_pool: OpponentPool,
    device: torch.device,
    network_config: NetworkConfig | None = None,
    temperature: float = 1.0,
    num_workers: int = 1,
    shaped_rewards: bool = False,
) -> list[Transition]:
    """Generate games, batching by opponent type for efficiency.

    Instead of sampling opponent per-game, pre-compute how many games
    to play against each opponent type based on weights, then generate
    all games for each opponent in a batch.

    This is more efficient than per-game opponent sampling because:
    1. Reduces opponent setup overhead (network loading, agent creation)
    2. Allows better batching of games with same opponent type
    3. Deterministic opponent distribution (exact weights vs stochastic)

    Args:
        network: Neural network for policy/value estimation.
        num_games: Total number of games to generate.
        opponent_pool: Pool of diverse opponents with configured weights.
        device: Device for network inference.
        network_config: Network configuration for creating opponent networks.
        temperature: Temperature for action sampling.
        num_workers: Number of parallel workers for game generation.
        shaped_rewards: If True, use score-margin shaped rewards.

    Returns:
        List of all transitions collected from games against all opponents.
        Only P0 (learning agent) transitions are collected when playing
        against non-self opponents.
    """
    if network_config is None:
        network_config = NetworkConfig()

    # Get opponent weights from pool config
    config = opponent_pool.config
    weights = {
        OpponentType.CURRENT: config.current_weight,
        OpponentType.CHECKPOINT: config.checkpoint_weight if opponent_pool.checkpoints else 0.0,
        OpponentType.RANDOM: config.random_weight,
        OpponentType.HEURISTIC: config.heuristic_weight,
        OpponentType.OUTCOME_HEURISTIC: config.outcome_heuristic_weight,
        OpponentType.MCTS: config.mcts_weight if config.mcts_configs else 0.0,
    }

    # Redistribute unavailable weights to CURRENT
    unavailable = 0.0
    if not opponent_pool.checkpoints:
        unavailable += config.checkpoint_weight
    if not config.mcts_configs:
        unavailable += config.mcts_weight
    weights[OpponentType.CURRENT] += unavailable

    # Compute games per opponent type
    # Use floor division first, then distribute remainder
    games_per_opponent: dict[OpponentType, int] = {}
    remaining_games = num_games

    # Sort by weight descending for stable allocation
    sorted_types = sorted(weights.keys(), key=lambda t: weights[t], reverse=True)

    for opp_type in sorted_types:
        weight = weights[opp_type]
        if weight <= 0:
            games_per_opponent[opp_type] = 0
            continue

        # Compute games for this opponent
        games = int(weight * num_games)

        # Ensure at least 1 game for active opponents (weight > 0)
        if games == 0 and weight > 0:
            games = 1

        # Don't exceed remaining games
        games = min(games, remaining_games)
        games_per_opponent[opp_type] = games
        remaining_games -= games

    # Distribute any remaining games to CURRENT (highest priority)
    if remaining_games > 0:
        games_per_opponent[OpponentType.CURRENT] += remaining_games

    # Generate games for each opponent type
    all_transitions: list[Transition] = []

    for opp_type, n_games in games_per_opponent.items():
        if n_games <= 0:
            continue

        # Setup opponent for this batch
        opponent_agent = None
        opponent_network = None
        collect_both_players = opp_type == OpponentType.CURRENT

        if opp_type == OpponentType.CURRENT:
            # Standard self-play
            pass
        elif opp_type == OpponentType.CHECKPOINT:
            # Sample a checkpoint and use it for all games in this batch
            cp = opponent_pool.checkpoints[
                opponent_pool._rng.randrange(len(opponent_pool.checkpoints))
            ]
            sampled = SampledOpponent(
                OpponentType.CHECKPOINT,
                network_state=cp.state_dict,
                iteration=cp.iteration,
            )
            opponent_network = create_opponent_network(
                sampled=sampled,
                network_class=BeastyBarNetwork,
                network_config=network_config,
                device=device,
            )
        elif opp_type == OpponentType.RANDOM:
            opponent_agent = opponent_pool.random_agent
        elif opp_type == OpponentType.HEURISTIC:
            opponent_agent = opponent_pool.heuristic_agent
        elif opp_type == OpponentType.OUTCOME_HEURISTIC:
            opponent_agent = opponent_pool.outcome_heuristic_agent
        elif opp_type == OpponentType.MCTS:
            agents = opponent_pool.mcts_agents
            config_name = opponent_pool._rng.choice(list(agents.keys()))
            opponent_agent = agents[config_name]

        # Generate games for this opponent batch
        trajectories = generate_games(
            network=network,
            num_games=n_games,
            temperature=temperature,
            device=device,
            num_workers=num_workers,
            shaped_rewards=shaped_rewards,
            opponent=opponent_agent,
            opponent_network=opponent_network,
        )

        # Convert trajectories to transitions
        for trajectory in trajectories:
            players_to_collect = [0, 1] if collect_both_players else [0]
            for player in players_to_collect:
                player_transitions = trajectory_to_player_transitions(trajectory, player)
                all_transitions.extend(player_transitions)

        logger.debug(
            f"Generated {n_games} games vs {opp_type.name.lower()}, "
            f"collected {len(all_transitions)} transitions total"
        )

    return all_transitions


__all__ = ["GameGenerator", "OpponentStatsTracker", "generate_games_batched_by_opponent"]
