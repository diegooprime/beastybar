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
    transitions, trajectories, opponent_name = generator.generate_games(
        num_games=256,
        shaped_rewards=False,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig
from _03_training.opponent_pool import (
    OpponentPool,
    OpponentType,
    create_opponent_network,
)
from _03_training.self_play import (
    GameTrajectory,
    generate_games,
    trajectory_to_player_transitions,
)

if TYPE_CHECKING:
    import torch

    from _03_training.replay_buffer import Transition

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
        """
        self.network = network
        self.device = device
        self.opponent_pool = opponent_pool
        self.network_config = network_config or NetworkConfig()
        self.temperature = temperature
        self.num_workers = num_workers

    def generate_games(
        self,
        num_games: int,
        shaped_rewards: bool = False,
        temperature: float | None = None,
    ) -> tuple[list[Transition], list[list[Transition]], str]:
        """Generate self-play games and return transitions with trajectory info.

        Uses the actual self-play module to generate real game trajectories.
        When opponent pool is enabled, samples diverse opponents to prevent
        self-play collapse.

        Args:
            num_games: Number of games to generate.
            shaped_rewards: If True, use score-margin shaped rewards.
            temperature: Temperature for action sampling. Uses default if None.

        Returns:
            Tuple of (all_transitions, trajectory_list, opponent_name) where:
            - all_transitions: All transitions collected for replay buffer
            - trajectory_list: Separate lists per trajectory for GAE computation
            - opponent_name: Name of sampled opponent for logging
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
            elif sampled.opponent_type in (OpponentType.RANDOM, OpponentType.HEURISTIC):
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

        return all_transitions, trajectory_list, opponent_name

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
            elif sampled.opponent_type in (OpponentType.RANDOM, OpponentType.HEURISTIC):
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


__all__ = ["GameGenerator"]
