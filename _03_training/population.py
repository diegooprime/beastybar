"""Population-based training for Beasty Bar.

This module implements population-based training (PBT) where multiple agents
co-evolve through self-play. The key innovation is the exploit-patch cycle:
exploiter agents actively search for weaknesses in the population, and when
they succeed, their strategies get integrated into the main population.

Key concepts:
- Population: A diverse set of agents that compete via round-robin tournaments
- Exploiters: Agents trained specifically to find and exploit weaknesses
- ELO ratings: Track relative skill levels within the population
- Culling: Remove underperforming agents to maintain quality
- Diversity: Encourage strategic variety through population structure

Example:
    config = PopulationConfig(population_size=8, max_exploiters=2)
    trainer = PopulationTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.agent import NeuralAgent
from _02_agents.neural.utils import NetworkConfig, get_device, seed_all
from _03_training.alphazero_trainer import (
    AlphaZeroBuffer,
    TrainingExample,
    get_learning_rate,
    policy_loss_cross_entropy,
    set_learning_rate,
    value_loss_mse,
)
from _03_training.elo import Leaderboard
from _03_training.evaluation import compare_agents, run_evaluation
from _03_training.tracking import ExperimentTracker, create_tracker

logger = logging.getLogger(__name__)


def _ensure_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install with: pip install torch")


# ============================================================================
# Population Configuration
# ============================================================================


@dataclass
class PopulationConfig:
    """Configuration for population-based training.

    Attributes:
        network_config: Neural network architecture configuration.
        population_size: Number of agents in the main population.
        max_exploiters: Maximum number of concurrent exploiter agents.
        initial_rating: Starting ELO rating for new agents.
        k_factor: ELO K-factor for rating updates.
        exploit_threshold: Win rate needed for exploiter to join population.
        exploit_max_iterations: Maximum iterations before abandoning exploiter.
        exploit_eval_frequency: Evaluate exploiter progress every N iterations.
        exploit_games_per_eval: Games per exploiter evaluation.
        cull_threshold: Minimum win rate to remain in population.
        cull_frequency: Check for culling every N iterations.
        tournament_frequency: Run tournament every N iterations.
        tournament_games: Games per matchup in tournament.
        games_per_iteration: Self-play games per training iteration.
        total_iterations: Total number of training iterations.
        batch_size: Batch size for network training.
        epochs_per_iteration: Training epochs per iteration.
        learning_rate: Base learning rate.
        weight_decay: L2 regularization weight.
        max_grad_norm: Maximum gradient norm for clipping.
        num_simulations: MCTS simulations per move.
        c_puct: PUCT exploration constant.
        checkpoint_frequency: Save checkpoint every N iterations.
        eval_frequency: Evaluate vs baselines every N iterations.
        eval_games_per_opponent: Games per evaluation opponent.
        seed: Random seed.
        device: Training device.
        experiment_name: Name for experiment tracking.
        checkpoint_dir: Directory for checkpoints.
    """

    # Network
    network_config: NetworkConfig = field(default_factory=NetworkConfig)

    # Population settings
    population_size: int = 8
    max_exploiters: int = 2
    initial_rating: float = 1500.0
    k_factor: float = 32.0

    # Exploit-patch cycle
    exploit_threshold: float = 0.60
    exploit_max_iterations: int = 500
    exploit_eval_frequency: int = 50
    exploit_games_per_eval: int = 100

    # Population management
    cull_threshold: float = 0.30
    cull_frequency: int = 1000
    diversity_bonus: float = 0.1

    # Tournament
    tournament_frequency: int = 100
    tournament_games: int = 50

    # Training
    games_per_iteration: int = 8192
    total_iterations: int = 5000
    batch_size: int = 2048
    epochs_per_iteration: int = 4
    buffer_size: int = 2_000_000
    min_buffer_size: int = 10_000

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.5
    value_loss_weight: float = 1.0

    # MCTS
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    parallel_games: int = 64

    # Learning rate schedule
    lr_warmup_iterations: int = 50
    lr_decay: str = "cosine"

    # Checkpointing and evaluation
    checkpoint_frequency: int = 100
    eval_frequency: int = 100
    eval_games_per_opponent: int = 200
    eval_opponents: list[str] = field(
        default_factory=lambda: ["random", "heuristic", "outcome_heuristic", "mcts-100"]
    )

    # Infrastructure
    seed: int = 42
    device: str = "auto"
    experiment_name: str = "population_phase4"
    checkpoint_dir: str = "checkpoints/population"
    log_frequency: int = 1

    # Torch compile
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict

        result = asdict(self)
        result["network_config"] = self.network_config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PopulationConfig:
        """Create from dictionary."""
        data = data.copy()

        if "network_config" in data:
            if isinstance(data["network_config"], dict):
                data["network_config"] = NetworkConfig.from_dict(data["network_config"])
        else:
            data["network_config"] = NetworkConfig()

        # Handle nested population_config
        if "population_config" in data:
            pop_config = data.pop("population_config")
            data.update(pop_config)

        # Filter to known fields
        import dataclasses

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PopulationConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.max_exploiters < 0:
            raise ValueError(f"max_exploiters must be >= 0, got {self.max_exploiters}")
        if not 0 < self.exploit_threshold < 1:
            raise ValueError(f"exploit_threshold must be in (0, 1), got {self.exploit_threshold}")
        if not 0 < self.cull_threshold < 1:
            raise ValueError(f"cull_threshold must be in (0, 1), got {self.cull_threshold}")
        if self.games_per_iteration <= 0:
            raise ValueError(f"games_per_iteration must be positive, got {self.games_per_iteration}")


# ============================================================================
# Population Member
# ============================================================================


@dataclass
class PopulationMember:
    """Represents a single agent in the population.

    Attributes:
        agent_id: Unique identifier for this agent.
        network: The neural network for this agent.
        optimizer: Optimizer for training this network.
        generation: Generation number (increases when evolved from exploiter).
        parent_id: ID of parent agent (if evolved from exploiter).
        created_iteration: Iteration when this agent was created.
        total_games: Total games played by this agent.
        total_wins: Total wins.
        total_losses: Total losses.
        total_draws: Total draws.
    """

    agent_id: str
    network: BeastyBarNetwork
    optimizer: torch.optim.Optimizer
    generation: int = 0
    parent_id: str | None = None
    created_iteration: int = 0
    total_games: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_draws: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate overall win rate."""
        if self.total_games == 0:
            return 0.5  # Default for new agents
        return self.total_wins / self.total_games

    def record_result(self, won: bool, lost: bool, drawn: bool) -> None:
        """Record a game result."""
        self.total_games += 1
        if won:
            self.total_wins += 1
        elif lost:
            self.total_losses += 1
        elif drawn:
            self.total_draws += 1


# ============================================================================
# Exploiter Agent
# ============================================================================


@dataclass
class Exploiter:
    """An exploiter agent trained to beat a specific target.

    Attributes:
        exploiter_id: Unique identifier.
        network: Neural network for exploiter.
        optimizer: Optimizer for training.
        target_id: ID of the target agent to exploit.
        created_iteration: When this exploiter was spawned.
        training_iterations: Number of training iterations so far.
        best_win_rate: Best win rate achieved against target.
        current_win_rate: Current win rate against target.
        abandoned: Whether this exploiter has been abandoned.
    """

    exploiter_id: str
    network: BeastyBarNetwork
    optimizer: torch.optim.Optimizer
    target_id: str
    created_iteration: int
    training_iterations: int = 0
    best_win_rate: float = 0.0
    current_win_rate: float = 0.0
    abandoned: bool = False


# ============================================================================
# Population Trainer
# ============================================================================


class PopulationTrainer:
    """Population-based trainer for Beasty Bar.

    Implements the exploit-patch training cycle:
    1. Maintain a diverse population of agents
    2. Run tournaments to update ELO ratings
    3. Train all agents via self-play against population
    4. Spawn exploiters to find weaknesses in best agents
    5. When exploiters succeed (>threshold win rate), add to population
    6. Cull weak agents to maintain population quality

    Attributes:
        config: Training configuration.
        population: List of population members.
        exploiters: List of active exploiter agents.
        leaderboard: ELO rating tracker.
        device: Training device.
    """

    def __init__(
        self,
        config: PopulationConfig,
        tracker: ExperimentTracker | None = None,
        initial_network: BeastyBarNetwork | None = None,
    ) -> None:
        """Initialize population trainer.

        Args:
            config: Training configuration.
            tracker: Optional experiment tracker.
            initial_network: Optional network to seed population.
        """
        _ensure_torch()
        config.validate()

        self.config = config
        self._iteration = 0
        self._total_games_played = 0
        self._exploiter_counter = 0
        self._training_start_time: float | None = None

        # Set random seeds
        seed_all(config.seed)

        # Determine device
        if config.device == "auto":
            self._device = get_device()
        else:
            self._device = torch.device(config.device)
        logger.info(f"Using device: {self._device}")

        # Initialize leaderboard
        from _03_training.elo import EloRating

        elo = EloRating(k_factor=config.k_factor, initial_rating=config.initial_rating)
        self.leaderboard = Leaderboard(elo=elo, initial_rating=config.initial_rating)

        # Initialize population
        self.population: list[PopulationMember] = []
        self._initialize_population(initial_network)

        # Initialize exploiters list
        self.exploiters: list[Exploiter] = []

        # Create shared replay buffer (all agents contribute)
        self.replay_buffer = AlphaZeroBuffer(
            max_size=config.buffer_size,
            observation_dim=config.network_config.observation_dim,
            action_dim=config.network_config.action_dim,
        )

        # Create or use provided tracker
        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = create_tracker(
                backend="console",
                project="beastybar",
                run_name=config.experiment_name,
                config=config.to_dict(),
            )

        # Checkpoint directory
        self._checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics history
        self._metrics_history: list[dict[str, float]] = []

        # BatchMCTS for game generation (shared across population)
        self._batch_mcts = None

        logger.info(
            f"Population Trainer initialized with {len(self.population)} agents, "
            f"max {config.max_exploiters} exploiters"
        )

    def _initialize_population(self, initial_network: BeastyBarNetwork | None) -> None:
        """Initialize the population with diverse agents.

        Args:
            initial_network: Optional network to seed population.
        """
        logger.info(f"Initializing population with {self.config.population_size} agents")

        for i in range(self.config.population_size):
            agent_id = f"agent_{i:03d}"

            if initial_network is not None and i == 0:
                # First agent uses provided network
                network = copy.deepcopy(initial_network).to(self._device)
            else:
                # Create new random network
                network = BeastyBarNetwork(self.config.network_config).to(self._device)

                # Apply slight noise to weights for diversity
                if initial_network is not None:
                    # Clone from initial but add noise
                    network.load_state_dict(initial_network.state_dict())
                    with torch.no_grad():
                        for param in network.parameters():
                            noise = torch.randn_like(param) * 0.01
                            param.add_(noise)

            # Create optimizer
            optimizer = torch.optim.AdamW(
                network.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            member = PopulationMember(
                agent_id=agent_id,
                network=network,
                optimizer=optimizer,
                generation=0,
                created_iteration=0,
            )

            self.population.append(member)
            self.leaderboard.register(agent_id)

            logger.debug(f"Created population member: {agent_id}")

    def _get_batch_mcts(self, network: BeastyBarNetwork):
        """Get BatchMCTS instance for a specific network."""
        from _02_agents.mcts.batch_mcts import BatchMCTS

        return BatchMCTS(
            network=network,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            batch_size=max(16, self.config.parallel_games * 2),
            device=self._device,
        )

    def _get_best_agent(self) -> PopulationMember:
        """Get the agent with highest ELO rating."""
        best_rating = -float("inf")
        best_member = self.population[0]

        for member in self.population:
            rating = self.leaderboard.get_rating(member.agent_id)
            if rating > best_rating:
                best_rating = rating
                best_member = member

        return best_member

    def _get_current_lr(self) -> float:
        """Get learning rate for current iteration."""
        return get_learning_rate(
            iteration=self._iteration,
            total_iterations=self.config.total_iterations,
            base_lr=self.config.learning_rate,
            warmup_iterations=self.config.lr_warmup_iterations,
            decay_type=self.config.lr_decay,
        )

    # ========================================================================
    # Tournament and ELO Updates
    # ========================================================================

    def run_tournament(self) -> dict[str, float]:
        """Run a round-robin tournament within the population.

        Each agent plays tournament_games games against every other agent.
        Results update the ELO leaderboard.

        Returns:
            Dictionary of tournament metrics.
        """
        logger.info(f"Running tournament at iteration {self._iteration}")
        tournament_start = time.time()

        games_played = 0
        total_matchups = 0

        # Round-robin: each pair plays
        for i, member_a in enumerate(self.population):
            agent_a = NeuralAgent(member_a.network, device=self._device, mode="greedy")

            for member_b in self.population[i + 1 :]:
                agent_b = NeuralAgent(member_b.network, device=self._device, mode="greedy")

                # Play games (alternating sides)
                results = compare_agents(
                    agent_a,
                    agent_b,
                    num_games=self.config.tournament_games,
                    play_both_sides=True,
                    seed=self._iteration * 1000 + total_matchups,
                )

                # Update leaderboard
                for _ in range(results["wins_1"]):
                    self.leaderboard.record_match(
                        member_a.agent_id, member_b.agent_id, 1, 0
                    )
                    member_a.record_result(won=True, lost=False, drawn=False)
                    member_b.record_result(won=False, lost=True, drawn=False)

                for _ in range(results["wins_2"]):
                    self.leaderboard.record_match(
                        member_a.agent_id, member_b.agent_id, 0, 1
                    )
                    member_a.record_result(won=False, lost=True, drawn=False)
                    member_b.record_result(won=True, lost=False, drawn=False)

                for _ in range(results["draws"]):
                    self.leaderboard.record_match(
                        member_a.agent_id, member_b.agent_id, 0, 0
                    )
                    member_a.record_result(won=False, lost=False, drawn=True)
                    member_b.record_result(won=False, lost=False, drawn=True)

                games_played += self.config.tournament_games
                total_matchups += 1

        tournament_time = time.time() - tournament_start

        # Log rankings
        logger.info("Tournament Rankings:")
        for rank, stats in enumerate(self.leaderboard.rankings()[:5], 1):
            logger.info(
                f"  {rank}. {stats.name}: ELO={stats.rating:.0f}, "
                f"W/L/D={stats.wins}/{stats.losses}/{stats.draws}"
            )

        return {
            "tournament/games_played": float(games_played),
            "tournament/matchups": float(total_matchups),
            "tournament/time": tournament_time,
            "tournament/top_elo": self.leaderboard.rankings()[0].rating,
        }

    # ========================================================================
    # Self-Play Training
    # ========================================================================

    def generate_self_play_data(
        self,
        member: PopulationMember,
        opponents: list[PopulationMember],
        num_games: int,
    ) -> list[TrainingExample]:
        """Generate training data from self-play against population.

        Args:
            member: The agent generating training data.
            opponents: List of opponent agents to play against.
            num_games: Total number of games to play.

        Returns:
            List of training examples from member's perspective.
        """
        from _01_simulator import simulate
        from _01_simulator.action_space import (
            ACTION_DIM,
            index_to_action,
            legal_action_mask_tensor,
        )
        from _01_simulator.observations import state_to_tensor
        from _03_training.game_utils import compute_rewards, compute_winner
        from _03_training.utils import inference_mode

        examples: list[TrainingExample] = []
        batch_mcts = self._get_batch_mcts(member.network)

        games_per_opponent = max(1, num_games // len(opponents))

        with inference_mode(member.network):
            for opponent in opponents:
                for game_idx in range(games_per_opponent):
                    game_examples: list[tuple] = []

                    seed = np.random.randint(0, 2**31)
                    state = simulate.new_game(seed)

                    # Alternate who plays as player 0
                    member_player = game_idx % 2

                    while not simulate.is_terminal(state):
                        player = state.active_player

                        if player == member_player:
                            # Member's turn - use MCTS
                            visit_dist = batch_mcts.search_batch(
                                states=[state],
                                perspective=player,
                                add_root_noise=True,
                            )[0]

                            if not visit_dist:
                                mask = legal_action_mask_tensor(state, player)
                                legal = np.where(mask > 0)[0]
                                visit_dist = {int(a): 1.0 / len(legal) for a in legal}

                            # Store example
                            mcts_policy = np.zeros(ACTION_DIM, dtype=np.float32)
                            for action_idx, prob in visit_dist.items():
                                if 0 <= action_idx < ACTION_DIM:
                                    mcts_policy[action_idx] = prob

                            obs = state_to_tensor(state, player)
                            mask = legal_action_mask_tensor(state, player)
                            game_examples.append((obs, mask, mcts_policy, player))

                            # Sample action
                            actions = list(visit_dist.keys())
                            probs = np.array([visit_dist[a] for a in actions])
                            probs = probs / (probs.sum() + 1e-10)
                            action_idx = actions[np.random.choice(len(actions), p=probs)]
                            action = index_to_action(action_idx)
                        else:
                            # Opponent's turn - use greedy
                            with inference_mode(opponent.network):
                                agent = NeuralAgent(
                                    opponent.network, device=self._device, mode="greedy"
                                )
                                legal = list(simulate.legal_actions(state, player))
                                action = agent.select_action(state, legal)

                        state = simulate.apply(state, action)

                    # Compute outcome
                    scores = simulate.score(state)
                    winner = compute_winner((scores[0], scores[1]))
                    value_p0, value_p1 = compute_rewards(
                        winner, (scores[0], scores[1]), shaped=False
                    )

                    # Create training examples
                    for obs, mask, policy, player in game_examples:
                        value = value_p0 if player == 0 else value_p1
                        examples.append(
                            TrainingExample(
                                observation=obs,
                                action_mask=mask,
                                mcts_policy=policy,
                                value=value,
                            )
                        )

        return examples

    def train_member(self, member: PopulationMember) -> dict[str, float]:
        """Train a single population member on the shared replay buffer.

        Args:
            member: Population member to train.

        Returns:
            Dictionary of training metrics.
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return {}

        # Update learning rate
        current_lr = self._get_current_lr()
        set_learning_rate(member.optimizer, current_lr)

        # Set network to training mode
        member.network.train()

        # Get all data from buffer
        observations, action_masks, mcts_policies, values = self.replay_buffer.sample_all()
        num_samples = len(observations)

        # Convert to tensors
        obs_tensor = torch.from_numpy(observations).to(self._device)
        mask_tensor = torch.from_numpy(action_masks).to(self._device)
        policy_tensor = torch.from_numpy(mcts_policies).to(self._device)
        value_tensor = torch.from_numpy(values).to(self._device)

        # Train for multiple epochs
        metrics_accum = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
        num_updates = 0

        for _epoch in range(self.config.epochs_per_iteration):
            indices = torch.randperm(num_samples, device=self._device)

            for start_idx in range(0, num_samples, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                obs_batch = obs_tensor[batch_indices]
                mask_batch = mask_tensor[batch_indices]
                policy_batch = policy_tensor[batch_indices]
                value_batch = value_tensor[batch_indices]

                # Forward pass
                policy_logits, pred_values = member.network(obs_batch, mask_batch)
                pred_values = pred_values.squeeze(-1)

                # Compute losses
                p_loss = policy_loss_cross_entropy(policy_logits, policy_batch, mask_batch)
                v_loss = value_loss_mse(pred_values, value_batch)
                total_loss = p_loss + self.config.value_loss_weight * v_loss

                # Optimization step
                member.optimizer.zero_grad()
                total_loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        member.network.parameters(),
                        self.config.max_grad_norm,
                    )

                member.optimizer.step()

                metrics_accum["policy_loss"] += p_loss.item()
                metrics_accum["value_loss"] += v_loss.item()
                metrics_accum["total_loss"] += total_loss.item()
                num_updates += 1

        if num_updates > 0:
            for key in metrics_accum:
                metrics_accum[key] /= num_updates

        metrics_accum["learning_rate"] = current_lr

        return metrics_accum

    # ========================================================================
    # Exploit-Patch Cycle
    # ========================================================================

    def spawn_exploiter(self) -> Exploiter | None:
        """Spawn a new exploiter agent targeting the best population member.

        Returns:
            New Exploiter if spawned, None if at capacity.
        """
        if len(self.exploiters) >= self.config.max_exploiters:
            return None

        best_agent = self._get_best_agent()

        # Clone best agent's network
        network = copy.deepcopy(best_agent.network).to(self._device)

        # Apply noise for diversity
        with torch.no_grad():
            for param in network.parameters():
                noise = torch.randn_like(param) * 0.02
                param.add_(noise)

        # Create optimizer with higher learning rate for faster adaptation
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate * 2.0,  # Faster learning
            weight_decay=self.config.weight_decay,
        )

        exploiter_id = f"exploiter_{self._exploiter_counter:03d}"
        self._exploiter_counter += 1

        exploiter = Exploiter(
            exploiter_id=exploiter_id,
            network=network,
            optimizer=optimizer,
            target_id=best_agent.agent_id,
            created_iteration=self._iteration,
        )

        self.exploiters.append(exploiter)
        logger.info(f"Spawned {exploiter_id} targeting {best_agent.agent_id}")

        return exploiter

    def train_exploiter(self, exploiter: Exploiter) -> dict[str, float]:
        """Train an exploiter against its target.

        Uses reward shaping: extra bonus for winning against target.

        Args:
            exploiter: Exploiter to train.

        Returns:
            Dictionary of training metrics.
        """
        from _01_simulator import simulate
        from _01_simulator.action_space import (
            ACTION_DIM,
            index_to_action,
            legal_action_mask_tensor,
        )
        from _01_simulator.observations import state_to_tensor
        from _03_training.game_utils import compute_winner
        from _03_training.utils import inference_mode

        # Find target agent
        target = None
        for member in self.population:
            if member.agent_id == exploiter.target_id:
                target = member
                break

        if target is None:
            logger.warning(f"Target {exploiter.target_id} not found, abandoning exploiter")
            exploiter.abandoned = True
            return {}

        # Generate games against target
        examples: list[TrainingExample] = []
        batch_mcts = self._get_batch_mcts(exploiter.network)
        num_games = 50  # Fewer games for faster iteration

        wins = 0
        games = 0

        with inference_mode(exploiter.network):
            for game_idx in range(num_games):
                game_examples: list[tuple] = []
                seed = np.random.randint(0, 2**31)
                state = simulate.new_game(seed)

                exploiter_player = game_idx % 2

                while not simulate.is_terminal(state):
                    player = state.active_player

                    if player == exploiter_player:
                        visit_dist = batch_mcts.search_batch(
                            states=[state],
                            perspective=player,
                            add_root_noise=True,
                        )[0]

                        if not visit_dist:
                            mask = legal_action_mask_tensor(state, player)
                            legal = np.where(mask > 0)[0]
                            visit_dist = {int(a): 1.0 / len(legal) for a in legal}

                        mcts_policy = np.zeros(ACTION_DIM, dtype=np.float32)
                        for action_idx, prob in visit_dist.items():
                            if 0 <= action_idx < ACTION_DIM:
                                mcts_policy[action_idx] = prob

                        obs = state_to_tensor(state, player)
                        mask = legal_action_mask_tensor(state, player)
                        game_examples.append((obs, mask, mcts_policy, player))

                        actions = list(visit_dist.keys())
                        probs = np.array([visit_dist[a] for a in actions])
                        probs = probs / (probs.sum() + 1e-10)
                        action_idx = actions[np.random.choice(len(actions), p=probs)]
                        action = index_to_action(action_idx)
                    else:
                        with inference_mode(target.network):
                            agent = NeuralAgent(
                                target.network, device=self._device, mode="greedy"
                            )
                            legal = list(simulate.legal_actions(state, player))
                            action = agent.select_action(state, legal)

                    state = simulate.apply(state, action)

                scores = simulate.score(state)
                winner = compute_winner((scores[0], scores[1]))
                games += 1

                # Determine exploiter's value
                if winner == exploiter_player:
                    wins += 1
                    base_value = 1.0
                elif winner == 1 - exploiter_player:
                    base_value = -1.0
                else:
                    base_value = 0.0

                # Reward shaping: bonus for beating target
                if base_value > 0:
                    value = min(1.0, base_value * 1.2)  # 20% bonus for wins
                else:
                    value = base_value

                for obs, mask, policy, player in game_examples:
                    player_value = value if player == exploiter_player else -value
                    examples.append(
                        TrainingExample(
                            observation=obs,
                            action_mask=mask,
                            mcts_policy=policy,
                            value=player_value,
                        )
                    )

        # Train on collected examples
        if not examples:
            return {}

        exploiter.network.train()

        obs = np.stack([e.observation for e in examples])
        masks = np.stack([e.action_mask for e in examples])
        policies = np.stack([e.mcts_policy for e in examples])
        values = np.array([e.value for e in examples])

        obs_tensor = torch.from_numpy(obs).to(self._device)
        mask_tensor = torch.from_numpy(masks).to(self._device)
        policy_tensor = torch.from_numpy(policies).to(self._device)
        value_tensor = torch.from_numpy(values).to(self._device)

        # Multiple training passes
        for _ in range(4):
            indices = torch.randperm(len(examples), device=self._device)

            for start_idx in range(0, len(examples), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(examples))
                batch_indices = indices[start_idx:end_idx]

                policy_logits, pred_values = exploiter.network(
                    obs_tensor[batch_indices], mask_tensor[batch_indices]
                )
                pred_values = pred_values.squeeze(-1)

                p_loss = policy_loss_cross_entropy(
                    policy_logits, policy_tensor[batch_indices], mask_tensor[batch_indices]
                )
                v_loss = value_loss_mse(pred_values, value_tensor[batch_indices])
                total_loss = p_loss + self.config.value_loss_weight * v_loss

                exploiter.optimizer.zero_grad()
                total_loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        exploiter.network.parameters(),
                        self.config.max_grad_norm,
                    )

                exploiter.optimizer.step()

        exploiter.training_iterations += 1
        exploiter.current_win_rate = wins / games if games > 0 else 0.0
        exploiter.best_win_rate = max(exploiter.best_win_rate, exploiter.current_win_rate)

        return {
            f"exploiter/{exploiter.exploiter_id}/win_rate": exploiter.current_win_rate,
            f"exploiter/{exploiter.exploiter_id}/best_win_rate": exploiter.best_win_rate,
            f"exploiter/{exploiter.exploiter_id}/iterations": float(
                exploiter.training_iterations
            ),
        }

    def evaluate_exploiter(self, exploiter: Exploiter) -> float:
        """Evaluate exploiter win rate against target.

        Args:
            exploiter: Exploiter to evaluate.

        Returns:
            Win rate against target.
        """
        target = None
        for member in self.population:
            if member.agent_id == exploiter.target_id:
                target = member
                break

        if target is None:
            return 0.0

        exploiter_agent = NeuralAgent(exploiter.network, device=self._device, mode="greedy")
        target_agent = NeuralAgent(target.network, device=self._device, mode="greedy")

        results = compare_agents(
            exploiter_agent,
            target_agent,
            num_games=self.config.exploit_games_per_eval,
            play_both_sides=True,
            seed=self._iteration,
        )

        return results["win_rate_1"]

    def integrate_exploiter(self, exploiter: Exploiter) -> PopulationMember:
        """Integrate a successful exploiter into the population.

        Replaces the lowest-rated population member.

        Args:
            exploiter: Successful exploiter to integrate.

        Returns:
            New population member created from exploiter.
        """
        # Find lowest rated member
        worst_member = min(
            self.population,
            key=lambda m: self.leaderboard.get_rating(m.agent_id),
        )

        # Create new population member
        new_agent_id = f"agent_{len(self.population) + self._exploiter_counter:03d}"

        new_member = PopulationMember(
            agent_id=new_agent_id,
            network=exploiter.network,
            optimizer=exploiter.optimizer,
            generation=worst_member.generation + 1,
            parent_id=exploiter.target_id,
            created_iteration=self._iteration,
        )

        # Replace worst member
        self.population.remove(worst_member)
        self.population.append(new_member)

        # Register in leaderboard
        self.leaderboard.register(new_agent_id)

        logger.info(
            f"Integrated {exploiter.exploiter_id} as {new_agent_id}, "
            f"replaced {worst_member.agent_id}"
        )

        return new_member

    # ========================================================================
    # Population Management
    # ========================================================================

    def cull_weak_agents(self) -> list[str]:
        """Remove agents below cull_threshold win rate.

        Weak agents are replaced with clones of the best agent (with noise).

        Returns:
            List of culled agent IDs.
        """
        if len(self.population) <= 2:
            return []  # Keep minimum population

        culled = []
        best_agent = self._get_best_agent()

        for member in list(self.population):
            if member.agent_id == best_agent.agent_id:
                continue  # Never cull the best

            # Check win rate from leaderboard stats
            stats = self.leaderboard.get_stats(member.agent_id)
            if stats.games > 0 and stats.win_rate < self.config.cull_threshold:
                logger.info(
                    f"Culling {member.agent_id}: win_rate={stats.win_rate:.1%} "
                    f"< threshold={self.config.cull_threshold:.1%}"
                )

                # Replace with clone of best
                new_network = copy.deepcopy(best_agent.network).to(self._device)
                with torch.no_grad():
                    for param in new_network.parameters():
                        noise = torch.randn_like(param) * 0.01
                        param.add_(noise)

                member.network = new_network
                member.optimizer = torch.optim.AdamW(
                    new_network.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
                member.generation += 1
                member.total_games = 0
                member.total_wins = 0
                member.total_losses = 0
                member.total_draws = 0

                culled.append(member.agent_id)

        return culled

    # ========================================================================
    # Main Training Loop
    # ========================================================================

    def train_iteration(self) -> dict[str, float]:
        """Execute a single training iteration.

        Returns:
            Dictionary of metrics from this iteration.
        """
        iteration_start = time.time()
        metrics: dict[str, float] = {"iteration": float(self._iteration)}

        # 1. Generate self-play data for each agent
        gen_start = time.time()
        all_examples: list[TrainingExample] = []

        games_per_agent = self.config.games_per_iteration // len(self.population)

        for member in self.population:
            # Play against random subset of population
            opponents = [m for m in self.population if m.agent_id != member.agent_id]
            if len(opponents) > 3:
                opponents = list(np.random.choice(opponents, size=3, replace=False))

            examples = self.generate_self_play_data(member, opponents, games_per_agent)
            all_examples.extend(examples)

        self.replay_buffer.add_batch(all_examples)
        gen_time = time.time() - gen_start
        metrics["self_play/generation_time"] = gen_time
        metrics["self_play/examples_collected"] = float(len(all_examples))

        # 2. Train all population members on shared buffer
        train_start = time.time()
        train_metrics: dict[str, float] = {}

        for member in self.population:
            member_metrics = self.train_member(member)
            for key, value in member_metrics.items():
                train_metrics[f"train/{member.agent_id}/{key}"] = value

        # Aggregate average training metrics
        if train_metrics:
            policy_losses = [
                v for k, v in train_metrics.items() if k.endswith("/policy_loss")
            ]
            value_losses = [v for k, v in train_metrics.items() if k.endswith("/value_loss")]
            if policy_losses:
                metrics["train/avg_policy_loss"] = np.mean(policy_losses)
            if value_losses:
                metrics["train/avg_value_loss"] = np.mean(value_losses)

        train_time = time.time() - train_start
        metrics["train/training_time"] = train_time

        # 3. Exploiter management
        # Train existing exploiters
        for exploiter in self.exploiters:
            if exploiter.abandoned:
                continue

            exp_metrics = self.train_exploiter(exploiter)
            metrics.update(exp_metrics)

            # Check if exploiter should be evaluated
            if (
                exploiter.training_iterations > 0
                and exploiter.training_iterations % self.config.exploit_eval_frequency == 0
            ):
                win_rate = self.evaluate_exploiter(exploiter)
                exploiter.current_win_rate = win_rate
                exploiter.best_win_rate = max(exploiter.best_win_rate, win_rate)

                logger.info(
                    f"{exploiter.exploiter_id} vs {exploiter.target_id}: "
                    f"win_rate={win_rate:.1%}"
                )

                # Check if exploiter succeeded
                if win_rate >= self.config.exploit_threshold:
                    logger.info(
                        f"{exploiter.exploiter_id} succeeded! "
                        f"Integrating into population"
                    )
                    self.integrate_exploiter(exploiter)
                    exploiter.abandoned = True

                # Check if exploiter should be abandoned
                elif (
                    exploiter.training_iterations >= self.config.exploit_max_iterations
                ):
                    logger.info(
                        f"Abandoning {exploiter.exploiter_id}: "
                        f"max iterations reached, best={exploiter.best_win_rate:.1%}"
                    )
                    exploiter.abandoned = True

        # Remove abandoned exploiters
        self.exploiters = [e for e in self.exploiters if not e.abandoned]

        # Spawn new exploiters if needed
        if len(self.exploiters) < self.config.max_exploiters:
            self.spawn_exploiter()

        metrics["exploiters/active"] = float(len(self.exploiters))

        # 4. Tournament (periodic)
        if (
            self.config.tournament_frequency > 0
            and (self._iteration + 1) % self.config.tournament_frequency == 0
        ):
            tournament_metrics = self.run_tournament()
            metrics.update(tournament_metrics)

        # 5. Culling (periodic)
        if (
            self.config.cull_frequency > 0
            and (self._iteration + 1) % self.config.cull_frequency == 0
        ):
            culled = self.cull_weak_agents()
            metrics["population/culled"] = float(len(culled))

        # Population stats
        metrics["population/size"] = float(len(self.population))
        metrics["population/top_elo"] = self.leaderboard.rankings()[0].rating

        iteration_time = time.time() - iteration_start
        metrics["iteration_time"] = iteration_time

        self._metrics_history.append(metrics)

        return metrics

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(
            f"Starting Population Training for {self.config.total_iterations} iterations"
        )
        self._training_start_time = time.time()

        # Log hyperparameters
        self.tracker.log_hyperparameters(self.config.to_dict())

        try:
            while self._iteration < self.config.total_iterations:
                metrics = self.train_iteration()

                # Log metrics
                if self._iteration % self.config.log_frequency == 0:
                    self.tracker.log_metrics(metrics, step=self._iteration)

                    elapsed = time.time() - self._training_start_time
                    eta = (
                        elapsed
                        / (self._iteration + 1)
                        * (self.config.total_iterations - self._iteration - 1)
                    )
                    logger.info(
                        f"Iteration {self._iteration}/{self.config.total_iterations} | "
                        f"Top ELO: {metrics.get('population/top_elo', 0):.0f} | "
                        f"Policy: {metrics.get('train/avg_policy_loss', 0):.4f} | "
                        f"Value: {metrics.get('train/avg_value_loss', 0):.4f} | "
                        f"Exploiters: {int(metrics.get('exploiters/active', 0))} | "
                        f"ETA: {eta / 60:.1f}min"
                    )

                # Checkpoint
                if (self._iteration + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint()

                # Evaluation
                if (self._iteration + 1) % self.config.eval_frequency == 0:
                    self._run_evaluation()

                self._iteration += 1

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            self._save_checkpoint(is_final=True)
            self.tracker.finish()

            total_time = time.time() - self._training_start_time
            logger.info(f"Training completed in {total_time / 60:.1f} minutes")
            logger.info(f"Final population rankings:")
            logger.info(self.leaderboard.summary())

    def _save_checkpoint(self, is_final: bool = False) -> Path:
        """Save training checkpoint."""
        checkpoint_name = "final.pt" if is_final else f"iter_{self._iteration:06d}.pt"
        checkpoint_path = self._checkpoint_dir / checkpoint_name

        # Save best agent's network as the main checkpoint
        best_agent = self._get_best_agent()

        checkpoint: dict[str, Any] = {
            "iteration": self._iteration,
            "model_state_dict": best_agent.network.state_dict(),
            "config": self.config.to_dict(),
            "population": [
                {
                    "agent_id": m.agent_id,
                    "state_dict": m.network.state_dict(),
                    "generation": m.generation,
                    "total_games": m.total_games,
                    "total_wins": m.total_wins,
                }
                for m in self.population
            ],
            "leaderboard": {
                name: stats.to_dict()
                for name, stats in self.leaderboard._players.items()
            },
            "metrics_history": self._metrics_history,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
        }

        torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save config as JSON
        config_path = checkpoint_path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self.tracker.log_artifact(str(checkpoint_path), checkpoint_name)

        return checkpoint_path

    def _run_evaluation(self) -> dict[str, float]:
        """Run evaluation of best agent against baselines."""
        from _03_training.utils import inference_mode

        best_agent = self._get_best_agent()
        logger.info(
            f"Running evaluation at iteration {self._iteration} "
            f"(best: {best_agent.agent_id})"
        )

        with inference_mode(best_agent.network):
            metrics = run_evaluation(
                network=best_agent.network,
                device=self._device,
                tracker=self.tracker,
                step=self._iteration,
                games_per_opponent=self.config.eval_games_per_opponent,
                opponents=self.config.eval_opponents,
                play_both_sides=True,
                mode="greedy",
            )

        logger.info(
            "Evaluation results: "
            + ", ".join(
                f"{k.split('/')[-2]}={v:.1%}"
                for k, v in metrics.items()
                if k.endswith("/win_rate")
            )
        )

        return metrics


__all__ = [
    "Exploiter",
    "PopulationConfig",
    "PopulationMember",
    "PopulationTrainer",
]
