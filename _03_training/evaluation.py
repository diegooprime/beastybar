"""Evaluation system for measuring agent performance against baselines.

This module provides tools for:
- Periodic evaluation against fixed opponent pools
- Statistical significance testing for win rates
- ELO estimation from evaluation results
- Comparative analysis between agents
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from _01_simulator import engine, simulate

if TYPE_CHECKING:
    from _02_agents.base import Agent

    from .tracking import ExperimentTracker


@dataclass
class EvaluationConfig:
    """Configuration for agent evaluation.

    Attributes:
        games_per_opponent: Number of games to play against each opponent.
        opponents: List of opponent names to evaluate against.
            Available: "random", "heuristic", "mcts-100", "mcts-500", "mcts-1000"
        play_both_sides: If True, play as both player 0 and player 1.
        seeds: Optional fixed seeds for reproducibility. If None, uses sequential seeds.
    """

    games_per_opponent: int = 100
    opponents: list[str] = field(default_factory=lambda: ["random", "heuristic"])
    play_both_sides: bool = True
    seeds: list[int] | None = None


@dataclass
class EvaluationResult:
    """Result of evaluating an agent against a single opponent.

    Attributes:
        opponent_name: Name of the opponent agent.
        games_played: Total number of games played.
        wins: Number of wins for the evaluated agent.
        losses: Number of losses for the evaluated agent.
        draws: Number of draws.
        win_rate: Win rate as a fraction (0.0 to 1.0).
        avg_point_margin: Average point difference (positive = agent ahead).
        avg_game_length: Average number of turns per game.
        confidence_interval_95: Wilson score 95% confidence interval for win rate.
    """

    opponent_name: str
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_point_margin: float
    avg_game_length: float
    confidence_interval_95: tuple[float, float]


def create_opponent(name: str) -> Agent:
    """Create opponent agent by name.

    Supported names:
    - "random": RandomAgent with default seed
    - "heuristic": HeuristicAgent with default settings
    - "aggressive": Aggressive heuristic (high bar weight, aggression=0.8)
    - "defensive": Defensive heuristic (low aggression=0.2)
    - "queue": Queue controller heuristic (prioritizes queue front)
    - "skunk": Skunk specialist heuristic
    - "noisy": Noisy/human-like heuristic (bounded rationality)
    - "online": Reactive counter-play agent (OnlineStrategies)
    - "outcome_heuristic": Forward simulation with hand-tuned weights
    - "distilled_outcome": Forward simulation with PPO-extracted weights
    - "mcts-N": MCTSAgent with N iterations (e.g., mcts-100, mcts-500)

    Args:
        name: The opponent name identifier.

    Returns:
        An Agent instance.

    Raises:
        ValueError: If the opponent name is not recognized.
    """
    from _02_agents.heuristic import HeuristicAgent, HeuristicConfig, OnlineStrategies
    from _02_agents.mcts import MCTSAgent
    from _02_agents.random_agent import RandomAgent

    name_lower = name.lower().strip()

    if name_lower == "random":
        return RandomAgent(seed=None)
    elif name_lower == "heuristic":
        return HeuristicAgent(seed=None)
    elif name_lower in ("aggressive", "heuristic-aggressive"):
        config = HeuristicConfig(bar_weight=3.0, aggression=0.8)
        return HeuristicAgent(config=config, seed=None)
    elif name_lower in ("defensive", "heuristic-defensive"):
        config = HeuristicConfig(bar_weight=1.0, aggression=0.2)
        return HeuristicAgent(config=config, seed=None)
    elif name_lower in ("queue", "heuristic-queue"):
        config = HeuristicConfig(queue_front_weight=2.0)
        return HeuristicAgent(config=config, seed=None)
    elif name_lower in ("skunk", "heuristic-skunk"):
        config = HeuristicConfig(species_weights={"skunk": 2.0})
        return HeuristicAgent(config=config, seed=None)
    elif name_lower in ("noisy", "heuristic-noisy"):
        config = HeuristicConfig(noise_epsilon=0.15)
        return HeuristicAgent(config=config, seed=None)
    elif name_lower in ("online", "online-strategies"):
        return OnlineStrategies(seed=None)
    elif name_lower in ("outcome_heuristic", "outcome-heuristic"):
        from _02_agents.outcome_heuristic import OutcomeHeuristic

        return OutcomeHeuristic()
    elif name_lower in ("distilled_outcome", "distilled-outcome"):
        from _02_agents.outcome_heuristic import DistilledOutcomeHeuristic

        return DistilledOutcomeHeuristic()
    elif name_lower.startswith("mcts-"):
        from _02_agents.neural.utils import get_device, load_network_from_checkpoint

        try:
            num_simulations = int(name_lower.split("-")[1])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid MCTS specification: {name}. Use 'mcts-N' where N is simulations.") from e

        # Load network from PPO checkpoint for meaningful MCTS search
        from pathlib import Path

        ppo_checkpoint = Path("checkpoints/v2/iter_000199.pt")
        if ppo_checkpoint.exists():
            network, _config, _step = load_network_from_checkpoint(ppo_checkpoint, device=get_device())
        else:
            # Fallback to default config with random weights
            from _02_agents.neural.network import BeastyBarNetwork
            from _02_agents.neural.utils import default_config

            network = BeastyBarNetwork(default_config())
            network = network.to(get_device())
        network.eval()

        return MCTSAgent(
            network=network,
            num_simulations=num_simulations,
            temperature=0.1,  # Near-greedy for evaluation
        )
    elif name_lower == "self" or name_lower == "ppo":
        # Self-play: load the same PPO model as opponent
        from _02_agents.neural.agent import load_neural_agent

        checkpoint = "checkpoints/v2/iter_000199.pt"
        return load_neural_agent(checkpoint, mode="greedy", device="auto")
    elif name_lower.startswith("neural:") or name_lower.startswith("ppo:"):
        # Load neural model from arbitrary path
        # Format: "neural:path/to/checkpoint.pt" or "ppo:path/to/checkpoint.pt"
        from pathlib import Path

        from _02_agents.neural.agent import load_neural_agent

        checkpoint_path = name.split(":", 1)[1]
        if not Path(checkpoint_path).exists():
            raise ValueError(f"Neural opponent checkpoint not found: {checkpoint_path}")
        return load_neural_agent(checkpoint_path, mode="greedy", device="auto")
    else:
        raise ValueError(
            f"Unknown opponent: {name}. "
            f"Available: random, heuristic, aggressive, defensive, queue, skunk, noisy, online, "
            f"outcome_heuristic, distilled_outcome, mcts-100, mcts-500, mcts-1000, self, "
            f"neural:<path>, ppo:<path>"
        )


def wilson_confidence_interval(
    wins: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for win rate.

    The Wilson score interval provides better coverage than the normal
    approximation, especially for extreme probabilities or small samples.

    Args:
        wins: Number of successes (wins).
        total: Total number of trials (games).
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound) for the win rate.
    """
    if total == 0:
        return (0.0, 1.0)

    # Z-score for confidence level
    # For 95% CI, z = 1.96; for 99% CI, z = 2.576
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    p_hat = wins / total
    n = total

    denominator = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)

    lower = (center - spread) / denominator
    upper = (center + spread) / denominator

    # Clamp to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return (lower, upper)


def is_significantly_better(
    result1: EvaluationResult,
    result2: EvaluationResult,
    significance: float = 0.05,
) -> bool:
    """Test if result1 is significantly better than result2.

    Uses a two-proportion z-test to determine if the win rate
    difference is statistically significant.

    Args:
        result1: First evaluation result.
        result2: Second evaluation result.
        significance: Significance level (default 0.05 for 95% confidence).

    Returns:
        True if result1's win rate is significantly higher than result2's.
    """
    n1 = result1.games_played
    n2 = result2.games_played

    if n1 == 0 or n2 == 0:
        return False

    p1 = result1.win_rate
    p2 = result2.win_rate

    # Pooled proportion
    p_pool = (result1.wins + result2.wins) / (n1 + n2)

    # Standard error of the difference
    if p_pool == 0 or p_pool == 1:
        return p1 > p2

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        return p1 > p2

    # Z statistic
    z = (p1 - p2) / se

    # Critical values for one-tailed test
    z_critical = {0.01: 2.326, 0.05: 1.645, 0.10: 1.282}
    z_crit = z_critical.get(significance, 1.645)

    return z > z_crit


def _play_evaluation_game(
    agent: Agent,
    opponent: Agent,
    seed: int,
    agent_is_player_0: bool,
) -> tuple[int, int, int, int]:
    """Play a single evaluation game.

    Args:
        agent: The agent being evaluated.
        opponent: The opponent agent.
        seed: Random seed for the game.
        agent_is_player_0: If True, agent plays as player 0.

    Returns:
        Tuple of (agent_score, opponent_score, game_length, agent_won).
        agent_won is 1 for win, 0 for loss, -1 for draw.
    """
    if agent_is_player_0:
        agent_a, agent_b = agent, opponent
    else:
        agent_a, agent_b = opponent, agent

    config = simulate.SimulationConfig(
        seed=seed,
        games=1,
        agent_a=agent_a,
        agent_b=agent_b,
    )

    for final_state in simulate.run(config):
        scores = engine.score(final_state)
        game_length = final_state.turn

        if agent_is_player_0:
            agent_score, opponent_score = scores[0], scores[1]
        else:
            agent_score, opponent_score = scores[1], scores[0]

        if agent_score > opponent_score:
            outcome = 1
        elif agent_score < opponent_score:
            outcome = 0
        else:
            outcome = -1

        return (agent_score, opponent_score, game_length, outcome)

    # Should not reach here
    return (0, 0, 0, -1)


def evaluate_agent(
    agent: Agent,
    config: EvaluationConfig,
    device: Any | None = None,  # torch.device - kept as Any to avoid import
) -> list[EvaluationResult]:
    """Evaluate agent against all configured opponents.

    Args:
        agent: The agent to evaluate.
        config: Evaluation configuration.
        device: Optional device for neural agents (unused for non-neural agents).

    Returns:
        List of EvaluationResult, one per opponent.
    """
    results: list[EvaluationResult] = []

    for opponent_name in config.opponents:
        opponent = create_opponent(opponent_name)

        wins = 0
        losses = 0
        draws = 0
        total_margin = 0.0
        total_game_length = 0

        # Determine number of games and seeds
        total_games = config.games_per_opponent
        if config.seeds is not None:
            seeds = config.seeds[:total_games]
            # Extend seeds if not enough provided
            while len(seeds) < total_games:
                seeds.append(seeds[-1] + len(seeds) if seeds else 0)
        else:
            seeds = list(range(total_games))

        # Play games
        for game_idx, seed in enumerate(seeds):
            # Determine which side to play
            agent_is_player_0 = game_idx % 2 == 0 if config.play_both_sides else True

            agent_score, opp_score, game_length, outcome = _play_evaluation_game(
                agent=agent,
                opponent=opponent,
                seed=seed,
                agent_is_player_0=agent_is_player_0,
            )

            if outcome == 1:
                wins += 1
            elif outcome == 0:
                losses += 1
            else:
                draws += 1

            total_margin += agent_score - opp_score
            total_game_length += game_length

        # Compute statistics
        games_played = wins + losses + draws
        win_rate = wins / games_played if games_played > 0 else 0.0
        avg_margin = total_margin / games_played if games_played > 0 else 0.0
        avg_length = total_game_length / games_played if games_played > 0 else 0.0
        ci = wilson_confidence_interval(wins, games_played)

        results.append(
            EvaluationResult(
                opponent_name=opponent_name,
                games_played=games_played,
                wins=wins,
                losses=losses,
                draws=draws,
                win_rate=win_rate,
                avg_point_margin=avg_margin,
                avg_game_length=avg_length,
                confidence_interval_95=ci,
            )
        )

    return results


def create_evaluation_report(
    results: list[EvaluationResult],
    iteration: int | None = None,
) -> str:
    """Create formatted evaluation summary.

    Args:
        results: List of evaluation results.
        iteration: Optional training iteration number.

    Returns:
        Formatted string report.
    """
    lines = []

    if iteration is not None:
        lines.append(f"Evaluation Report (Iteration {iteration})")
    else:
        lines.append("Evaluation Report")
    lines.append("=" * 80)

    # Header
    header = (
        f"{'Opponent':<15} {'Games':>8} {'W-L-D':>12} "
        f"{'Win%':>8} {'CI (95%)':>16} {'Margin':>8} {'Length':>8}"
    )
    lines.append(header)
    lines.append("-" * 80)

    # Results
    for r in results:
        record = f"{r.wins}-{r.losses}-{r.draws}"
        ci_str = f"[{r.confidence_interval_95[0]:.2f}, {r.confidence_interval_95[1]:.2f}]"
        lines.append(
            f"{r.opponent_name:<15} {r.games_played:>8} {record:>12} "
            f"{r.win_rate * 100:>7.1f}% {ci_str:>16} {r.avg_point_margin:>+8.2f} {r.avg_game_length:>8.1f}"
        )

    lines.append("=" * 80)

    # Summary statistics
    total_games = sum(r.games_played for r in results)
    total_wins = sum(r.wins for r in results)
    total_losses = sum(r.losses for r in results)
    total_draws = sum(r.draws for r in results)
    overall_wr = total_wins / total_games if total_games > 0 else 0.0

    lines.append(
        f"Overall: {total_games} games, {total_wins}W-{total_losses}L-{total_draws}D "
        f"({overall_wr * 100:.1f}% win rate)"
    )

    return "\n".join(lines)


def log_evaluation_results(
    tracker: ExperimentTracker,
    results: list[EvaluationResult],
    step: int,
) -> None:
    """Log evaluation results to experiment tracker.

    Args:
        tracker: The experiment tracker to use.
        results: List of evaluation results.
        step: Training step number.
    """
    metrics: dict[str, float] = {}

    for r in results:
        prefix = f"eval/{r.opponent_name}"
        metrics[f"{prefix}/win_rate"] = r.win_rate
        metrics[f"{prefix}/games_played"] = float(r.games_played)
        metrics[f"{prefix}/avg_margin"] = r.avg_point_margin
        metrics[f"{prefix}/avg_game_length"] = r.avg_game_length
        metrics[f"{prefix}/ci_lower"] = r.confidence_interval_95[0]
        metrics[f"{prefix}/ci_upper"] = r.confidence_interval_95[1]

    # Overall metrics
    total_games = sum(r.games_played for r in results)
    total_wins = sum(r.wins for r in results)
    if total_games > 0:
        metrics["eval/overall_win_rate"] = total_wins / total_games
        metrics["eval/overall_avg_margin"] = sum(
            r.avg_point_margin * r.games_played for r in results
        ) / total_games

    tracker.log_metrics(metrics, step)


def compare_agents(
    agent1: Agent,
    agent2: Agent,
    num_games: int = 100,
    play_both_sides: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Head-to-head comparison between two agents.

    Args:
        agent1: First agent.
        agent2: Second agent.
        num_games: Number of games to play.
        play_both_sides: If True, alternate starting positions.
        seed: Base random seed.

    Returns:
        Dictionary with comparison results including:
        - wins_1, wins_2, draws: Game outcomes
        - win_rate_1, win_rate_2: Win rates
        - avg_margin_1: Average point margin for agent1
        - p_value: Approximate p-value for difference
        - significantly_different: Whether results are statistically significant
    """
    wins_1 = 0
    wins_2 = 0
    draws = 0
    total_margin_1 = 0.0
    total_game_length = 0

    for game_idx in range(num_games):
        game_seed = seed + game_idx

        agent1_is_player_0 = game_idx % 2 == 0 if play_both_sides else True

        if agent1_is_player_0:
            agent_a, agent_b = agent1, agent2
        else:
            agent_a, agent_b = agent2, agent1

        config = simulate.SimulationConfig(
            seed=game_seed,
            games=1,
            agent_a=agent_a,
            agent_b=agent_b,
        )

        for final_state in simulate.run(config):
            scores = engine.score(final_state)
            game_length = final_state.turn

            if agent1_is_player_0:
                score_1, score_2 = scores[0], scores[1]
            else:
                score_1, score_2 = scores[1], scores[0]

            if score_1 > score_2:
                wins_1 += 1
            elif score_1 < score_2:
                wins_2 += 1
            else:
                draws += 1

            total_margin_1 += score_1 - score_2
            total_game_length += game_length

    total = wins_1 + wins_2 + draws
    win_rate_1 = wins_1 / total if total > 0 else 0.0
    win_rate_2 = wins_2 / total if total > 0 else 0.0
    avg_margin_1 = total_margin_1 / total if total > 0 else 0.0
    avg_length = total_game_length / total if total > 0 else 0.0

    # Confidence intervals
    ci_1 = wilson_confidence_interval(wins_1, total)
    ci_2 = wilson_confidence_interval(wins_2, total)

    # Statistical test (binomial test approximation)
    # Null hypothesis: win_rate_1 = 0.5 (excluding draws)
    non_draw_games = wins_1 + wins_2
    if non_draw_games > 0:
        p_hat = wins_1 / non_draw_games
        se = math.sqrt(0.25 / non_draw_games)  # SE under null of p=0.5
        z = (p_hat - 0.5) / se if se > 0 else 0.0
        # Two-tailed p-value (approximation)
        p_value = 2 * (1 - _normal_cdf(abs(z)))
    else:
        p_value = 1.0

    return {
        "wins_1": wins_1,
        "wins_2": wins_2,
        "draws": draws,
        "win_rate_1": win_rate_1,
        "win_rate_2": win_rate_2,
        "confidence_interval_1": ci_1,
        "confidence_interval_2": ci_2,
        "avg_margin_1": avg_margin_1,
        "avg_game_length": avg_length,
        "p_value": p_value,
        "significantly_different": p_value < 0.05,
    }


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def run_evaluation(
    network: Any,  # BeastyBarNetwork, kept as Any to avoid import
    device: Any,  # torch.device
    tracker: ExperimentTracker,
    step: int,
    games_per_opponent: int = 50,
    opponents: list[str] | None = None,
    play_both_sides: bool = True,
    mode: str = "greedy",
) -> dict[str, float]:
    """Run evaluation against baseline opponents and log results.

    This is a convenience function that combines agent creation, evaluation,
    and logging into a single call. Used by both Trainer and MCTSTrainer.

    Args:
        network: Neural network model (BeastyBarNetwork).
        device: Torch device for inference.
        tracker: Experiment tracker for logging metrics.
        step: Training step/iteration number for logging.
        games_per_opponent: Number of games to play against each opponent.
        opponents: List of opponent names. Defaults to ["random", "heuristic"].
        play_both_sides: If True, play as both player 0 and player 1.
        mode: Agent action selection mode ("greedy" or "sample").

    Returns:
        Dictionary of evaluation metrics with keys like "eval/{opponent}/win_rate".

    Example:
        metrics = run_evaluation(
            network=self.network,
            device=self._device,
            tracker=self.tracker,
            step=self._iteration,
        )
    """
    from _02_agents.neural.agent import NeuralAgent

    if opponents is None:
        opponents = ["random", "heuristic"]

    # Create agent from network
    agent = NeuralAgent(
        model=network,
        device=device,
        mode=mode,
    )

    # Configure evaluation
    eval_config = EvaluationConfig(
        games_per_opponent=games_per_opponent,
        opponents=opponents,
        play_both_sides=play_both_sides,
    )

    # Run evaluation
    results = evaluate_agent(agent, eval_config, device=device)

    # Log results to tracker
    log_evaluation_results(tracker, results, step=step)

    # Build metrics dictionary
    metrics: dict[str, float] = {}
    for result in results:
        prefix = f"eval/{result.opponent_name}"
        metrics[f"{prefix}/win_rate"] = result.win_rate
        metrics[f"{prefix}/avg_margin"] = result.avg_point_margin
        metrics[f"{prefix}/games"] = float(result.games_played)

    return metrics


def estimate_elo(
    results: list[EvaluationResult],
    baseline_elos: dict[str, float] | None = None,
) -> float:
    """Estimate ELO rating from evaluation results.

    Uses maximum likelihood estimation to find the ELO rating
    that best explains the observed results against known opponents.

    Args:
        results: Evaluation results against various opponents.
        baseline_elos: Known ELO ratings for opponents. Defaults to:
            - random: 800
            - heuristic: 1200
            - mcts-100: 1400
            - mcts-500: 1600
            - mcts-1000: 1800

    Returns:
        Estimated ELO rating for the evaluated agent.
    """
    if baseline_elos is None:
        baseline_elos = {
            "random": 800,
            "heuristic": 1200,
            "mcts-100": 1400,
            "mcts-500": 1600,
            "mcts-1000": 1800,
        }

    # Filter results to those with known baseline ELOs
    valid_results = [r for r in results if r.opponent_name in baseline_elos]

    if not valid_results:
        return 1500.0  # Default rating

    # Use iterative MLE to estimate ELO
    # Start with weighted average based on win rates
    total_weight = 0.0
    weighted_sum = 0.0

    for r in valid_results:
        opp_elo = baseline_elos[r.opponent_name]
        # Convert win rate to expected ELO difference
        # E(A vs B) = 1 / (1 + 10^((B-A)/400))
        # If we observe win rate p, then: p = 1/(1 + 10^((opp_elo - our_elo)/400))
        # Solving: our_elo = opp_elo - 400 * log10(1/p - 1)

        if r.win_rate > 0 and r.win_rate < 1:
            implied_elo = opp_elo - 400 * math.log10(1 / r.win_rate - 1)
        elif r.win_rate >= 1.0:
            implied_elo = opp_elo + 400  # Capped estimate for 100% win rate
        else:
            implied_elo = opp_elo - 400  # Capped estimate for 0% win rate

        weight = r.games_played
        weighted_sum += implied_elo * weight
        total_weight += weight

    if total_weight == 0:
        return 1500.0

    initial_estimate = weighted_sum / total_weight

    # Refine with Newton-Raphson for MLE
    elo = initial_estimate

    for _ in range(10):  # 10 iterations usually sufficient
        gradient = 0.0
        hessian = 0.0

        for r in valid_results:
            opp_elo = baseline_elos[r.opponent_name]
            # Expected win probability
            expected = 1 / (1 + 10 ** ((opp_elo - elo) / 400))
            # Observed wins and total games
            n = r.games_played
            k = r.wins

            # Gradient of log-likelihood
            gradient += (k - n * expected) * math.log(10) / 400
            # Hessian
            hessian -= n * expected * (1 - expected) * (math.log(10) / 400) ** 2

        if abs(hessian) < 1e-10:
            break

        # Newton-Raphson update
        elo = elo - gradient / hessian

        # Clamp to reasonable range
        elo = max(100, min(3000, elo))

    return elo


__all__ = [
    "EvaluationConfig",
    "EvaluationResult",
    "compare_agents",
    "create_evaluation_report",
    "create_opponent",
    "estimate_elo",
    "evaluate_agent",
    "is_significantly_better",
    "log_evaluation_results",
    "run_evaluation",
    "wilson_confidence_interval",
]
