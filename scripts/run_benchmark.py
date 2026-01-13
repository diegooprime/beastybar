#!/usr/bin/env python3
"""Benchmark evaluation script for comparing multiple models."""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig
from _02_agents.neural.agent import NeuralAgent
from _02_agents.mcts.agent import MCTSAgent
from _03_training.evaluation import evaluate_agent, create_opponent, EvaluationConfig


def load_neural_network(checkpoint_path: str, device: str = "mps") -> BeastyBarNetwork:
    """Load neural network from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config_dict = checkpoint.get("config", {}).get("network_config", {})

    # Detect architecture from checkpoint's value_head structure
    state_dict = checkpoint["model_state_dict"]
    # If value_head.0.weight has shape [hidden_dim, hidden_dim], it's deep architecture
    # If value_head.0.weight has shape [hidden_dim//2, hidden_dim], it's simple architecture
    value_head_0_shape = state_dict.get("value_head.0.weight", torch.zeros(1)).shape
    hidden_dim = config_dict.get("hidden_dim", 256)
    deep_value_head = value_head_0_shape[0] == hidden_dim  # True if [256, 256], False if [128, 256]

    config = NetworkConfig(
        observation_dim=config_dict.get("observation_dim", 988),
        action_dim=config_dict.get("action_dim", 124),
        hidden_dim=hidden_dim,
        num_heads=config_dict.get("num_heads", 8),
        num_layers=config_dict.get("num_layers", 4),
        dropout=config_dict.get("dropout", 0.1),
        species_embedding_dim=config_dict.get("species_embedding_dim", 64),
        card_feature_dim=config_dict.get("card_feature_dim", 17),
        num_species=config_dict.get("num_species", 12),
        max_queue_length=config_dict.get("max_queue_length", 5),
        max_bar_length=config_dict.get("max_bar_length", 24),
        hand_size=config_dict.get("hand_size", 4),
        deep_value_head=deep_value_head,
    )

    network = BeastyBarNetwork(config)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(device)
    network.eval()
    return network


def get_opponents() -> dict:
    """Get all 10 opponent agents."""
    opponent_names = [
        "random",
        "heuristic",
        "aggressive",
        "defensive",
        "queue",
        "skunk",
        "noisy",
        "online",
        "outcome_heuristic",
        "distilled_outcome",
    ]
    return {name: create_opponent(name) for name in opponent_names}


def run_evaluation(agent, agent_name: str, opponent_names: list[str], games_per_opponent: int = 100):
    """Run evaluation against all opponents."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {agent_name}")
    print(f"{'='*60}")

    config = EvaluationConfig(
        games_per_opponent=games_per_opponent,
        opponents=opponent_names,
        play_both_sides=True,
    )

    eval_results = evaluate_agent(agent=agent, config=config)

    results = {}
    total_wins = 0
    total_games = 0

    for result in eval_results:
        results[result.opponent_name] = result
        total_wins += result.wins
        total_games += result.games_played

        ci_low, ci_high = result.confidence_interval_95
        print(f"  vs {result.opponent_name:20s}: {result.win_rate*100:5.1f}% "
              f"({result.wins}W/{result.losses}L/{result.draws}D) "
              f"[{ci_low*100:.1f}%-{ci_high*100:.1f}%]")

    overall_wr = total_wins / total_games if total_games > 0 else 0
    print(f"\n  Overall: {overall_wr*100:.1f}% ({total_wins}/{total_games})")

    return results


def main():
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    games_per_opponent = 100

    opponent_names = [
        "random",
        "heuristic",
        "aggressive",
        "defensive",
        "queue",
        "skunk",
        "noisy",
        "online",
        "outcome_heuristic",
        "distilled_outcome",
    ]

    print("Beasty Bar Model Benchmark")
    print("="*60)
    print(f"Games per opponent: {games_per_opponent}")
    print(f"Device: {device}")
    print(f"Opponents: {', '.join(opponent_names)}")

    # Model 1: 92max with 10 MCTS simulations
    print("\nLoading 92max.pt with MCTS (10 sims)...")
    network_92max = load_neural_network("checkpoints/92max.pt", device)
    agent_92max = MCTSAgent(
        network=network_92max,
        num_simulations=10,
        temperature=0.1,  # Near-greedy
        device=device,
        name="92max+MCTS(10)",
    )

    # Model 2: final.pt (greedy)
    print("Loading v4/final.pt...")
    network_final = load_neural_network("checkpoints/v4/final.pt", device)
    agent_final = NeuralAgent(
        model=network_final,
        device=torch.device(device),
        mode="greedy",
    )

    # Model 3: iter_600_final.pt (greedy)
    print("Loading v4/iter_600_final.pt...")
    network_600 = load_neural_network("checkpoints/v4/iter_600_final.pt", device)
    agent_600 = NeuralAgent(
        model=network_600,
        device=torch.device(device),
        mode="greedy",
    )

    # Run evaluations
    results_92max = run_evaluation(agent_92max, "92max + MCTS(10 sims)", opponent_names, games_per_opponent)
    results_final = run_evaluation(agent_final, "v4/final.pt (greedy)", opponent_names, games_per_opponent)
    results_600 = run_evaluation(agent_600, "v4/iter_600_final.pt (greedy)", opponent_names, games_per_opponent)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Opponent':<20} {'92max+MCTS(10)':>15} {'final.pt':>15} {'iter_600':>15}")
    print("-"*80)

    for opp_name in opponent_names:
        wr_92 = results_92max[opp_name].win_rate * 100
        wr_final = results_final[opp_name].win_rate * 100
        wr_600 = results_600[opp_name].win_rate * 100
        print(f"{opp_name:<20} {wr_92:>14.1f}% {wr_final:>14.1f}% {wr_600:>14.1f}%")

    print("-"*80)

    # Calculate overall
    total_92 = sum(r.wins for r in results_92max.values())
    total_final = sum(r.wins for r in results_final.values())
    total_600 = sum(r.wins for r in results_600.values())
    total_games = games_per_opponent * len(opponent_names)

    print(f"{'OVERALL':<20} {total_92/total_games*100:>14.1f}% {total_final/total_games*100:>14.1f}% {total_600/total_games*100:>14.1f}%")


if __name__ == "__main__":
    main()
