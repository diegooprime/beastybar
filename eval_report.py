#!/usr/bin/env python3
"""Evaluation report script for Beasty Bar PPO model."""

import torch
from _03_training.evaluation import create_opponent, wilson_confidence_interval
from _02_agents.neural import NeuralAgent
from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig
from _01_simulator import engine, simulate

# Load checkpoint
checkpoint_path = "checkpoints/v4/final.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Create network - config is nested inside 'config' dict
config = checkpoint.get("config", {})
network_config_dict = config.get("network_config", {})
network_config = NetworkConfig(**network_config_dict)
network = BeastyBarNetwork(config=network_config)
network.load_state_dict(checkpoint["model_state_dict"])
network.eval()

# Create agent
agent = NeuralAgent(model=network, temperature=0.01)  # near-greedy for eval

# All opponents to evaluate against
opponents = [
    "random",
    "heuristic",
    "aggressive",
    "defensive",
    "queue",
    "skunk",
    "noisy",
    "online",
]

print("=" * 60)
print("BEASTY BAR PPO EVALUATION REPORT")
print("Model: v4/final.pt")
print("Games per opponent: 100")
print("=" * 60)
print()

results = {}
for opp_name in opponents:
    print(f"Evaluating vs {opp_name}...", end=" ", flush=True)
    opponent = create_opponent(opp_name)

    wins = 0
    losses = 0
    draws = 0
    total_games = 100
    total_point_margin = 0
    total_game_length = 0

    for game_idx in range(total_games):
        # Alternate starting positions
        agent_is_player_0 = game_idx % 2 == 0

        if agent_is_player_0:
            agent_a, agent_b = agent, opponent
        else:
            agent_a, agent_b = opponent, agent

        sim_config = simulate.SimulationConfig(
            seed=game_idx,
            games=1,
            agent_a=agent_a,
            agent_b=agent_b,
        )

        for final_state in simulate.run(sim_config):
            scores = engine.score(final_state)
            game_length = final_state.turn

            if agent_is_player_0:
                agent_score, opponent_score = scores[0], scores[1]
            else:
                agent_score, opponent_score = scores[1], scores[0]

            total_point_margin += agent_score - opponent_score
            total_game_length += game_length

            if agent_score > opponent_score:
                wins += 1
            elif agent_score < opponent_score:
                losses += 1
            else:
                draws += 1

    win_rate = wins / total_games * 100
    avg_margin = total_point_margin / total_games
    avg_length = total_game_length / total_games
    ci_low, ci_high = wilson_confidence_interval(wins, total_games)

    results[opp_name] = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_margin": avg_margin,
        "avg_length": avg_length,
        "ci_95": (ci_low * 100, ci_high * 100),
    }
    print(f"{win_rate:.0f}% ({wins}W/{losses}L/{draws}D)")

print()
print("=" * 60)
print("DETAILED RESULTS")
print("=" * 60)
print(f"{'Opponent':<22} {'Win Rate':>10} {'95% CI':>16} {'Avg Margin':>12} {'Avg Length':>12}")
print("-" * 72)

# Sort by win rate descending
sorted_results = sorted(results.items(), key=lambda x: x[1]["win_rate"], reverse=True)
for opp, data in sorted_results:
    ci_str = f"[{data['ci_95'][0]:.1f}%, {data['ci_95'][1]:.1f}%]"
    print(
        f"{opp:<22} {data['win_rate']:>9.1f}% {ci_str:>16} {data['avg_margin']:>+11.1f} {data['avg_length']:>11.1f}"
    )

avg_win_rate = sum(r["win_rate"] for r in results.values()) / len(results)
print("-" * 72)
print(f"{'AVERAGE':<22} {avg_win_rate:>9.1f}%")
print("=" * 60)

# Summary statistics
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
total_wins = sum(r["wins"] for r in results.values())
total_losses = sum(r["losses"] for r in results.values())
total_draws = sum(r["draws"] for r in results.values())
total = total_wins + total_losses + total_draws

print(f"Total games played: {total}")
print(f"Overall record: {total_wins}W / {total_losses}L / {total_draws}D")
print(f"Overall win rate: {total_wins/total*100:.1f}%")
print()

# Performance tiers
print("Performance by tier:")
print(f"  vs Random:           {results['random']['win_rate']:.0f}%")
heuristic_variants = ['heuristic', 'aggressive', 'defensive', 'queue', 'skunk', 'noisy']
print(f"  vs Heuristic family: {sum(results[k]['win_rate'] for k in heuristic_variants)/6:.0f}% avg")
print(f"  vs Online:           {results['online']['win_rate']:.0f}%")
print("=" * 60)
