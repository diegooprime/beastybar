"""Behavioral pattern analysis for trained Beasty Bar neural network model.

This script analyzes the model's decision-making patterns across various
game situations to extract strategic insights.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from _01_simulator import engine, rules, state
from _01_simulator.action_space import (
    ACTION_DIM,
    action_index,
    canonical_actions,
    index_to_action,
    legal_action_mask_tensor,
    legal_action_space,
)
from _01_simulator.observations import (
    OBSERVATION_DIM,
    species_name,
    state_to_tensor,
    _INDEX_TO_SPECIES,
    _SPECIES_INDEX,
)
from _02_agents.neural.agent import load_neural_agent


# Species name mapping (sorted order used in observations.py)
SPECIES_NAMES = sorted([s for s in rules.SPECIES.keys() if s != "unknown"])
# Map from species index in one-hot encoding to species name
# The one-hot excludes 'unknown', so we need to account for that
UNKNOWN_IDX = _SPECIES_INDEX["unknown"]


def get_species_name_from_idx(idx: int) -> str:
    """Convert one-hot index (excluding unknown) to species name."""
    # In observations.py, unknown is excluded from one-hot
    # So we need to map the index back to the species name
    for species, species_idx in _SPECIES_INDEX.items():
        if species == "unknown":
            continue
        # The adjusted_id is species_idx if species_idx < UNKNOWN_IDX else species_idx - 1
        adjusted_id = species_idx if species_idx < UNKNOWN_IDX else species_idx - 1
        if adjusted_id == idx:
            return species
    return "unknown"


# Build the mapping once
IDX_TO_SPECIES = {i: get_species_name_from_idx(i) for i in range(12)}


def generate_random_game_states(n_states: int, seed: int = 42) -> list[tuple[state.State, int]]:
    """Generate random game states by playing random moves.

    Returns list of (state, turn_number) tuples.
    """
    rng = random.Random(seed)
    states = []

    for game_idx in range(n_states * 2):  # Generate more games to get diverse states
        game_seed = rng.randint(0, 1_000_000)
        game_state = state.initial_state(game_seed, starting_player=rng.randint(0, 1))

        # Play 0 to 15 random turns
        turns_to_play = rng.randint(0, 15)

        for turn in range(turns_to_play):
            if engine.is_terminal(game_state):
                break

            player = game_state.active_player
            legal = list(engine.legal_actions(game_state, player))
            if not legal:
                break

            action = rng.choice(legal)
            game_state = engine.step(game_state, action)

        if not engine.is_terminal(game_state):
            states.append((game_state, game_state.turn))

        if len(states) >= n_states:
            break

    return states[:n_states]


def get_hand_species(game_state: state.State, player: int) -> list[str]:
    """Get list of species in a player's hand."""
    return [card.species for card in game_state.players[player].hand]


def get_queue_species(game_state: state.State) -> list[str]:
    """Get list of species in the queue."""
    return [card.species for card in game_state.zones.queue]


def has_threatening_cards(queue_species: list[str]) -> dict[str, bool]:
    """Check if queue contains specific threatening cards."""
    threats = ["lion", "hippo", "crocodile"]
    return {threat: threat in queue_species for threat in threats}


def analyze_model_preferences(
    model_path: str,
    n_states: int = 500,
    device: str = "cpu",
) -> dict[str, Any]:
    """Analyze model's card play preferences across game states."""

    # Load model
    agent = load_neural_agent(model_path, mode="greedy", device=device)

    # Generate random game states
    print(f"Generating {n_states} random game states...")
    states_with_turns = generate_random_game_states(n_states, seed=42)
    print(f"Generated {len(states_with_turns)} valid game states")

    # Analysis containers
    # Card preferences by turn phase
    early_game_prefs = defaultdict(list)  # turns 1-3
    mid_game_prefs = defaultdict(list)    # turns 4-7
    late_game_prefs = defaultdict(list)   # turns 8+

    # Defensive play analysis - what's played when threats are present
    defensive_prefs = {
        "lion_present": defaultdict(list),
        "hippo_present": defaultdict(list),
        "crocodile_present": defaultdict(list),
        "any_threat_present": defaultdict(list),
        "no_threats": defaultdict(list),
    }

    # Queue manipulation usage
    manipulation_usage = {
        "seal": {"early": 0, "mid": 0, "late": 0, "total": 0},
        "snake": {"early": 0, "mid": 0, "late": 0, "total": 0},
        "parrot": {"early": 0, "mid": 0, "late": 0, "total": 0},
    }

    # Monkey pair tracking
    monkey_plays = {"with_monkey_in_queue": 0, "without_monkey_in_queue": 0}

    # Overall card play frequency
    card_play_counts = defaultdict(int)
    card_availability_counts = defaultdict(int)

    # Track policy distributions for detailed analysis
    policy_by_species = defaultdict(list)

    # Analyze each state
    for game_state, turn in states_with_turns:
        player = game_state.active_player
        hand_species = get_hand_species(game_state, player)
        queue_species = get_queue_species(game_state)
        threats = has_threatening_cards(queue_species)

        # Get model's policy
        policy_probs, mask, value = agent.get_policy_and_value(game_state)
        policy_probs = policy_probs.cpu().numpy()
        mask = mask.cpu().numpy()

        # Get legal actions
        action_space = legal_action_space(game_state, player)

        # Map probabilities to species being played
        species_probs = defaultdict(float)
        for idx in action_space.legal_indices:
            action = index_to_action(idx)
            card_in_hand = game_state.players[player].hand[action.hand_index]
            species = card_in_hand.species
            species_probs[species] += policy_probs[idx]

        # Normalize
        total_prob = sum(species_probs.values())
        if total_prob > 0:
            species_probs = {k: v / total_prob for k, v in species_probs.items()}

        # Record availability
        for species in hand_species:
            card_availability_counts[species] += 1

        # Find the highest probability species
        if species_probs:
            best_species = max(species_probs.items(), key=lambda x: x[1])[0]
            card_play_counts[best_species] += 1

        # Store by turn phase
        if turn <= 3:
            phase = "early"
            for species, prob in species_probs.items():
                early_game_prefs[species].append(prob)
        elif turn <= 7:
            phase = "mid"
            for species, prob in species_probs.items():
                mid_game_prefs[species].append(prob)
        else:
            phase = "late"
            for species, prob in species_probs.items():
                late_game_prefs[species].append(prob)

        # Defensive analysis
        any_threat = threats["lion"] or threats["hippo"] or threats["crocodile"]
        if any_threat:
            for species, prob in species_probs.items():
                defensive_prefs["any_threat_present"][species].append(prob)
            if threats["lion"]:
                for species, prob in species_probs.items():
                    defensive_prefs["lion_present"][species].append(prob)
            if threats["hippo"]:
                for species, prob in species_probs.items():
                    defensive_prefs["hippo_present"][species].append(prob)
            if threats["crocodile"]:
                for species, prob in species_probs.items():
                    defensive_prefs["crocodile_present"][species].append(prob)
        else:
            for species, prob in species_probs.items():
                defensive_prefs["no_threats"][species].append(prob)

        # Queue manipulation tracking
        for manip_species in ["seal", "snake", "parrot"]:
            if manip_species in species_probs:
                prob = species_probs[manip_species]
                if prob > 0.3:  # Significant preference
                    manipulation_usage[manip_species][phase] += 1
                    manipulation_usage[manip_species]["total"] += 1

        # Monkey pair analysis
        if "monkey" in species_probs:
            if "monkey" in queue_species:
                monkey_plays["with_monkey_in_queue"] += 1
            else:
                monkey_plays["without_monkey_in_queue"] += 1

        # Track detailed policy by species in hand
        for species, prob in species_probs.items():
            policy_by_species[species].append(prob)

    # Compile results
    def compute_phase_stats(prefs: dict) -> dict:
        stats = {}
        for species, probs in prefs.items():
            if probs:
                stats[species] = {
                    "mean": float(np.mean(probs)),
                    "std": float(np.std(probs)),
                    "count": len(probs),
                }
        return stats

    results = {
        "n_states_analyzed": len(states_with_turns),
        "early_game_preferences": compute_phase_stats(early_game_prefs),
        "mid_game_preferences": compute_phase_stats(mid_game_prefs),
        "late_game_preferences": compute_phase_stats(late_game_prefs),
        "defensive_analysis": {
            context: compute_phase_stats(prefs)
            for context, prefs in defensive_prefs.items()
        },
        "queue_manipulation_usage": manipulation_usage,
        "monkey_pair_plays": monkey_plays,
        "overall_play_frequency": dict(card_play_counts),
        "card_availability": dict(card_availability_counts),
        "policy_by_species": {
            species: {
                "mean": float(np.mean(probs)),
                "std": float(np.std(probs)),
                "max": float(np.max(probs)),
                "min": float(np.min(probs)),
                "count": len(probs),
            }
            for species, probs in policy_by_species.items()
            if probs
        },
    }

    return results


def compute_play_rate(results: dict) -> dict[str, float]:
    """Compute play rate = plays / availability for each species."""
    play_counts = results["overall_play_frequency"]
    availability = results["card_availability"]

    rates = {}
    for species in availability:
        if availability[species] > 0:
            rate = play_counts.get(species, 0) / availability[species]
            rates[species] = rate

    return rates


def rank_by_phase(results: dict, phase: str) -> list[tuple[str, float]]:
    """Rank species by mean preference in a given phase."""
    prefs = results.get(f"{phase}_game_preferences", {})
    ranked = [(s, p["mean"]) for s, p in prefs.items()]
    ranked.sort(key=lambda x: -x[1])
    return ranked


def main():
    model_path = PROJECT_ROOT / "checkpoints" / "v4" / "final.pt"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        sys.exit(1)

    print(f"Analyzing model: {model_path}")
    print("=" * 60)

    results = analyze_model_preferences(str(model_path), n_states=500)

    # Save raw results
    output_path = PROJECT_ROOT / "_05_analysis" / "behavioral_patterns_raw.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nStates analyzed: {results['n_states_analyzed']}")

    # Play rates
    play_rates = compute_play_rate(results)
    print("\n--- Overall Play Rates (plays / availability) ---")
    sorted_rates = sorted(play_rates.items(), key=lambda x: -x[1])
    for species, rate in sorted_rates:
        print(f"  {species:12s}: {rate:.3f}")

    # Phase preferences
    for phase in ["early", "mid", "late"]:
        print(f"\n--- {phase.upper()} Game Preferences ---")
        ranked = rank_by_phase(results, phase)
        for species, mean_prob in ranked[:6]:
            print(f"  {species:12s}: {mean_prob:.3f}")

    # Queue manipulation
    print("\n--- Queue Manipulation Usage (high preference counts) ---")
    for species, usage in results["queue_manipulation_usage"].items():
        print(f"  {species}: early={usage['early']}, mid={usage['mid']}, late={usage['late']}, total={usage['total']}")

    # Defensive plays
    print("\n--- Defensive Preferences (when threats present) ---")
    threat_prefs = results["defensive_analysis"].get("any_threat_present", {})
    ranked_threat = sorted(threat_prefs.items(), key=lambda x: -x[1]["mean"])
    for species, stats in ranked_threat[:6]:
        print(f"  {species:12s}: {stats['mean']:.3f} (n={stats['count']})")

    # Monkey analysis
    print("\n--- Monkey Pair Analysis ---")
    mp = results["monkey_pair_plays"]
    print(f"  Monkey played with monkey in queue: {mp['with_monkey_in_queue']}")
    print(f"  Monkey played without monkey in queue: {mp['without_monkey_in_queue']}")

    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
