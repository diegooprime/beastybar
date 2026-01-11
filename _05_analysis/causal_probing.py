"""Causal Probing via Activation Patching.

This module implements causal discovery techniques to find which features
actually CAUSE the PPO model's decisions, not just correlate with them.

Technique: Activation Patching
1. Find state pairs where PPO makes different decisions
2. Patch specific features from state A → B
3. Measure if decision changes
4. Features that flip decisions are causal

This goes beyond correlational analysis to identify the true decision drivers.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from _01_simulator import engine, state
from _01_simulator.action_space import (
    index_to_action,
    legal_action_mask_tensor,
)
from _01_simulator.observations import (
    state_to_tensor,
)

# Feature groups for patching experiments
FEATURE_GROUPS = {
    "queue_length": list(range(0, 5)),  # Approximate - adjust based on obs encoding
    "queue_front": list(range(0, 17)),  # First card in queue (17 features)
    "queue_back": list(range(68, 85)),  # Last card position
    "own_hand": list(range(408 + 408 + 85, 408 + 408 + 85 + 68)),  # Own hand region
    "score_diff": [980, 981],  # Scalar features
    "turn": [985],
}


@dataclass
class PatchingResult:
    """Result of a single patching experiment."""

    state_a_id: int
    state_b_id: int
    original_action_b: int
    patched_action_b: int
    feature_group: str
    decision_changed: bool


@dataclass
class CausalFeatureStats:
    """Statistics for a feature group's causal impact."""

    feature_group: str
    n_experiments: int = 0
    n_decision_flips: int = 0
    flip_rate: float = 0.0

    # Breakdown by original decision type
    flips_by_species: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self) -> None:
        """Update computed stats."""
        if self.n_experiments > 0:
            self.flip_rate = self.n_decision_flips / self.n_experiments


@dataclass
class StatePair:
    """Pair of states with different PPO decisions."""

    state_a: state.State
    state_b: state.State
    obs_a: np.ndarray
    obs_b: np.ndarray
    action_a: int
    action_b: int
    species_a: str
    species_b: str


def find_decision_pairs(
    model_path: str,
    n_pairs: int = 5000,
    device: str = "cpu",
) -> list[StatePair]:
    """Find pairs of states where PPO makes different decisions.

    Args:
        model_path: Path to trained PPO model.
        n_pairs: Number of pairs to find.
        device: Device to run model on.

    Returns:
        List of StatePair objects.
    """
    try:
        import torch

        from _02_agents.neural.agent import load_neural_agent
        from _02_agents.neural.utils import greedy_action
    except ImportError as e:
        raise ImportError(f"Required dependencies not available: {e}")

    agent = load_neural_agent(model_path, device=device)
    agent.model.eval()

    pairs = []
    states_by_action: dict[str, list[tuple[state.State, np.ndarray, int]]] = defaultdict(list)
    rng = random.Random(42)

    # Collect states grouped by chosen action
    print("Collecting states...")
    collected = 0
    while collected < n_pairs * 10:  # Collect more to have good variety
        seed = rng.randint(0, 1_000_000)
        game_state = state.initial_state(seed)

        # Play random turns to get varied states
        for _ in range(rng.randint(0, 15)):
            if engine.is_terminal(game_state):
                break
            player = game_state.active_player
            legal = list(engine.legal_actions(game_state, player))
            if not legal:
                break
            game_state = engine.step(game_state, rng.choice(legal))

        if engine.is_terminal(game_state):
            continue

        player = game_state.active_player
        legal = list(engine.legal_actions(game_state, player))
        if not legal:
            continue

        # Get PPO choice
        obs_tensor = state_to_tensor(game_state, player)
        mask = legal_action_mask_tensor(game_state, player)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs_tensor).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            logits, _ = agent.model(obs_t, mask_t)
            chosen_idx = greedy_action(logits[0].cpu().numpy(), mask)

        action = index_to_action(chosen_idx)
        hand = game_state.players[player].hand
        if action.hand_index >= len(hand):
            continue

        species = hand[action.hand_index].species
        states_by_action[species].append((game_state, obs_tensor, chosen_idx))
        collected += 1

        if collected % 5000 == 0:
            print(f"  Collected {collected} states...")

    # Create pairs with different decisions
    print("Creating pairs...")
    species_list = list(states_by_action.keys())
    for _ in range(n_pairs):
        if len(species_list) < 2:
            break

        # Pick two different species
        sp_a, sp_b = rng.sample(species_list, 2)
        if not states_by_action[sp_a] or not states_by_action[sp_b]:
            continue

        state_a, obs_a, action_a = rng.choice(states_by_action[sp_a])
        state_b, obs_b, action_b = rng.choice(states_by_action[sp_b])

        pairs.append(StatePair(
            state_a=state_a,
            state_b=state_b,
            obs_a=obs_a,
            obs_b=obs_b,
            action_a=action_a,
            action_b=action_b,
            species_a=sp_a,
            species_b=sp_b,
        ))

    return pairs


def run_patching_experiment(
    model_path: str,
    pairs: list[StatePair],
    feature_groups: dict[str, list[int]],
    device: str = "cpu",
) -> dict[str, CausalFeatureStats]:
    """Run activation patching experiments.

    For each pair and each feature group:
    1. Take observation B
    2. Patch features from observation A into B
    3. Check if model's decision on patched B changes

    Args:
        model_path: Path to trained PPO model.
        pairs: State pairs with different decisions.
        feature_groups: Mapping of feature group names to indices.
        device: Device to run model on.

    Returns:
        Dictionary mapping feature groups to their causal stats.
    """
    try:
        import torch

        from _02_agents.neural.agent import load_neural_agent
        from _02_agents.neural.utils import greedy_action
    except ImportError as e:
        raise ImportError(f"Required dependencies not available: {e}")

    agent = load_neural_agent(model_path, device=device)
    agent.model.eval()

    stats = {name: CausalFeatureStats(feature_group=name) for name in feature_groups}

    for pair_idx, pair in enumerate(pairs):
        # Get legal actions for state B (patching context)
        player = pair.state_b.active_player
        mask = legal_action_mask_tensor(pair.state_b, player)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

        for group_name, indices in feature_groups.items():
            # Create patched observation: B with features from A
            obs_patched = pair.obs_b.copy()
            valid_indices = [i for i in indices if i < len(obs_patched) and i < len(pair.obs_a)]
            for i in valid_indices:
                obs_patched[i] = pair.obs_a[i]

            # Get model's decision on patched observation
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_patched).unsqueeze(0).to(device)
                logits, _ = agent.model(obs_t, mask_t)
                patched_action = greedy_action(logits[0].cpu().numpy(), mask)

            # Record result
            decision_changed = patched_action != pair.action_b
            stats[group_name].n_experiments += 1

            if decision_changed:
                stats[group_name].n_decision_flips += 1
                stats[group_name].flips_by_species[pair.species_b] += 1

        if (pair_idx + 1) % 500 == 0:
            print(f"  Processed {pair_idx + 1}/{len(pairs)} pairs...")

    # Update computed stats
    for stat in stats.values():
        stat.update()

    return stats


def compute_feature_group_indices() -> dict[str, list[int]]:
    """Compute feature group indices based on observation encoding.

    Returns indices for each logical feature group in the observation vector.
    """
    # Based on observations.py encoding:
    # Queue: 5 cards × 17 features = 85
    # Beasty Bar: 24 cards × 17 features = 408
    # That's It: 24 cards × 17 features = 408
    # Own hand: 4 cards × 17 features = 68
    # Opponent hand: 4 cards × 3 features = 12
    # Scalars: 7 features

    CARD_DIM = 17
    MASKED_CARD_DIM = 3
    QUEUE_LEN = 5
    BAR_LEN = 24
    THATS_IT_LEN = 24
    HAND_LEN = 4

    offset = 0
    groups = {}

    # Queue positions
    for i in range(QUEUE_LEN):
        start = offset + i * CARD_DIM
        groups[f"queue_pos_{i}"] = list(range(start, start + CARD_DIM))
    groups["queue_all"] = list(range(offset, offset + QUEUE_LEN * CARD_DIM))
    offset += QUEUE_LEN * CARD_DIM

    # Beasty Bar
    groups["beasty_bar"] = list(range(offset, offset + BAR_LEN * CARD_DIM))
    offset += BAR_LEN * CARD_DIM

    # That's It
    groups["thats_it"] = list(range(offset, offset + THATS_IT_LEN * CARD_DIM))
    offset += THATS_IT_LEN * CARD_DIM

    # Own hand positions
    for i in range(HAND_LEN):
        start = offset + i * CARD_DIM
        groups[f"hand_pos_{i}"] = list(range(start, start + CARD_DIM))
    groups["hand_all"] = list(range(offset, offset + HAND_LEN * CARD_DIM))
    offset += HAND_LEN * CARD_DIM

    # Opponent hand (masked)
    groups["opponent_hand"] = list(range(offset, offset + HAND_LEN * MASKED_CARD_DIM))
    offset += HAND_LEN * MASKED_CARD_DIM

    # Scalars
    groups["deck_counts"] = [offset, offset + 1]
    groups["hand_counts"] = [offset + 2, offset + 3]
    groups["is_active"] = [offset + 4]
    groups["turn"] = [offset + 5]
    groups["queue_length_scalar"] = [offset + 6]
    groups["scalars_all"] = list(range(offset, offset + 7))

    return groups


def identify_causal_features(
    stats: dict[str, CausalFeatureStats],
    threshold: float = 0.1,
) -> list[tuple[str, float]]:
    """Identify features that causally affect decisions.

    Args:
        stats: Patching experiment results.
        threshold: Minimum flip rate to consider causal.

    Returns:
        List of (feature_group, flip_rate) sorted by causal impact.
    """
    causal = [
        (name, stat.flip_rate)
        for name, stat in stats.items()
        if stat.flip_rate >= threshold and stat.n_experiments >= 100
    ]
    return sorted(causal, key=lambda x: -x[1])


def generate_causal_probing_report(
    stats: dict[str, CausalFeatureStats],
    causal_features: list[tuple[str, float]],
    output_path: str = "_05_analysis/08_causal_probing.md",
) -> None:
    """Generate markdown report of causal probing results."""

    report = []
    report.append("# Causal Feature Analysis via Activation Patching\n")
    report.append("This report identifies which features CAUSE the PPO model's decisions,")
    report.append("not just correlate with them.\n")

    report.append("## Methodology\n")
    report.append("1. Found pairs of states where PPO makes different decisions")
    report.append("2. For each feature group, patched features from state A into state B")
    report.append("3. Measured if patching changed the model's decision")
    report.append("4. High flip rate → feature is causally important\n")

    # Summary
    report.append("## Summary\n")
    total_exp = sum(s.n_experiments for s in stats.values())
    report.append(f"- Total patching experiments: {total_exp}")
    report.append(f"- Feature groups tested: {len(stats)}")
    report.append(f"- Causal features identified: {len(causal_features)}\n")

    # Causal features
    report.append("## Causal Features (sorted by impact)\n")
    report.append("*Higher flip rate = more causal impact*\n")
    report.append("| Feature Group | Flip Rate | Experiments |")
    report.append("|---------------|-----------|-------------|")
    for name, flip_rate in causal_features:
        stat = stats[name]
        report.append(f"| {name} | {flip_rate:.1%} | {stat.n_experiments} |")

    # All features
    report.append("\n## All Feature Groups\n")
    report.append("| Feature Group | Flip Rate | Flips | Experiments |")
    report.append("|---------------|-----------|-------|-------------|")
    sorted_stats = sorted(stats.items(), key=lambda x: -x[1].flip_rate)
    for name, stat in sorted_stats:
        report.append(
            f"| {name} | {stat.flip_rate:.1%} | {stat.n_decision_flips} | {stat.n_experiments} |"
        )

    # Flip breakdown by species
    report.append("\n## Decision Flips by Species\n")
    report.append("Which species' decisions are most affected by patching:\n")

    all_species_flips: dict[str, int] = defaultdict(int)
    for stat in stats.values():
        for species, count in stat.flips_by_species.items():
            all_species_flips[species] += count

    sorted_species = sorted(all_species_flips.items(), key=lambda x: -x[1])
    for species, count in sorted_species:
        report.append(f"- {species}: {count} flips")

    # Interpretation
    report.append("\n## Interpretation\n")
    if causal_features:
        top_feature = causal_features[0][0]
        report.append(f"**Most causal feature: {top_feature}**\n")
        report.append("Features with high causal impact should be prioritized")
        report.append("in heuristic agents for decision-making.\n")

        report.append("### Recommendations\n")
        for name, flip_rate in causal_features[:5]:
            if "queue" in name:
                report.append(f"- **{name}** ({flip_rate:.1%}): Queue state is critical for decisions")
            elif "hand" in name:
                report.append(f"- **{name}** ({flip_rate:.1%}): Hand composition drives choices")
            elif "score" in name or "bar" in name:
                report.append(f"- **{name}** ({flip_rate:.1%}): Score awareness affects strategy")
            else:
                report.append(f"- **{name}** ({flip_rate:.1%}): Important decision factor")

    # Write report
    output = Path(project_root) / output_path
    output.write_text("\n".join(report))
    print(f"Report written to {output}")


def main():
    """Run causal probing analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Causal probing via activation patching")
    parser.add_argument(
        "--model",
        default="checkpoints/v4/final.pt",
        help="Path to trained PPO model",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=2000,
        help="Number of state pairs to analyze",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run model on",
    )
    args = parser.parse_args()

    print(f"Finding {args.pairs} decision pairs from {args.model}...")
    pairs = find_decision_pairs(args.model, args.pairs, args.device)
    print(f"Found {len(pairs)} pairs with different decisions")

    print("\nComputing feature group indices...")
    feature_groups = compute_feature_group_indices()
    print(f"Defined {len(feature_groups)} feature groups")

    print("\nRunning patching experiments...")
    stats = run_patching_experiment(args.model, pairs, feature_groups, args.device)

    print("\nIdentifying causal features...")
    causal_features = identify_causal_features(stats)
    print(f"Found {len(causal_features)} causal features")

    print("\nTop causal features:")
    for name, flip_rate in causal_features[:5]:
        print(f"  {name}: {flip_rate:.1%}")

    print("\nGenerating report...")
    generate_causal_probing_report(stats, causal_features)


if __name__ == "__main__":
    main()
