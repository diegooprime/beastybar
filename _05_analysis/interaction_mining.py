"""Species-Pair Interaction Mining.

This module mines interaction bonuses from embedding similarities and behavioral data
to extract rules like:
- Hippo-Zebra: Zebra blocks Hippo advancement
- Crocodile-Zebra: Zebra blocks Crocodile eating
- Monkey-Monkey: Pair triggers predator removal

The extracted rules can be used to improve heuristic agents by capturing
species-pair interactions that linear scoring functions miss.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import random
from collections import defaultdict
from dataclasses import dataclass, field

from _01_simulator import engine, rules, state
from _01_simulator.action_space import (
    index_to_action,
    legal_action_mask_tensor,
)
from _01_simulator.observations import state_to_tensor

# All species (sorted)
SPECIES_NAMES = sorted([s for s in rules.SPECIES.keys() if s != "unknown"])


@dataclass
class InteractionStats:
    """Statistics for a species pair interaction."""

    species_a: str
    species_b: str

    # Play rate changes when the other species is present
    play_rate_with: float = 0.0
    play_rate_without: float = 0.0
    play_rate_delta: float = 0.0

    # Win rate changes
    win_rate_with: float = 0.0
    win_rate_without: float = 0.0
    win_rate_delta: float = 0.0

    # Value changes (from value head)
    value_with: float = 0.0
    value_without: float = 0.0
    value_delta: float = 0.0

    # Sample counts
    n_with: int = 0
    n_without: int = 0

    # Derived flags
    is_synergy: bool = False  # Positive interaction
    is_counter: bool = False  # Negative interaction

    def compute_deltas(self) -> None:
        """Compute delta values and flags."""
        if self.n_without > 0 and self.n_with > 0:
            self.play_rate_delta = self.play_rate_with - self.play_rate_without
            self.win_rate_delta = self.win_rate_with - self.win_rate_without
            self.value_delta = self.value_with - self.value_without

            # Synergy if play rate increases significantly
            self.is_synergy = self.play_rate_delta > 0.1 or self.value_delta > 0.05
            # Counter if play rate decreases significantly
            self.is_counter = self.play_rate_delta < -0.1 or self.value_delta < -0.05


@dataclass
class SpeciesInteractionMatrix:
    """Matrix of all species pair interactions."""

    # Mapping from (species_a, species_b) -> InteractionStats
    interactions: dict[tuple[str, str], InteractionStats] = field(default_factory=dict)

    # Raw behavioral data
    play_counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    total_opportunities: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    # Embedding similarities (if available)
    embedding_similarities: dict[tuple[str, str], float] = field(default_factory=dict)

    def add_interaction(
        self,
        played_species: str,
        queue_species: set[str],
        opp_queue_species: set[str],
    ) -> None:
        """Record a play decision for interaction mining.

        Args:
            played_species: Species that was played
            queue_species: All species currently in queue
            opp_queue_species: Opponent's species in queue
        """
        for other in SPECIES_NAMES:
            key = (played_species, other)
            if other in queue_species or other in opp_queue_species:
                self.play_counts[key] += 1
            self.total_opportunities[key] += 1

    def compute_all_interactions(self) -> None:
        """Compute interaction stats for all species pairs."""
        for a in SPECIES_NAMES:
            for b in SPECIES_NAMES:
                key = (a, b)
                n_with = self.play_counts.get(key, 0)
                n_total = self.total_opportunities.get(key, 0)
                n_without = n_total - n_with

                stats = InteractionStats(
                    species_a=a,
                    species_b=b,
                    n_with=n_with,
                    n_without=n_without,
                )

                if n_with > 0 and n_without > 0:
                    # Estimate play rates (will be refined with actual data)
                    stats.play_rate_with = n_with / max(n_with + n_without, 1)
                    stats.play_rate_without = n_without / max(n_with + n_without, 1)
                    stats.compute_deltas()

                self.interactions[key] = stats

    def get_synergies(self, min_samples: int = 100) -> list[InteractionStats]:
        """Get all synergistic species pairs."""
        return [
            stats for stats in self.interactions.values()
            if stats.is_synergy and stats.n_with + stats.n_without >= min_samples
        ]

    def get_counters(self, min_samples: int = 100) -> list[InteractionStats]:
        """Get all counter species pairs."""
        return [
            stats for stats in self.interactions.values()
            if stats.is_counter and stats.n_with + stats.n_without >= min_samples
        ]


def mine_embedding_similarities(
    model_path: str,
    device: str = "cpu",
) -> dict[tuple[str, str], float]:
    """Extract embedding similarities from the trained model.

    Computes cosine similarity between species embeddings in the model's
    learned representation space.

    Args:
        model_path: Path to trained PPO model.
        device: Device to run model on.

    Returns:
        Dictionary mapping species pairs to cosine similarities.
    """
    try:
        import torch

        from _02_agents.neural.agent import load_neural_agent
    except ImportError as e:
        print(f"Cannot load model for embedding extraction: {e}")
        return {}

    try:
        agent = load_neural_agent(model_path, device=device)
        model = agent.model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return {}

    similarities: dict[tuple[str, str], float] = {}

    # Try to extract species embeddings from model
    # This depends on model architecture - adjust based on actual model
    try:
        # Look for embedding layer
        if hasattr(model, "card_embedding"):
            embeddings = model.card_embedding.weight.data.cpu().numpy()
        elif hasattr(model, "species_embedding"):
            embeddings = model.species_embedding.weight.data.cpu().numpy()
        else:
            print("Could not find embedding layer in model")
            return {}

        # Compute pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(embeddings)

        for i, a in enumerate(SPECIES_NAMES):
            for j, b in enumerate(SPECIES_NAMES):
                similarities[(a, b)] = float(sim_matrix[i, j])

    except Exception as e:
        print(f"Failed to extract embeddings: {e}")
        return {}

    return similarities


def mine_behavioral_interactions(
    model_path: str,
    n_games: int = 10000,
    device: str = "cpu",
) -> SpeciesInteractionMatrix:
    """Mine species interactions from PPO model behavior.

    Plays games with the PPO model and tracks which species it plays
    when different opponent species are present in the queue.

    Args:
        model_path: Path to trained PPO model.
        n_games: Number of games to simulate.
        device: Device to run model on.

    Returns:
        SpeciesInteractionMatrix with behavioral data.
    """
    try:
        import torch

        from _02_agents.neural.agent import load_neural_agent
        from _02_agents.neural.utils import greedy_action
    except ImportError as e:
        raise ImportError(f"Required dependencies not available: {e}")

    agent = load_neural_agent(model_path, device=device)
    agent.model.eval()

    matrix = SpeciesInteractionMatrix()
    rng = random.Random(42)

    for game_idx in range(n_games):
        seed = rng.randint(0, 1_000_000)
        game_state = state.initial_state(seed)

        while not engine.is_terminal(game_state):
            player = game_state.active_player
            legal = list(engine.legal_actions(game_state, player))
            if not legal:
                break

            # Get PPO choice
            obs_tensor = state_to_tensor(game_state, player)
            mask = legal_action_mask_tensor(game_state, player)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_tensor).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                logits, _ = agent.model(obs_t, mask_t)
                chosen_idx = greedy_action(logits[0].cpu().numpy(), mask)

            chosen_action = index_to_action(chosen_idx)
            hand = game_state.players[player].hand
            if chosen_action.hand_index >= len(hand):
                break

            played_card = hand[chosen_action.hand_index]
            opponent = 1 - player

            # Record interaction
            queue_species = {c.species for c in game_state.zones.queue}
            opp_queue_species = {
                c.species for c in game_state.zones.queue
                if c.owner == opponent
            }

            matrix.add_interaction(
                played_species=played_card.species,
                queue_species=queue_species,
                opp_queue_species=opp_queue_species,
            )

            game_state = engine.step(game_state, chosen_action)

        if (game_idx + 1) % 1000 == 0:
            print(f"Processed {game_idx + 1}/{n_games} games...")

    matrix.compute_all_interactions()
    return matrix


def extract_interaction_rules(
    matrix: SpeciesInteractionMatrix,
    min_delta: float = 0.05,
    min_samples: int = 100,
) -> dict[str, list[tuple[str, str, float]]]:
    """Extract interpretable interaction rules from mined data.

    Args:
        matrix: Mined interaction matrix.
        min_delta: Minimum delta to consider significant.
        min_samples: Minimum samples for statistical reliability.

    Returns:
        Dictionary with 'synergies', 'counters', and 'neutral' lists.
    """
    rules_dict = {
        "synergies": [],
        "counters": [],
        "neutral": [],
    }

    for (a, b), stats in matrix.interactions.items():
        if a == b:
            continue
        if stats.n_with + stats.n_without < min_samples:
            continue

        delta = stats.play_rate_delta
        if abs(delta) < min_delta:
            rules_dict["neutral"].append((a, b, delta))
        elif delta > 0:
            rules_dict["synergies"].append((a, b, delta))
        else:
            rules_dict["counters"].append((a, b, delta))

    # Sort by magnitude
    rules_dict["synergies"].sort(key=lambda x: -x[2])
    rules_dict["counters"].sort(key=lambda x: x[2])

    return rules_dict


def generate_heuristic_rules_code(
    rules_dict: dict[str, list[tuple[str, str, float]]],
    threshold: float = 0.1,
) -> str:
    """Generate Python code for heuristic rules from extracted interactions.

    Args:
        rules_dict: Extracted interaction rules.
        threshold: Minimum delta for inclusion.

    Returns:
        Python code string implementing the rules.
    """
    lines = []
    lines.append("# Auto-generated species interaction rules")
    lines.append("# Extracted from PPO model behavior\n")

    lines.append("SPECIES_SYNERGIES = {")
    for a, b, delta in rules_dict["synergies"]:
        if abs(delta) >= threshold:
            lines.append(f'    ("{a}", "{b}"): {delta:.3f},')
    lines.append("}\n")

    lines.append("SPECIES_COUNTERS = {")
    for a, b, delta in rules_dict["counters"]:
        if abs(delta) >= threshold:
            lines.append(f'    ("{a}", "{b}"): {delta:.3f},')
    lines.append("}\n")

    lines.append("""
def get_interaction_bonus(
    playing_species: str,
    queue_species: set[str],
    opp_queue_species: set[str],
) -> float:
    \"\"\"Get bonus/penalty based on species interactions.\"\"\"
    bonus = 0.0

    # Check for synergies
    for (a, b), value in SPECIES_SYNERGIES.items():
        if playing_species == a and b in queue_species:
            bonus += value

    # Check for counters
    for (a, b), value in SPECIES_COUNTERS.items():
        if playing_species == a and b in opp_queue_species:
            bonus += value  # value is negative

    return bonus
""")

    return "\n".join(lines)


def generate_interaction_report(
    matrix: SpeciesInteractionMatrix,
    rules_dict: dict[str, list[tuple[str, str, float]]],
    embedding_sims: dict[tuple[str, str], float],
    output_path: str = "_05_analysis/07_species_interactions.md",
) -> None:
    """Generate markdown report of species interactions."""

    report = []
    report.append("# Species Interaction Analysis\n")
    report.append("This report analyzes how the PPO model's play decisions change")
    report.append("based on which species are present in the queue.\n")

    # Summary statistics
    report.append("## Summary\n")
    report.append(f"- Total synergistic pairs: {len(rules_dict['synergies'])}")
    report.append(f"- Total counter pairs: {len(rules_dict['counters'])}")
    report.append(f"- Total neutral pairs: {len(rules_dict['neutral'])}\n")

    # Top synergies
    report.append("## Top Synergies\n")
    report.append("*When species B is in queue, model plays species A more often*\n")
    report.append("| Species A | Species B | Play Rate Delta |")
    report.append("|-----------|-----------|-----------------|")
    for a, b, delta in rules_dict["synergies"][:15]:
        report.append(f"| {a} | {b} | +{delta:.3f} |")

    # Top counters
    report.append("\n## Top Counters\n")
    report.append("*When species B is in queue, model plays species A less often*\n")
    report.append("| Species A | Species B | Play Rate Delta |")
    report.append("|-----------|-----------|-----------------|")
    for a, b, delta in rules_dict["counters"][:15]:
        report.append(f"| {a} | {b} | {delta:.3f} |")

    # Known game interactions
    report.append("\n## Known Game Mechanics\n")
    report.append("These interactions align with game rules:\n")

    known_interactions = [
        ("monkey", "monkey", "Monkey pair triggers heavyweight removal"),
        ("zebra", "hippo", "Zebra blocks Hippo's push ability"),
        ("zebra", "crocodile", "Zebra blocks Crocodile's eat ability"),
        ("lion", "monkey", "Lion roar sends Monkeys to That's It"),
        ("parrot", "crocodile", "Parrot can remove threatening Crocodile"),
        ("skunk", "lion", "Skunk removes high-strength cards like Lion"),
    ]

    for a, b, reason in known_interactions:
        key = (a, b)
        if key in matrix.interactions:
            stats = matrix.interactions[key]
            delta = stats.play_rate_delta
            direction = "+" if delta > 0 else ""
            report.append(f"- **{a}** → **{b}**: {direction}{delta:.3f} ({reason})")

    # Embedding similarities if available
    if embedding_sims:
        report.append("\n## Embedding Similarities\n")
        report.append("Cosine similarity between species embeddings:\n")

        # Sort by similarity
        sorted_sims = sorted(
            [(k, v) for k, v in embedding_sims.items() if k[0] < k[1]],
            key=lambda x: -x[1]
        )

        report.append("### Most Similar Pairs")
        report.append("| Species A | Species B | Similarity |")
        report.append("|-----------|-----------|------------|")
        for (a, b), sim in sorted_sims[:10]:
            report.append(f"| {a} | {b} | {sim:.3f} |")

        report.append("\n### Most Dissimilar Pairs")
        report.append("| Species A | Species B | Similarity |")
        report.append("|-----------|-----------|------------|")
        for (a, b), sim in sorted_sims[-10:]:
            report.append(f"| {a} | {b} | {sim:.3f} |")

    # Generated code
    report.append("\n## Generated Heuristic Rules\n")
    report.append("```python")
    report.append(generate_heuristic_rules_code(rules_dict))
    report.append("```\n")

    # Write report
    output = Path(project_root) / output_path
    output.write_text("\n".join(report))
    print(f"Report written to {output}")


def main():
    """Run species interaction mining."""
    import argparse

    parser = argparse.ArgumentParser(description="Mine species pair interactions")
    parser.add_argument(
        "--model",
        default="checkpoints/v4/final.pt",
        help="Path to trained PPO model",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5000,
        help="Number of games to analyze",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run model on",
    )
    args = parser.parse_args()

    print(f"Mining interactions from {args.games} games using {args.model}...")

    # Mine behavioral interactions
    matrix = mine_behavioral_interactions(args.model, args.games, args.device)

    # Extract rules
    rules_dict = extract_interaction_rules(matrix)

    print(f"\nFound {len(rules_dict['synergies'])} synergies")
    print(f"Found {len(rules_dict['counters'])} counters")

    # Try to get embedding similarities
    print("\nExtracting embedding similarities...")
    embedding_sims = mine_embedding_similarities(args.model, args.device)

    # Generate report
    print("\nGenerating report...")
    generate_interaction_report(matrix, rules_dict, embedding_sims)


if __name__ == "__main__":
    main()
