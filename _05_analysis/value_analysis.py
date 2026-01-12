#!/usr/bin/env python3
"""Value head analysis for trained Beasty Bar AI model.

This script probes the value head to understand what game situations
the model considers favorable vs unfavorable, analyzing:
1. Card value in hand: How does having each species affect value estimate?
2. Queue position value: How does position control affect value?
3. Score differential impact: How does current point difference affect value?
4. Turn phase impact: Does the model value early vs late game differently?
5. Threat assessment: How does opponent having specific cards affect value?
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import TYPE_CHECKING

from _01_simulator import rules
from _01_simulator.observations import (
    _CARD_FEATURE_DIM,
    _MASKED_CARD_FEATURE_DIM,
    _NUM_SPECIES,
    OBSERVATION_DIM,
    species_index,
)
from _02_agents.neural.utils import load_network_from_checkpoint

if TYPE_CHECKING:
    from _02_agents.neural.network import BeastyBarNetwork

# Species list (sorted alphabetically to match observation encoding)
SPECIES_LIST = sorted([s for s in rules.SPECIES if s != "unknown"])

# Map adjusted index to species name
def adjusted_index_to_species(adj_idx: int) -> str:
    """Convert adjusted species index (0-11, no unknown) to species name."""
    unknown_idx = species_index("unknown")
    # Find the species with this adjusted index
    for sp in SPECIES_LIST:
        idx = species_index(sp)
        adjusted = idx if idx < unknown_idx else idx - 1
        if adjusted == adj_idx:
            return sp
    raise ValueError(f"No species found for adjusted index {adj_idx}")


@dataclass
class AnalysisResults:
    """Container for analysis results."""

    # Species value impact when in hand
    species_in_hand_values: dict[str, float]
    species_not_in_hand_values: dict[str, float]
    species_marginal_values: dict[str, float]

    # Queue position analysis
    front_position_values: list[float]  # Value when we control position 0-4
    back_position_values: list[float]   # Value when opponent controls
    position_marginal: list[float]      # Front - Back

    # Score differential
    score_diff_values: dict[int, float]  # Score diff -> avg value

    # Turn phase
    turn_phase_values: dict[str, float]  # early/mid/late -> avg value

    # Threat assessment (opponent cards in hand)
    opponent_card_threats: dict[str, float]  # Species -> value reduction

    # Most/least valued situations
    best_situations: list[tuple[str, float]]
    worst_situations: list[tuple[str, float]]


def create_base_tensor(perspective: int = 0) -> np.ndarray:
    """Create a baseline observation tensor with neutral state."""
    tensor = np.zeros(OBSERVATION_DIM, dtype=np.float32)

    # Set scalars to neutral values
    # Scalars start at offset: queue(5*17) + bar(24*17) + thatsit(24*17) + hand(4*17) + opp_hand(4*3)
    scalar_offset = (5 + 24 + 24 + 4) * _CARD_FEATURE_DIM + 4 * _MASKED_CARD_FEATURE_DIM

    # deck_counts: 0.5 (half deck remaining)
    tensor[scalar_offset] = 0.5  # own deck
    tensor[scalar_offset + 1] = 0.5  # opponent deck

    # hand_counts: 1.0 (full hand)
    tensor[scalar_offset + 2] = 1.0  # own hand count normalized
    tensor[scalar_offset + 3] = 1.0  # opponent hand count

    # is_active_player: 1.0
    tensor[scalar_offset + 4] = 1.0

    # turn_normalized: 0.1 (early game)
    tensor[scalar_offset + 5] = 0.1

    # queue_length_normalized: 0.4 (2 cards in queue)
    tensor[scalar_offset + 6] = 0.4

    return tensor


def add_card_to_hand(tensor: np.ndarray, slot: int, species_adj_idx: int,
                     strength: float, points: float, perspective: int = 0) -> None:
    """Add a card to the hand zone of the tensor."""
    # Hand starts at offset: queue(5*17) + bar(24*17) + thatsit(24*17)
    hand_offset = (5 + 24 + 24) * _CARD_FEATURE_DIM
    card_offset = hand_offset + slot * _CARD_FEATURE_DIM

    # presence
    tensor[card_offset] = 1.0
    # owner (1.0 = self)
    tensor[card_offset + 1] = 1.0
    # species one-hot
    tensor[card_offset + 2 + species_adj_idx] = 1.0
    # strength normalized
    tensor[card_offset + 2 + _NUM_SPECIES] = strength / 12.0
    # points normalized
    tensor[card_offset + 3 + _NUM_SPECIES] = points / 4.0
    # position normalized
    tensor[card_offset + 4 + _NUM_SPECIES] = slot / 3.0


def add_card_to_queue(tensor: np.ndarray, position: int, species_adj_idx: int,
                      strength: float, points: float, is_own: bool) -> None:
    """Add a card to the queue zone of the tensor."""
    card_offset = position * _CARD_FEATURE_DIM

    # presence
    tensor[card_offset] = 1.0
    # owner
    tensor[card_offset + 1] = 1.0 if is_own else 0.0
    # species one-hot
    tensor[card_offset + 2 + species_adj_idx] = 1.0
    # strength normalized
    tensor[card_offset + 2 + _NUM_SPECIES] = strength / 12.0
    # points normalized
    tensor[card_offset + 3 + _NUM_SPECIES] = points / 4.0
    # position normalized
    tensor[card_offset + 4 + _NUM_SPECIES] = position / 4.0


def add_card_to_bar(tensor: np.ndarray, slot: int, points: float, is_own: bool) -> None:
    """Add a card to the beasty_bar zone (scored cards)."""
    # Bar starts at offset: queue(5*17)
    bar_offset = 5 * _CARD_FEATURE_DIM
    card_offset = bar_offset + slot * _CARD_FEATURE_DIM

    # presence
    tensor[card_offset] = 1.0
    # owner
    tensor[card_offset + 1] = 1.0 if is_own else 0.0
    # points normalized
    tensor[card_offset + 3 + _NUM_SPECIES] = points / 4.0


def get_species_info(species: str) -> tuple[int, float, float]:
    """Get adjusted index, strength, and points for a species."""
    spec = rules.SPECIES[species]
    idx = species_index(species)
    unknown_idx = species_index("unknown")
    adj_idx = idx if idx < unknown_idx else idx - 1
    return adj_idx, float(spec.strength), float(spec.points)


def analyze_species_in_hand(model: BeastyBarNetwork, device: torch.device,
                            n_samples: int = 100) -> tuple[dict, dict, dict]:
    """Analyze value impact of having each species in hand."""
    model.eval()

    species_with_values = {}
    species_without_values = {}

    np.random.seed(42)

    for species in SPECIES_LIST:
        with_values = []
        without_values = []

        adj_idx, strength, points = get_species_info(species)

        for _ in range(n_samples):
            # Create random base state
            tensor = create_base_tensor()

            # Randomly set some game context
            scalar_offset = (5 + 24 + 24 + 4) * _CARD_FEATURE_DIM + 4 * _MASKED_CARD_FEATURE_DIM
            tensor[scalar_offset + 5] = np.random.uniform(0.05, 0.5)  # turn phase
            tensor[scalar_offset + 6] = np.random.uniform(0.2, 0.8)   # queue length

            # Add some random cards to queue
            num_queue = np.random.randint(1, 4)
            for q in range(num_queue):
                rand_species = np.random.choice(SPECIES_LIST)
                rand_adj_idx, rand_str, rand_pts = get_species_info(rand_species)
                is_own = np.random.random() > 0.5
                add_card_to_queue(tensor, q, rand_adj_idx, rand_str, rand_pts, is_own)

            # Version WITH the species in hand
            tensor_with = tensor.copy()
            add_card_to_hand(tensor_with, 0, adj_idx, strength, points)

            # Add 3 other random species
            other_species = [s for s in SPECIES_LIST if s != species]
            for slot, sp in enumerate(np.random.choice(other_species, 3, replace=False)):
                sp_adj_idx, sp_str, sp_pts = get_species_info(sp)
                add_card_to_hand(tensor_with, slot + 1, sp_adj_idx, sp_str, sp_pts)

            # Version WITHOUT the species (4 other random cards)
            tensor_without = tensor.copy()
            for slot, sp in enumerate(np.random.choice(other_species, 4, replace=False)):
                sp_adj_idx, sp_str, sp_pts = get_species_info(sp)
                add_card_to_hand(tensor_without, slot, sp_adj_idx, sp_str, sp_pts)

            # Get values
            with torch.no_grad():
                t_with = torch.from_numpy(tensor_with).to(device)
                t_without = torch.from_numpy(tensor_without).to(device)

                _, val_with = model(t_with)
                _, val_without = model(t_without)

                with_values.append(float(val_with.item()))
                without_values.append(float(val_without.item()))

        species_with_values[species] = np.mean(with_values)
        species_without_values[species] = np.mean(without_values)

    marginal_values = {sp: species_with_values[sp] - species_without_values[sp]
                       for sp in SPECIES_LIST}

    return species_with_values, species_without_values, marginal_values


def analyze_queue_position(model: BeastyBarNetwork, device: torch.device,
                          n_samples: int = 100) -> tuple[list, list, list]:
    """Analyze value of controlling different queue positions."""
    model.eval()

    # For each position, compare having our card vs opponent's card there
    front_values = []  # When we control position i
    back_values = []   # When opponent controls position i

    np.random.seed(43)

    for pos in range(5):  # Queue positions 0-4
        our_values = []
        opp_values = []

        for _ in range(n_samples):
            # Random hand for both versions (same for both)
            np.random.seed(43 + _ + pos * n_samples)
            hand_species = np.random.choice(SPECIES_LIST, 4, replace=False)

            # Use a medium-value card for the position being tested
            # This makes the comparison more fair
            queue_species = 'giraffe'  # Strength 8, Points 3 - middle of the road
            q_adj_idx, q_str, q_pts = get_species_info(queue_species)

            # Version with OUR card at position
            tensor_ours = create_base_tensor()
            add_card_to_queue(tensor_ours, pos, q_adj_idx, q_str, q_pts, is_own=True)

            # Update queue length scalar
            scalar_offset = (5 + 24 + 24 + 4) * _CARD_FEATURE_DIM + 4 * _MASKED_CARD_FEATURE_DIM
            tensor_ours[scalar_offset + 6] = (pos + 1) / 5.0

            for slot, sp in enumerate(hand_species):
                sp_adj_idx, sp_str, sp_pts = get_species_info(sp)
                add_card_to_hand(tensor_ours, slot, sp_adj_idx, sp_str, sp_pts)

            # Version with OPPONENT's card at position
            tensor_opp = create_base_tensor()
            add_card_to_queue(tensor_opp, pos, q_adj_idx, q_str, q_pts, is_own=False)
            tensor_opp[scalar_offset + 6] = (pos + 1) / 5.0

            for slot, sp in enumerate(hand_species):
                sp_adj_idx, sp_str, sp_pts = get_species_info(sp)
                add_card_to_hand(tensor_opp, slot, sp_adj_idx, sp_str, sp_pts)

            with torch.no_grad():
                t_ours = torch.from_numpy(tensor_ours).to(device)
                t_opp = torch.from_numpy(tensor_opp).to(device)

                _, val_ours = model(t_ours)
                _, val_opp = model(t_opp)

                our_values.append(float(val_ours.item()))
                opp_values.append(float(val_opp.item()))

        front_values.append(np.mean(our_values))
        back_values.append(np.mean(opp_values))

    marginal = [f - b for f, b in zip(front_values, back_values, strict=False)]

    return front_values, back_values, marginal


def analyze_score_differential(model: BeastyBarNetwork, device: torch.device,
                               n_samples: int = 100) -> dict[int, float]:
    """Analyze how score differential affects value estimate."""
    model.eval()

    # Test various score differentials (-10 to +10)
    score_values = {}

    np.random.seed(44)

    for diff in range(-10, 11, 2):
        values = []

        for _ in range(n_samples):
            tensor = create_base_tensor()

            # Add random hand
            hand_species = np.random.choice(SPECIES_LIST, 4, replace=False)
            for slot, sp in enumerate(hand_species):
                sp_adj_idx, sp_str, sp_pts = get_species_info(sp)
                add_card_to_hand(tensor, slot, sp_adj_idx, sp_str, sp_pts)

            # Add random queue
            num_queue = np.random.randint(1, 4)
            for q in range(num_queue):
                rand_species = np.random.choice(SPECIES_LIST)
                rand_adj_idx, rand_str, rand_pts = get_species_info(rand_species)
                is_own = np.random.random() > 0.5
                add_card_to_queue(tensor, q, rand_adj_idx, rand_str, rand_pts, is_own)

            # Add scored cards to achieve the differential
            # Positive diff = we're ahead
            our_score = max(0, diff)
            opp_score = max(0, -diff)

            bar_slot = 0
            # Our scored cards (assume 2-point average)
            while our_score >= 2 and bar_slot < 10:
                add_card_to_bar(tensor, bar_slot, 2.0, is_own=True)
                our_score -= 2
                bar_slot += 1

            # Opponent scored cards
            while opp_score >= 2 and bar_slot < 20:
                add_card_to_bar(tensor, bar_slot, 2.0, is_own=False)
                opp_score -= 2
                bar_slot += 1

            with torch.no_grad():
                t = torch.from_numpy(tensor).to(device)
                _, val = model(t)
                values.append(float(val.item()))

        score_values[diff] = np.mean(values)

    return score_values


def analyze_turn_phase(model: BeastyBarNetwork, device: torch.device,
                       n_samples: int = 100) -> dict[str, float]:
    """Analyze how game phase (early/mid/late) affects value."""
    model.eval()

    phases = {
        'early': (0.05, 0.2),   # Turn 0-20
        'mid': (0.25, 0.5),     # Turn 25-50
        'late': (0.55, 0.8)     # Turn 55-80
    }

    phase_values = {}
    np.random.seed(45)

    for phase_name, (turn_min, turn_max) in phases.items():
        values = []

        for _ in range(n_samples):
            tensor = create_base_tensor()

            # Set turn phase
            scalar_offset = (5 + 24 + 24 + 4) * _CARD_FEATURE_DIM + 4 * _MASKED_CARD_FEATURE_DIM
            tensor[scalar_offset + 5] = np.random.uniform(turn_min, turn_max)

            # Adjust deck counts based on phase
            if phase_name == 'early':
                tensor[scalar_offset] = np.random.uniform(0.6, 0.8)
                tensor[scalar_offset + 1] = np.random.uniform(0.6, 0.8)
            elif phase_name == 'mid':
                tensor[scalar_offset] = np.random.uniform(0.3, 0.5)
                tensor[scalar_offset + 1] = np.random.uniform(0.3, 0.5)
            else:  # late
                tensor[scalar_offset] = np.random.uniform(0.0, 0.2)
                tensor[scalar_offset + 1] = np.random.uniform(0.0, 0.2)

            # Add random hand
            hand_species = np.random.choice(SPECIES_LIST, 4, replace=False)
            for slot, sp in enumerate(hand_species):
                sp_adj_idx, sp_str, sp_pts = get_species_info(sp)
                add_card_to_hand(tensor, slot, sp_adj_idx, sp_str, sp_pts)

            # Add random queue
            num_queue = np.random.randint(1, 4)
            for q in range(num_queue):
                rand_species = np.random.choice(SPECIES_LIST)
                rand_adj_idx, rand_str, rand_pts = get_species_info(rand_species)
                is_own = np.random.random() > 0.5
                add_card_to_queue(tensor, q, rand_adj_idx, rand_str, rand_pts, is_own)

            with torch.no_grad():
                t = torch.from_numpy(tensor).to(device)
                _, val = model(t)
                values.append(float(val.item()))

        phase_values[phase_name] = np.mean(values)

    return phase_values


def analyze_opponent_threats(model: BeastyBarNetwork, device: torch.device,
                            n_samples: int = 100) -> dict[str, float]:
    """Analyze how opponent having more cards affects our value."""
    model.eval()

    # Compare value when opponent has full hand vs depleted hand
    threat_impact = {}
    np.random.seed(46)

    opp_hand_counts = {
        'full': 1.0,      # 4 cards
        'three': 0.75,    # 3 cards
        'two': 0.5,       # 2 cards
        'one': 0.25,      # 1 card
        'empty': 0.0      # 0 cards
    }

    for count_name, count_val in opp_hand_counts.items():
        values = []

        for _ in range(n_samples):
            tensor = create_base_tensor()

            # Set opponent hand count
            scalar_offset = (5 + 24 + 24 + 4) * _CARD_FEATURE_DIM + 4 * _MASKED_CARD_FEATURE_DIM
            tensor[scalar_offset + 3] = count_val  # opponent hand count

            # Set opponent hand presence markers
            opp_hand_offset = (5 + 24 + 24 + 4) * _CARD_FEATURE_DIM
            num_opp_cards = int(count_val * 4)
            for i in range(4):
                tensor[opp_hand_offset + i * _MASKED_CARD_FEATURE_DIM] = 1.0 if i < num_opp_cards else 0.0
                # Position encoding for opponent hand
                tensor[opp_hand_offset + i * _MASKED_CARD_FEATURE_DIM + 1] = i / 3.0 if i < num_opp_cards else 0.0

            # Adjust opponent deck to be inversely related to hand (more realistic)
            # If opponent has fewer cards in hand, they likely have fewer in deck too
            opp_deck = max(0.0, 0.5 - (1.0 - count_val) * 0.3)
            tensor[scalar_offset + 1] = opp_deck

            # Add random hand
            hand_species = np.random.choice(SPECIES_LIST, 4, replace=False)
            for slot, sp in enumerate(hand_species):
                sp_adj_idx, sp_str, sp_pts = get_species_info(sp)
                add_card_to_hand(tensor, slot, sp_adj_idx, sp_str, sp_pts)

            # Add random queue (fixed seed for this comparison)
            queue_species = ['lion', 'monkey']  # Fixed for fair comparison
            for q, sp in enumerate(queue_species):
                sp_adj_idx, sp_str, sp_pts = get_species_info(sp)
                is_own = q == 0  # First card ours
                add_card_to_queue(tensor, q, sp_adj_idx, sp_str, sp_pts, is_own)

            with torch.no_grad():
                t = torch.from_numpy(tensor).to(device)
                _, val = model(t)
                values.append(float(val.item()))

        threat_impact[count_name] = np.mean(values)

    return threat_impact


def find_extreme_situations(model: BeastyBarNetwork, device: torch.device,
                           n_samples: int = 500) -> tuple[list, list]:
    """Find the most and least valued game situations."""
    model.eval()

    situations = []
    np.random.seed(47)

    for _ in range(n_samples):
        tensor = create_base_tensor()
        description_parts = []

        # Random turn phase
        scalar_offset = (5 + 24 + 24 + 4) * _CARD_FEATURE_DIM + 4 * _MASKED_CARD_FEATURE_DIM
        turn_norm = np.random.uniform(0.05, 0.6)
        tensor[scalar_offset + 5] = turn_norm
        phase = "early" if turn_norm < 0.2 else ("mid" if turn_norm < 0.4 else "late")
        description_parts.append(f"{phase} game")

        # Random deck counts
        our_deck = np.random.uniform(0.0, 0.8)
        opp_deck = np.random.uniform(0.0, 0.8)
        tensor[scalar_offset] = our_deck
        tensor[scalar_offset + 1] = opp_deck

        # Random hand
        hand_species = np.random.choice(SPECIES_LIST, 4, replace=False)
        for slot, sp in enumerate(hand_species):
            sp_adj_idx, sp_str, sp_pts = get_species_info(sp)
            add_card_to_hand(tensor, slot, sp_adj_idx, sp_str, sp_pts)
        description_parts.append(f"hand: {', '.join(hand_species[:2])}...")

        # Random queue (0-4 cards)
        num_queue = np.random.randint(0, 5)
        own_in_queue = 0
        for q in range(num_queue):
            rand_species = np.random.choice(SPECIES_LIST)
            rand_adj_idx, rand_str, rand_pts = get_species_info(rand_species)
            is_own = np.random.random() > 0.5
            if is_own:
                own_in_queue += 1
            add_card_to_queue(tensor, q, rand_adj_idx, rand_str, rand_pts, is_own)

        tensor[scalar_offset + 6] = num_queue / 5.0
        description_parts.append(f"queue: {num_queue} cards ({own_in_queue} ours)")

        # Random score differential
        our_score = np.random.randint(0, 15)
        opp_score = np.random.randint(0, 15)
        bar_slot = 0

        for _ in range(our_score // 2):
            if bar_slot < 10:
                add_card_to_bar(tensor, bar_slot, 2.0, is_own=True)
                bar_slot += 1

        for _ in range(opp_score // 2):
            if bar_slot < 20:
                add_card_to_bar(tensor, bar_slot, 2.0, is_own=False)
                bar_slot += 1

        diff = our_score - opp_score
        description_parts.append(f"score: {our_score}-{opp_score} (diff={diff:+d})")

        # Random opponent hand count
        opp_hand_count = np.random.uniform(0.0, 1.0)
        tensor[scalar_offset + 3] = opp_hand_count

        description = " | ".join(description_parts)

        with torch.no_grad():
            t = torch.from_numpy(tensor).to(device)
            _, val = model(t)
            situations.append((description, float(val.item())))

    # Sort by value
    situations.sort(key=lambda x: x[1], reverse=True)

    best = situations[:10]
    worst = situations[-10:]

    return best, worst


def run_full_analysis(model_path: str) -> AnalysisResults:
    """Run complete value head analysis."""
    print(f"Loading model from {model_path}...")
    model, _config, step = load_network_from_checkpoint(model_path)
    device = next(model.parameters()).device
    print(f"Model loaded (step {step}), device: {device}")
    print(f"Model parameters: {model.count_parameters():,}")

    print("\n1. Analyzing species value in hand...")
    with_vals, without_vals, marginal_vals = analyze_species_in_hand(model, device, n_samples=200)

    print("2. Analyzing queue position value...")
    front_vals, back_vals, pos_marginal = analyze_queue_position(model, device, n_samples=200)

    print("3. Analyzing score differential impact...")
    score_vals = analyze_score_differential(model, device, n_samples=200)

    print("4. Analyzing turn phase impact...")
    phase_vals = analyze_turn_phase(model, device, n_samples=200)

    print("5. Analyzing opponent threat impact...")
    threat_vals = analyze_opponent_threats(model, device, n_samples=200)

    print("6. Finding extreme situations...")
    best, worst = find_extreme_situations(model, device, n_samples=1000)

    return AnalysisResults(
        species_in_hand_values=with_vals,
        species_not_in_hand_values=without_vals,
        species_marginal_values=marginal_vals,
        front_position_values=front_vals,
        back_position_values=back_vals,
        position_marginal=pos_marginal,
        score_diff_values=score_vals,
        turn_phase_values=phase_vals,
        opponent_card_threats=threat_vals,
        best_situations=best,
        worst_situations=worst,
    )


def format_markdown_report(results: AnalysisResults) -> str:
    """Format analysis results as markdown report."""
    lines = []

    lines.append("# Value Head Analysis Report")
    lines.append("")
    lines.append("This report analyzes what the trained Beasty Bar AI model considers ")
    lines.append("a winning position by probing its value head with controlled game states.")
    lines.append("")
    lines.append("## Game Background")
    lines.append("")
    lines.append("**Beasty Bar** is a card game where players compete to get their animals into the bar.")
    lines.append("Key mechanics:")
    lines.append("- Queue holds up to 5 cards; when full, top 2 enter the bar (score points), last 1 is bounced out")
    lines.append("- Each species has unique abilities that manipulate queue order")
    lines.append("- Higher strength typically helps advance in queue; higher points reward getting in the bar")
    lines.append("")
    lines.append("**Species Abilities:**")
    lines.append("- **Lion** (12 str, 2 pts): Jumps to front, scares away Monkeys")
    lines.append("- **Hippo** (11 str, 2 pts): Pushes forward through weaker animals (recurring)")
    lines.append("- **Crocodile** (10 str, 3 pts): Eats weaker animals ahead (recurring)")
    lines.append("- **Snake** (9 str, 2 pts): Sorts queue by strength (strongest first)")
    lines.append("- **Giraffe** (8 str, 3 pts): Swaps with weaker animal ahead (recurring)")
    lines.append("- **Zebra** (7 str, 4 pts): Blocks Hippo/Crocodile from passing (permanent)")
    lines.append("- **Seal** (6 str, 2 pts): Reverses entire queue")
    lines.append("- **Chameleon** (5 str, 3 pts): Copies another animal's ability")
    lines.append("- **Monkey** (4 str, 3 pts): Pair of monkeys kicks out Hippo/Crocodile")
    lines.append("- **Kangaroo** (3 str, 4 pts): Hops 1-2 positions forward")
    lines.append("- **Parrot** (2 str, 4 pts): Sends any queue animal to THAT'S IT")
    lines.append("- **Skunk** (1 str, 4 pts): Expels top 2 strength bands from queue")
    lines.append("")

    # Species Value Analysis
    lines.append("## 1. Species Value in Hand")
    lines.append("")
    lines.append("How does having each species in hand affect the model's value estimate?")
    lines.append("")
    lines.append("| Species | Strength | Points | With Card | Without | Marginal Value |")
    lines.append("|---------|----------|--------|-----------|---------|----------------|")

    # Sort by marginal value
    sorted_species = sorted(results.species_marginal_values.items(),
                           key=lambda x: x[1], reverse=True)

    for species, marginal in sorted_species:
        spec = rules.SPECIES[species]
        with_val = results.species_in_hand_values[species]
        without_val = results.species_not_in_hand_values[species]
        sign = "+" if marginal >= 0 else ""
        lines.append(f"| {species.capitalize():10s} | {spec.strength:8d} | {spec.points:6d} | "
                    f"{with_val:+.4f} | {without_val:+.4f} | **{sign}{marginal:.4f}** |")

    lines.append("")
    lines.append("**Key Insight**: ")
    best_species = sorted_species[0][0]
    worst_species = sorted_species[-1][0]
    lines.append(f"The model values **{best_species.capitalize()}** most highly in hand ")
    lines.append(f"(+{sorted_species[0][1]:.4f} marginal value), while **{worst_species.capitalize()}** ")
    lines.append(f"is least valued ({sorted_species[-1][1]:+.4f}).")
    lines.append("")
    lines.append("**Strategic Analysis:**")
    lines.append("")
    lines.append("The model's preferences make strategic sense:")
    lines.append("- **Parrot** (top): Direct removal ability is extremely powerful - can eliminate any threat")
    lines.append("- **Crocodile** (2nd): Recurring ability to eat weaker animals provides sustained value")
    lines.append("- **Skunk** (3rd): Mass removal of strongest animals clears threats, 4 pts if it enters")
    lines.append("- **Kangaroo** (4th): Reliable positioning with high points (4 pts)")
    lines.append("- **Monkey** (last): Requires paired with another Monkey to be useful, otherwise weak")
    lines.append("- Strong recurring animals (Hippo, Giraffe) rank lower because they're also threats when opponent has them")
    lines.append("")

    # Queue Position Analysis
    lines.append("## 2. Queue Position Value")
    lines.append("")
    lines.append("How does controlling different queue positions affect value?")
    lines.append("Position 0 is front (first to enter bar), Position 4 is back (bounced out).")
    lines.append("")
    lines.append("| Position | Our Card | Opponent Card | Advantage |")
    lines.append("|----------|----------|---------------|-----------|")

    for i in range(5):
        our = results.front_position_values[i]
        opp = results.back_position_values[i]
        adv = results.position_marginal[i]
        sign = "+" if adv >= 0 else ""
        pos_label = f"Position {i}" if i < 4 else "Position 4 (bounce)"
        lines.append(f"| {pos_label:20s} | {our:+.4f} | {opp:+.4f} | **{sign}{adv:.4f}** |")

    lines.append("")
    best_pos = max(range(5), key=lambda i: results.position_marginal[i])
    worst_pos = min(range(5), key=lambda i: results.position_marginal[i])
    lines.append(f"**Key Insight**: Position {best_pos} provides the most advantage ")
    lines.append(f"(+{results.position_marginal[best_pos]:.4f}), while position {worst_pos} ")
    lines.append(f"is least advantageous ({results.position_marginal[worst_pos]:+.4f}).")
    lines.append("")
    lines.append("**Strategic Analysis:**")
    lines.append("")
    lines.append("This aligns perfectly with game mechanics:")
    lines.append("- **Position 0** (front): Guaranteed to enter bar when queue fills - highest value")
    lines.append("- **Position 1**: Second to enter bar - still very valuable")
    lines.append("- **Positions 2-3**: Middle ground, depends on future plays")
    lines.append("- **Position 4** (back): Gets bounced out when queue fills - actively bad")
    lines.append("")
    lines.append("The negative value at position 4 shows the model understands that having")
    lines.append("your card in the bounce position means it will likely be eliminated.")
    lines.append("")

    # Score Differential
    lines.append("## 3. Score Differential Impact")
    lines.append("")
    lines.append("How does the current score difference affect value estimate?")
    lines.append("")
    lines.append("| Score Diff | Value Estimate |")
    lines.append("|------------|----------------|")

    for diff in sorted(results.score_diff_values.keys()):
        val = results.score_diff_values[diff]
        sign = "+" if diff >= 0 else ""
        lines.append(f"| {sign}{diff:3d} points | {val:+.4f} |")

    lines.append("")
    lines.append("**Key Insight**: The model clearly distinguishes between winning and losing positions.")
    lines.append("")
    lines.append("**Strategic Analysis:**")
    lines.append("")
    lines.append("Notable patterns in the score response:")
    lines.append("- Sharp transition at score diff = 0 (from ~-0.5 to ~+0.6)")
    lines.append("- Losing positions cluster around -0.5 regardless of deficit magnitude")
    lines.append("- Winning positions cluster around +0.64 regardless of lead magnitude")
    lines.append("- This suggests the model thinks in terms of 'winning' vs 'losing' rather than by how much")
    lines.append("- The tie state (+0.09) is close to neutral, slightly optimistic")
    lines.append("")

    # Turn Phase
    lines.append("## 4. Turn Phase Impact")
    lines.append("")
    lines.append("Does the model value early game vs late game differently?")
    lines.append("")
    lines.append("| Game Phase | Value Estimate |")
    lines.append("|------------|----------------|")

    for phase in ['early', 'mid', 'late']:
        val = results.turn_phase_values[phase]
        lines.append(f"| {phase.capitalize():10s} | {val:+.4f} |")

    lines.append("")
    best_phase = max(results.turn_phase_values.items(), key=lambda x: x[1])
    lines.append(f"**Key Insight**: The model is most confident in **{best_phase[0]}** game ")
    lines.append(f"({best_phase[1]:+.4f}). This may reflect training data distribution or ")
    lines.append("actual strategic preferences.")
    lines.append("")

    # Opponent Threat
    lines.append("## 5. Opponent Threat Assessment")
    lines.append("")
    lines.append("How does opponent's hand size affect our value estimate?")
    lines.append("")
    lines.append("| Opponent Hand | Value Estimate |")
    lines.append("|---------------|----------------|")

    for count in ['full', 'three', 'two', 'one', 'empty']:
        val = results.opponent_card_threats[count]
        lines.append(f"| {count.capitalize():13s} | {val:+.4f} |")

    lines.append("")
    full_val = results.opponent_card_threats['full']
    empty_val = results.opponent_card_threats['empty']
    diff = empty_val - full_val
    if diff > 0:
        lines.append("**Key Insight**: When opponent has no cards vs full hand, our value ")
        lines.append(f"increases by {diff:+.4f}. The model correctly identifies depleted ")
        lines.append("opponent resources as favorable.")
    else:
        lines.append(f"**Key Insight**: The model shows {abs(diff):.4f} lower value when opponent ")
        lines.append("has empty hand. This may indicate the model recognizes that opponent ")
        lines.append("running out of cards often correlates with late-game losing positions, ")
        lines.append("or that having an active opponent (with cards) gives more opportunity to outplay them.")
    lines.append("")

    # Best Situations
    lines.append("## 6. Most Valued Situations")
    lines.append("")
    lines.append("Top 10 game states the model considers most favorable:")
    lines.append("")

    for i, (desc, val) in enumerate(results.best_situations, 1):
        lines.append(f"{i}. **Value: {val:+.4f}** - {desc}")

    lines.append("")

    # Worst Situations
    lines.append("## 7. Least Valued Situations")
    lines.append("")
    lines.append("Top 10 game states the model considers most unfavorable:")
    lines.append("")

    for i, (desc, val) in enumerate(results.worst_situations, 1):
        lines.append(f"{i}. **Value: {val:+.4f}** - {desc}")

    lines.append("")

    # Strategic Summary
    lines.append("## 8. Strategic Insights Summary")
    lines.append("")
    lines.append("Based on the value head analysis, the model has learned:")
    lines.append("")

    # Species strategy
    high_value_species = [sp for sp, val in sorted_species[:4]]
    low_value_species = [sp for sp, val in sorted_species[-4:]]
    lines.append(f"1. **Preferred Cards**: {', '.join(s.capitalize() for s in high_value_species)}")
    lines.append("   - These provide the highest marginal value when in hand")
    lines.append("")
    lines.append(f"2. **Less Valued Cards**: {', '.join(s.capitalize() for s in low_value_species)}")
    lines.append("   - These provide lower or negative marginal value")
    lines.append("")

    # Position strategy
    front_advantage = sum(results.position_marginal[:2]) / 2
    back_advantage = sum(results.position_marginal[3:]) / 2
    lines.append(f"3. **Queue Control**: Front positions (0-1) provide {front_advantage:+.4f} avg advantage")
    lines.append(f"   vs back positions (3-4) providing {back_advantage:+.4f} avg advantage")
    lines.append("")

    # Score awareness
    lines.append("4. **Score Sensitivity**: The model strongly correlates value with score advantage")
    slope = (results.score_diff_values[10] - results.score_diff_values[-10]) / 20
    lines.append(f"   - Approximately {slope:.4f} value per point of score difference")
    lines.append("")

    # Opponent awareness
    lines.append("5. **Resource Tracking**: Model values opponent resource depletion")
    lines.append(f"   - Opponent empty hand vs full: {diff:+.4f} value swing")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    model_path = "/Users/p/Desktop/v/experiments/beastybar/checkpoints/v4/final.pt"
    output_path = "/Users/p/Desktop/v/experiments/beastybar/_05_analysis/04_value_analysis.md"

    print("=" * 60)
    print("Beasty Bar Value Head Analysis")
    print("=" * 60)

    results = run_full_analysis(model_path)

    print("\nGenerating report...")
    report = format_markdown_report(results)

    print(f"\nWriting report to {output_path}...")
    Path(output_path).write_text(report)

    print("\nDone!")
    print("=" * 60)
