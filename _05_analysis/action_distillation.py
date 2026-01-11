"""Action-level policy distillation from PPO model.

This module trains separate classifiers for each action type to understand
how the PPO model makes decisions about:
- Parrot: Which queue position to target?
- Kangaroo: How far to hop (1 or 2)?
- Chameleon: Which card to copy?
- Basic plays: When to play each non-parameterized card?

The goal is to achieve 60%+ per-action fidelity, capturing the action-specific
nuances that species-level distillation misses.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from _01_simulator import engine, rules, state
from _01_simulator.action_space import (
    index_to_action,
    legal_action_mask_tensor,
)
from _01_simulator.observations import (
    state_to_tensor,
)

# Species names (sorted alphabetically)
SPECIES_NAMES = sorted([s for s in rules.SPECIES.keys() if s != "unknown"])
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_NAMES)}


@dataclass
class ActionFeatures:
    """Features for action-level decision making."""

    # Context features
    queue_length: int = 0
    turn_number: int = 0
    score_diff: int = 0
    my_bar_points: int = 0
    opp_bar_points: int = 0

    # Hand context
    hand_size: int = 0
    has_monkey: bool = False
    has_parrot: bool = False
    has_kangaroo: bool = False
    has_chameleon: bool = False

    # Queue composition for each position
    queue_species: list[str] = field(default_factory=list)
    queue_owners: list[int] = field(default_factory=list)
    queue_strengths: list[int] = field(default_factory=list)
    queue_points: list[int] = field(default_factory=list)

    # Aggregated queue stats
    opp_cards_in_queue: int = 0
    my_cards_in_queue: int = 0
    max_opp_strength: int = 0
    max_my_strength: int = 0
    opp_points_in_queue: int = 0
    my_points_in_queue: int = 0

    # Threat indicators
    has_opp_lion: bool = False
    has_opp_hippo: bool = False
    has_opp_crocodile: bool = False
    has_opp_zebra: bool = False
    has_monkey_in_queue: bool = False

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for sklearn."""
        # Base features
        features = [
            self.queue_length,
            self.turn_number,
            self.score_diff,
            self.my_bar_points,
            self.opp_bar_points,
            self.hand_size,
            int(self.has_monkey),
            int(self.has_parrot),
            int(self.has_kangaroo),
            int(self.has_chameleon),
            self.opp_cards_in_queue,
            self.my_cards_in_queue,
            self.max_opp_strength,
            self.max_my_strength,
            self.opp_points_in_queue,
            self.my_points_in_queue,
            int(self.has_opp_lion),
            int(self.has_opp_hippo),
            int(self.has_opp_crocodile),
            int(self.has_opp_zebra),
            int(self.has_monkey_in_queue),
        ]

        # Per-position features (padded to 5 positions)
        for i in range(5):
            if i < len(self.queue_species):
                species = self.queue_species[i]
                owner = self.queue_owners[i]
                strength = self.queue_strengths[i]
                points = self.queue_points[i]

                features.extend([
                    SPECIES_TO_IDX.get(species, -1),
                    owner,
                    strength,
                    points,
                    1,  # present flag
                ])
            else:
                features.extend([0, -1, 0, 0, 0])

        return np.array(features, dtype=np.float32)


@dataclass
class ParrotTargetFeatures:
    """Features specific to parrot targeting decision."""

    # Target card info
    target_species: str = ""
    target_owner: int = -1
    target_strength: int = 0
    target_points: int = 0
    target_position: int = 0

    # Contextual
    is_opponent_card: bool = False
    is_front_card: bool = False
    is_threat: bool = False
    is_recurring: bool = False

    # Relative to alternatives
    is_highest_points_target: bool = False
    is_highest_strength_target: bool = False

    # Simulated outcome (if available)
    points_removed: int = 0
    threat_removed: bool = False

    def to_array(self) -> np.ndarray:
        """Convert to feature array."""
        return np.array([
            SPECIES_TO_IDX.get(self.target_species, -1),
            self.target_owner,
            self.target_strength,
            self.target_points,
            self.target_position,
            int(self.is_opponent_card),
            int(self.is_front_card),
            int(self.is_threat),
            int(self.is_recurring),
            int(self.is_highest_points_target),
            int(self.is_highest_strength_target),
            self.points_removed,
            int(self.threat_removed),
        ], dtype=np.float32)


@dataclass
class KangarooHopFeatures:
    """Features for kangaroo hop distance decision."""

    queue_length: int = 0
    position_after_hop1: int = 0
    position_after_hop2: int = 0

    # Cards at landing positions
    hop1_lands_behind_species: str = ""
    hop2_lands_behind_species: str = ""

    # Safety indicators
    hop1_safer: bool = False  # e.g., avoids crocodile
    hop2_reaches_front: bool = False

    def to_array(self) -> np.ndarray:
        return np.array([
            self.queue_length,
            self.position_after_hop1,
            self.position_after_hop2,
            SPECIES_TO_IDX.get(self.hop1_lands_behind_species, -1),
            SPECIES_TO_IDX.get(self.hop2_lands_behind_species, -1),
            int(self.hop1_safer),
            int(self.hop2_reaches_front),
        ], dtype=np.float32)


def extract_action_features(game_state: state.State, player: int) -> ActionFeatures:
    """Extract features for action-level decision making."""
    opponent = 1 - player
    queue = game_state.zones.queue

    # Score calculation
    my_bar = sum(c.points for c in game_state.zones.beasty_bar if c.owner == player)
    opp_bar = sum(c.points for c in game_state.zones.beasty_bar if c.owner == opponent)

    # Hand info
    hand = game_state.players[player].hand
    hand_species = {c.species for c in hand}

    # Queue analysis
    queue_species = [c.species for c in queue]
    queue_owners = [c.owner for c in queue]
    queue_strengths = [c.strength for c in queue]
    queue_points = [c.points for c in queue]

    opp_cards = [c for c in queue if c.owner == opponent]
    my_cards = [c for c in queue if c.owner == player]

    threat_species = {"lion", "hippo", "crocodile", "zebra"}

    return ActionFeatures(
        queue_length=len(queue),
        turn_number=game_state.turn,
        score_diff=my_bar - opp_bar,
        my_bar_points=my_bar,
        opp_bar_points=opp_bar,
        hand_size=len(hand),
        has_monkey="monkey" in hand_species,
        has_parrot="parrot" in hand_species,
        has_kangaroo="kangaroo" in hand_species,
        has_chameleon="chameleon" in hand_species,
        queue_species=queue_species,
        queue_owners=queue_owners,
        queue_strengths=queue_strengths,
        queue_points=queue_points,
        opp_cards_in_queue=len(opp_cards),
        my_cards_in_queue=len(my_cards),
        max_opp_strength=max((c.strength for c in opp_cards), default=0),
        max_my_strength=max((c.strength for c in my_cards), default=0),
        opp_points_in_queue=sum(c.points for c in opp_cards),
        my_points_in_queue=sum(c.points for c in my_cards),
        has_opp_lion=any(c.species == "lion" for c in opp_cards),
        has_opp_hippo=any(c.species == "hippo" for c in opp_cards),
        has_opp_crocodile=any(c.species == "crocodile" for c in opp_cards),
        has_opp_zebra=any(c.species == "zebra" for c in opp_cards),
        has_monkey_in_queue=any(c.species == "monkey" for c in queue),
    )


def extract_parrot_target_features(
    game_state: state.State,
    player: int,
    target_idx: int,
) -> ParrotTargetFeatures:
    """Extract features for a specific parrot target."""
    queue = game_state.zones.queue
    opponent = 1 - player

    if target_idx < 0 or target_idx >= len(queue):
        return ParrotTargetFeatures()

    target = queue[target_idx]
    threat_species = {"lion", "hippo", "crocodile"}
    recurring_species = {"giraffe", "hippo", "crocodile"}

    # Find highest points/strength targets for relative comparison
    opp_targets = [c for c in queue if c.owner == opponent]
    max_points = max((c.points for c in opp_targets), default=0)
    max_strength = max((c.strength for c in opp_targets), default=0)

    return ParrotTargetFeatures(
        target_species=target.species,
        target_owner=target.owner,
        target_strength=target.strength,
        target_points=target.points,
        target_position=target_idx,
        is_opponent_card=target.owner == opponent,
        is_front_card=target_idx == 0,
        is_threat=target.species in threat_species,
        is_recurring=target.species in recurring_species,
        is_highest_points_target=target.points == max_points and target.owner == opponent,
        is_highest_strength_target=target.strength == max_strength and target.owner == opponent,
        points_removed=target.points if target.owner == opponent else -target.points,
        threat_removed=target.owner == opponent and target.species in threat_species,
    )


def extract_kangaroo_features(
    game_state: state.State,
    player: int,
) -> KangarooHopFeatures:
    """Extract features for kangaroo hop decision."""
    queue = game_state.zones.queue
    queue_len = len(queue)

    # After playing kangaroo, it lands at back then hops
    # Position after hop1: queue_len - 1 (hop over 1 card)
    # Position after hop2: queue_len - 2 (hop over 2 cards)

    features = KangarooHopFeatures(queue_length=queue_len)

    if queue_len >= 1:
        features.position_after_hop1 = queue_len - 1
        # Card at position that hop1 would land behind
        if queue_len >= 1:
            features.hop1_lands_behind_species = queue[queue_len - 1].species

    if queue_len >= 2:
        features.position_after_hop2 = queue_len - 2
        features.hop2_reaches_front = queue_len <= 3
        if queue_len >= 2:
            features.hop2_lands_behind_species = queue[queue_len - 2].species
            # Check if hop2 avoids danger
            hop1_card = queue[queue_len - 1]
            if hop1_card.species in {"crocodile"} and hop1_card.owner != player:
                features.hop1_safer = False
            else:
                features.hop1_safer = True

    return features


@dataclass
class ActionDistillationData:
    """Training data for action distillation."""

    # Parrot targeting data
    parrot_features: list[np.ndarray] = field(default_factory=list)
    parrot_context: list[np.ndarray] = field(default_factory=list)
    parrot_labels: list[int] = field(default_factory=list)  # target position chosen

    # Kangaroo hop data
    kangaroo_features: list[np.ndarray] = field(default_factory=list)
    kangaroo_labels: list[int] = field(default_factory=list)  # hop distance chosen

    # Chameleon copy data
    chameleon_features: list[np.ndarray] = field(default_factory=list)
    chameleon_labels: list[int] = field(default_factory=list)  # target position chosen

    # Species selection data (which card to play when multiple available)
    species_features: list[np.ndarray] = field(default_factory=list)
    species_labels: list[str] = field(default_factory=list)  # species chosen


def collect_action_data_from_ppo(
    model_path: str,
    n_samples: int = 100000,
    device: str = "cpu",
) -> ActionDistillationData:
    """Collect action-level training data from PPO model.

    Generates game states and records which action parameters the PPO
    model selects for each action type.

    Args:
        model_path: Path to trained PPO model.
        n_samples: Number of decision samples to collect.
        device: Device to run model on.

    Returns:
        ActionDistillationData with samples for each action type.
    """
    try:
        import torch

        from _02_agents.neural.agent import load_neural_agent
        from _02_agents.neural.utils import greedy_action
    except ImportError as e:
        raise ImportError(f"Required dependencies not available: {e}")

    # Load model
    agent = load_neural_agent(model_path, device=device)
    agent.model.eval()

    data = ActionDistillationData()
    rng = random.Random(42)

    collected = 0
    while collected < n_samples:
        # Generate random game state
        seed = rng.randint(0, 1_000_000)
        game_state = state.initial_state(seed)

        # Play some random moves to get interesting states
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

        # Get PPO model's choice
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
            continue

        chosen_card = hand[chosen_action.hand_index]
        context_features = extract_action_features(game_state, player)

        # Record action-specific data
        if chosen_card.species == "parrot" and chosen_action.params:
            target_idx = chosen_action.params[0]
            target_features = extract_parrot_target_features(game_state, player, target_idx)
            data.parrot_features.append(target_features.to_array())
            data.parrot_context.append(context_features.to_array())
            data.parrot_labels.append(target_idx)

        elif chosen_card.species == "kangaroo" and chosen_action.params:
            hop_distance = chosen_action.params[0]
            hop_features = extract_kangaroo_features(game_state, player)
            data.kangaroo_features.append(hop_features.to_array())
            data.kangaroo_labels.append(hop_distance)

        elif chosen_card.species == "chameleon" and chosen_action.params:
            target_idx = chosen_action.params[0]
            data.chameleon_features.append(context_features.to_array())
            data.chameleon_labels.append(target_idx)

        # Always record species selection
        data.species_features.append(context_features.to_array())
        data.species_labels.append(chosen_card.species)

        collected += 1
        if collected % 10000 == 0:
            print(f"Collected {collected}/{n_samples} samples...")

    return data


def train_parrot_classifier(
    data: ActionDistillationData,
    max_depth: int = 6,
) -> tuple[Any, float, dict[str, float]]:
    """Train decision tree classifier for parrot targeting.

    Returns:
        Tuple of (classifier, accuracy, feature_importances).
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        raise ImportError("scikit-learn required for training classifiers")

    if len(data.parrot_features) < 100:
        print("Not enough parrot samples for training")
        return None, 0.0, {}

    # Combine target features with context
    X = np.array([
        np.concatenate([pf, cf])
        for pf, cf in zip(data.parrot_features, data.parrot_context)
    ])
    y = np.array(data.parrot_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    # Feature names
    parrot_feature_names = [
        "target_species", "target_owner", "target_strength", "target_points",
        "target_position", "is_opponent", "is_front", "is_threat",
        "is_recurring", "is_highest_points", "is_highest_strength",
        "points_removed", "threat_removed",
    ]
    context_feature_names = _get_context_feature_names()
    feature_names = parrot_feature_names + context_feature_names

    importances = dict(zip(feature_names, clf.feature_importances_))

    return clf, accuracy, importances


def train_kangaroo_classifier(
    data: ActionDistillationData,
    max_depth: int = 4,
) -> tuple[Any, float, dict[str, float]]:
    """Train classifier for kangaroo hop distance."""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        raise ImportError("scikit-learn required for training classifiers")

    if len(data.kangaroo_features) < 100:
        print("Not enough kangaroo samples for training")
        return None, 0.0, {}

    X = np.array(data.kangaroo_features)
    y = np.array(data.kangaroo_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    feature_names = [
        "queue_length", "position_after_hop1", "position_after_hop2",
        "hop1_behind_species", "hop2_behind_species",
        "hop1_safer", "hop2_reaches_front",
    ]
    importances = dict(zip(feature_names, clf.feature_importances_))

    return clf, accuracy, importances


def train_species_classifier(
    data: ActionDistillationData,
    max_depth: int = 8,
) -> tuple[Any, float, dict[str, float]]:
    """Train classifier for species selection."""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        raise ImportError("scikit-learn required for training classifiers")

    if len(data.species_features) < 100:
        print("Not enough species samples for training")
        return None, 0.0, {}

    X = np.array(data.species_features)
    y = np.array(data.species_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    feature_names = _get_context_feature_names()
    importances = dict(zip(feature_names, clf.feature_importances_))

    return clf, accuracy, importances


def _get_context_feature_names() -> list[str]:
    """Get feature names for ActionFeatures."""
    names = [
        "queue_length", "turn_number", "score_diff", "my_bar_points",
        "opp_bar_points", "hand_size", "has_monkey", "has_parrot",
        "has_kangaroo", "has_chameleon", "opp_cards_in_queue",
        "my_cards_in_queue", "max_opp_strength", "max_my_strength",
        "opp_points_in_queue", "my_points_in_queue", "has_opp_lion",
        "has_opp_hippo", "has_opp_crocodile", "has_opp_zebra",
        "has_monkey_in_queue",
    ]

    # Per-position features
    for i in range(5):
        names.extend([
            f"pos{i}_species", f"pos{i}_owner", f"pos{i}_strength",
            f"pos{i}_points", f"pos{i}_present",
        ])

    return names


def generate_action_distillation_report(
    data: ActionDistillationData,
    parrot_clf: Any,
    parrot_acc: float,
    parrot_imp: dict[str, float],
    kangaroo_clf: Any,
    kangaroo_acc: float,
    kangaroo_imp: dict[str, float],
    species_clf: Any,
    species_acc: float,
    species_imp: dict[str, float],
    output_path: str = "_05_analysis/06_action_distillation.md",
) -> None:
    """Generate markdown report of action distillation results."""

    report = []
    report.append("# Action-Level Distillation Report\n")
    report.append("This report analyzes per-action-type decision patterns from the PPO model.\n")

    # Data summary
    report.append("## Data Summary\n")
    report.append(f"- Parrot targeting samples: {len(data.parrot_labels)}")
    report.append(f"- Kangaroo hop samples: {len(data.kangaroo_labels)}")
    report.append(f"- Chameleon copy samples: {len(data.chameleon_labels)}")
    report.append(f"- Species selection samples: {len(data.species_labels)}\n")

    # Parrot analysis
    report.append("## Parrot Targeting\n")
    if parrot_clf:
        report.append(f"**Accuracy: {parrot_acc:.1%}**\n")
        report.append("### Top Feature Importances\n")
        top_parrot = sorted(parrot_imp.items(), key=lambda x: -x[1])[:10]
        for name, imp in top_parrot:
            report.append(f"- {name}: {imp:.3f}")

        # Target distribution
        target_counts = Counter(data.parrot_labels)
        report.append("\n### Target Position Distribution\n")
        for pos in sorted(target_counts.keys()):
            count = target_counts[pos]
            pct = count / len(data.parrot_labels) * 100
            report.append(f"- Position {pos}: {count} ({pct:.1f}%)")
    else:
        report.append("*Insufficient data for parrot classifier*\n")

    # Kangaroo analysis
    report.append("\n## Kangaroo Hop Distance\n")
    if kangaroo_clf:
        report.append(f"**Accuracy: {kangaroo_acc:.1%}**\n")
        report.append("### Top Feature Importances\n")
        top_kangaroo = sorted(kangaroo_imp.items(), key=lambda x: -x[1])[:5]
        for name, imp in top_kangaroo:
            report.append(f"- {name}: {imp:.3f}")

        # Hop distribution
        hop_counts = Counter(data.kangaroo_labels)
        report.append("\n### Hop Distance Distribution\n")
        for hop in sorted(hop_counts.keys()):
            count = hop_counts[hop]
            pct = count / len(data.kangaroo_labels) * 100
            report.append(f"- Hop {hop}: {count} ({pct:.1f}%)")
    else:
        report.append("*Insufficient data for kangaroo classifier*\n")

    # Species selection analysis
    report.append("\n## Species Selection\n")
    if species_clf:
        report.append(f"**Accuracy: {species_acc:.1%}**\n")
        report.append("### Top Feature Importances\n")
        top_species = sorted(species_imp.items(), key=lambda x: -x[1])[:15]
        for name, imp in top_species:
            report.append(f"- {name}: {imp:.3f}")

        # Species distribution
        species_counts = Counter(data.species_labels)
        report.append("\n### Species Play Distribution\n")
        for species, count in species_counts.most_common():
            pct = count / len(data.species_labels) * 100
            report.append(f"- {species}: {count} ({pct:.1f}%)")
    else:
        report.append("*Insufficient data for species classifier*\n")

    # Write report
    output = Path(project_root) / output_path
    output.write_text("\n".join(report))
    print(f"Report written to {output}")


def main():
    """Run action distillation analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Action-level policy distillation")
    parser.add_argument(
        "--model",
        default="checkpoints/v4/final.pt",
        help="Path to trained PPO model",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50000,
        help="Number of decision samples to collect",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run model on",
    )
    args = parser.parse_args()

    print(f"Collecting {args.samples} action samples from {args.model}...")
    data = collect_action_data_from_ppo(args.model, args.samples, args.device)

    print("\nTraining classifiers...")

    print("  Training parrot classifier...")
    parrot_clf, parrot_acc, parrot_imp = train_parrot_classifier(data)
    if parrot_clf:
        print(f"  Parrot accuracy: {parrot_acc:.1%}")

    print("  Training kangaroo classifier...")
    kangaroo_clf, kangaroo_acc, kangaroo_imp = train_kangaroo_classifier(data)
    if kangaroo_clf:
        print(f"  Kangaroo accuracy: {kangaroo_acc:.1%}")

    print("  Training species classifier...")
    species_clf, species_acc, species_imp = train_species_classifier(data)
    if species_clf:
        print(f"  Species accuracy: {species_acc:.1%}")

    print("\nGenerating report...")
    generate_action_distillation_report(
        data,
        parrot_clf, parrot_acc, parrot_imp,
        kangaroo_clf, kangaroo_acc, kangaroo_imp,
        species_clf, species_acc, species_imp,
    )


if __name__ == "__main__":
    main()
