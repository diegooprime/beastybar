"""Decision Tree Analysis: Extract interpretable rules from neural network.

This script trains an interpretable decision tree to mimic the neural network's
decisions, extracting human-readable if-then rules for strategy analysis.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

import torch

from _01_simulator import engine, rules, state
from _01_simulator.action_space import (
    ACTION_DIM,
    index_to_action,
    legal_action_mask_tensor,
)
from _01_simulator.observations import (
    OBSERVATION_DIM,
    build_observation,
    species_index,
    state_to_tensor,
)
from _02_agents.neural.agent import load_neural_agent
from _02_agents.neural.utils import greedy_action


# Species names (sorted alphabetically, matching observations.py)
SPECIES_NAMES = sorted([s for s in rules.SPECIES.keys() if s != "unknown"])
SPECIES_TO_IDX = {name: idx for idx, name in enumerate(SPECIES_NAMES)}


@dataclass
class InterpretableFeatures:
    """Human-interpretable features derived from game state."""

    # Cards in hand (one-hot for each of 12 species)
    hand_lion: int
    hand_hippo: int
    hand_crocodile: int
    hand_snake: int
    hand_giraffe: int
    hand_zebra: int
    hand_seal: int
    hand_chameleon: int
    hand_monkey: int
    hand_kangaroo: int
    hand_parrot: int
    hand_skunk: int

    # Queue information
    queue_length: int
    queue_has_own_cards: int
    queue_has_opponent_cards: int
    own_cards_in_queue: int
    opponent_cards_in_queue: int

    # Queue composition (count of each species)
    queue_lion: int
    queue_hippo: int
    queue_crocodile: int
    queue_snake: int
    queue_giraffe: int
    queue_zebra: int
    queue_seal: int
    queue_chameleon: int
    queue_monkey: int
    queue_kangaroo: int
    queue_parrot: int
    queue_skunk: int

    # Front/back of queue
    front_species_idx: int  # -1 if empty
    back_species_idx: int  # -1 if empty
    front_is_own: int  # 1 if own, 0 if opponent, -1 if empty
    back_is_own: int

    # Turn and scoring
    turn_number: int
    own_score: int
    opponent_score: int
    score_difference: int  # own - opponent

    # Beasty Bar status
    beasty_bar_count: int
    own_in_bar: int
    opponent_in_bar: int

    # That's It status
    thats_it_count: int
    own_in_thats_it: int
    opponent_in_thats_it: int

    # Strategic indicators
    has_monkey_pair: int  # Do I have monkey AND there's another monkey in queue?
    can_use_lion: int  # Do I have lion AND queue not empty?
    can_use_parrot: int  # Do I have parrot AND queue not empty?
    has_high_strength_in_hand: int  # strength >= 8
    has_low_strength_in_hand: int  # strength <= 4
    queue_avg_strength: float
    queue_max_strength: int
    queue_min_strength: int

    # Deck/hand counts
    own_deck_remaining: int
    opponent_deck_remaining: int
    own_hand_size: int
    opponent_hand_size: int

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for sklearn."""
        return np.array([
            self.hand_lion, self.hand_hippo, self.hand_crocodile, self.hand_snake,
            self.hand_giraffe, self.hand_zebra, self.hand_seal, self.hand_chameleon,
            self.hand_monkey, self.hand_kangaroo, self.hand_parrot, self.hand_skunk,
            self.queue_length, self.queue_has_own_cards, self.queue_has_opponent_cards,
            self.own_cards_in_queue, self.opponent_cards_in_queue,
            self.queue_lion, self.queue_hippo, self.queue_crocodile, self.queue_snake,
            self.queue_giraffe, self.queue_zebra, self.queue_seal, self.queue_chameleon,
            self.queue_monkey, self.queue_kangaroo, self.queue_parrot, self.queue_skunk,
            self.front_species_idx, self.back_species_idx,
            self.front_is_own, self.back_is_own,
            self.turn_number, self.own_score, self.opponent_score, self.score_difference,
            self.beasty_bar_count, self.own_in_bar, self.opponent_in_bar,
            self.thats_it_count, self.own_in_thats_it, self.opponent_in_thats_it,
            self.has_monkey_pair, self.can_use_lion, self.can_use_parrot,
            self.has_high_strength_in_hand, self.has_low_strength_in_hand,
            self.queue_avg_strength, self.queue_max_strength, self.queue_min_strength,
            self.own_deck_remaining, self.opponent_deck_remaining,
            self.own_hand_size, self.opponent_hand_size,
        ], dtype=np.float32)


def get_feature_names() -> list[str]:
    """Get names of all interpretable features."""
    return [
        "hand_lion", "hand_hippo", "hand_crocodile", "hand_snake",
        "hand_giraffe", "hand_zebra", "hand_seal", "hand_chameleon",
        "hand_monkey", "hand_kangaroo", "hand_parrot", "hand_skunk",
        "queue_length", "queue_has_own_cards", "queue_has_opponent_cards",
        "own_cards_in_queue", "opponent_cards_in_queue",
        "queue_lion", "queue_hippo", "queue_crocodile", "queue_snake",
        "queue_giraffe", "queue_zebra", "queue_seal", "queue_chameleon",
        "queue_monkey", "queue_kangaroo", "queue_parrot", "queue_skunk",
        "front_species_idx", "back_species_idx",
        "front_is_own", "back_is_own",
        "turn_number", "own_score", "opponent_score", "score_difference",
        "beasty_bar_count", "own_in_bar", "opponent_in_bar",
        "thats_it_count", "own_in_thats_it", "opponent_in_thats_it",
        "has_monkey_pair", "can_use_lion", "can_use_parrot",
        "has_high_strength_in_hand", "has_low_strength_in_hand",
        "queue_avg_strength", "queue_max_strength", "queue_min_strength",
        "own_deck_remaining", "opponent_deck_remaining",
        "own_hand_size", "opponent_hand_size",
    ]


def extract_features(game_state: state.State, perspective: int) -> InterpretableFeatures:
    """Extract interpretable features from a game state."""
    player_state = game_state.players[perspective]
    opponent_state = game_state.players[1 - perspective]
    zones = game_state.zones

    # Count species in hand
    hand_species = Counter(card.species for card in player_state.hand)

    # Queue analysis
    queue = zones.queue
    queue_species = Counter(card.species for card in queue)
    own_in_queue = sum(1 for c in queue if c.owner == perspective)
    opp_in_queue = len(queue) - own_in_queue

    # Front/back of queue
    front_species_idx = -1
    front_is_own = -1
    back_species_idx = -1
    back_is_own = -1
    if queue:
        front_card = queue[0]
        front_species_idx = SPECIES_TO_IDX.get(front_card.species, -1)
        front_is_own = 1 if front_card.owner == perspective else 0
        back_card = queue[-1]
        back_species_idx = SPECIES_TO_IDX.get(back_card.species, -1)
        back_is_own = 1 if back_card.owner == perspective else 0

    # Score calculation
    own_score = sum(c.points for c in zones.beasty_bar if c.owner == perspective)
    opp_score = sum(c.points for c in zones.beasty_bar if c.owner != perspective)

    # Beasty bar counts
    own_in_bar = sum(1 for c in zones.beasty_bar if c.owner == perspective)
    opp_in_bar = len(zones.beasty_bar) - own_in_bar

    # That's it counts
    own_in_thats_it = sum(1 for c in zones.thats_it if c.owner == perspective)
    opp_in_thats_it = len(zones.thats_it) - own_in_thats_it

    # Strategic indicators
    has_monkey = hand_species.get("monkey", 0) > 0
    queue_has_monkey = queue_species.get("monkey", 0) > 0
    has_monkey_pair = 1 if has_monkey and queue_has_monkey else 0

    has_lion = hand_species.get("lion", 0) > 0
    can_use_lion = 1 if has_lion and len(queue) > 0 else 0

    has_parrot = hand_species.get("parrot", 0) > 0
    can_use_parrot = 1 if has_parrot and len(queue) > 0 else 0

    # Strength analysis
    hand_strengths = [rules.SPECIES[c.species].strength for c in player_state.hand]
    has_high = 1 if any(s >= 8 for s in hand_strengths) else 0
    has_low = 1 if any(s <= 4 for s in hand_strengths) else 0

    queue_strengths = [c.strength for c in queue]
    queue_avg = np.mean(queue_strengths) if queue_strengths else 0.0
    queue_max = max(queue_strengths) if queue_strengths else 0
    queue_min = min(queue_strengths) if queue_strengths else 0

    return InterpretableFeatures(
        hand_lion=hand_species.get("lion", 0),
        hand_hippo=hand_species.get("hippo", 0),
        hand_crocodile=hand_species.get("crocodile", 0),
        hand_snake=hand_species.get("snake", 0),
        hand_giraffe=hand_species.get("giraffe", 0),
        hand_zebra=hand_species.get("zebra", 0),
        hand_seal=hand_species.get("seal", 0),
        hand_chameleon=hand_species.get("chameleon", 0),
        hand_monkey=hand_species.get("monkey", 0),
        hand_kangaroo=hand_species.get("kangaroo", 0),
        hand_parrot=hand_species.get("parrot", 0),
        hand_skunk=hand_species.get("skunk", 0),
        queue_length=len(queue),
        queue_has_own_cards=1 if own_in_queue > 0 else 0,
        queue_has_opponent_cards=1 if opp_in_queue > 0 else 0,
        own_cards_in_queue=own_in_queue,
        opponent_cards_in_queue=opp_in_queue,
        queue_lion=queue_species.get("lion", 0),
        queue_hippo=queue_species.get("hippo", 0),
        queue_crocodile=queue_species.get("crocodile", 0),
        queue_snake=queue_species.get("snake", 0),
        queue_giraffe=queue_species.get("giraffe", 0),
        queue_zebra=queue_species.get("zebra", 0),
        queue_seal=queue_species.get("seal", 0),
        queue_chameleon=queue_species.get("chameleon", 0),
        queue_monkey=queue_species.get("monkey", 0),
        queue_kangaroo=queue_species.get("kangaroo", 0),
        queue_parrot=queue_species.get("parrot", 0),
        queue_skunk=queue_species.get("skunk", 0),
        front_species_idx=front_species_idx,
        back_species_idx=back_species_idx,
        front_is_own=front_is_own,
        back_is_own=back_is_own,
        turn_number=game_state.turn,
        own_score=own_score,
        opponent_score=opp_score,
        score_difference=own_score - opp_score,
        beasty_bar_count=len(zones.beasty_bar),
        own_in_bar=own_in_bar,
        opponent_in_bar=opp_in_bar,
        thats_it_count=len(zones.thats_it),
        own_in_thats_it=own_in_thats_it,
        opponent_in_thats_it=opp_in_thats_it,
        has_monkey_pair=has_monkey_pair,
        can_use_lion=can_use_lion,
        can_use_parrot=can_use_parrot,
        has_high_strength_in_hand=has_high,
        has_low_strength_in_hand=has_low,
        queue_avg_strength=float(queue_avg),
        queue_max_strength=queue_max,
        queue_min_strength=queue_min,
        own_deck_remaining=len(player_state.deck),
        opponent_deck_remaining=len(opponent_state.deck),
        own_hand_size=len(player_state.hand),
        opponent_hand_size=len(opponent_state.hand),
    )


def action_to_simplified_label(action_idx: int, game_state: state.State, perspective: int) -> str:
    """Convert action index to simplified label (species played)."""
    action = index_to_action(action_idx)
    hand_idx = action.hand_index
    player_hand = game_state.players[perspective].hand
    if hand_idx < len(player_hand):
        return player_hand[hand_idx].species
    return "unknown"


def generate_training_data(
    agent,
    num_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate training data by playing games with the neural network.

    Returns:
        X: Feature matrix (num_samples, num_features)
        y: Labels (num_samples,) - species played
        species_labels: List of unique species labels
    """
    random.seed(seed)
    np.random.seed(seed)

    X_list = []
    y_list = []

    game_seeds = list(range(seed, seed + num_samples * 2))  # Extra seeds in case
    sample_count = 0
    game_idx = 0

    device = agent.device

    print(f"Generating {num_samples} samples...")

    while sample_count < num_samples and game_idx < len(game_seeds):
        game_seed = game_seeds[game_idx]
        game_idx += 1

        try:
            game = state.initial_state(seed=game_seed, starting_player=game_seed % 2)
        except Exception:
            continue

        while not engine.is_terminal(game):
            perspective = game.active_player
            legal = list(engine.legal_actions(game, perspective))

            if not legal:
                break

            # Get features
            features = extract_features(game, perspective)

            # Get neural network decision
            obs_tensor = torch.from_numpy(state_to_tensor(game, perspective)).to(device)
            mask_tensor = torch.from_numpy(legal_action_mask_tensor(game, perspective)).to(device)

            with torch.no_grad():
                policy_logits, _ = agent.model(obs_tensor, mask_tensor)
                action_idx = greedy_action(policy_logits, mask_tensor)

            # Get species label
            species_label = action_to_simplified_label(action_idx, game, perspective)

            X_list.append(features.to_array())
            y_list.append(species_label)
            sample_count += 1

            if sample_count >= num_samples:
                break

            # Apply action
            action = index_to_action(action_idx)
            try:
                game = engine.step(game, action)
            except Exception:
                break

        if sample_count % 500 == 0:
            print(f"  Generated {sample_count}/{num_samples} samples...")

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def train_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 7,
) -> tuple[DecisionTreeClassifier, float]:
    """Train a decision tree classifier.

    Returns:
        clf: Trained classifier
        accuracy: Training accuracy (fidelity to NN)
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )
    clf.fit(X, y)

    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)

    return clf, accuracy


def extract_decision_rules(
    clf: DecisionTreeClassifier,
    feature_names: list[str],
    max_rules: int = 30,
) -> str:
    """Extract human-readable decision rules from the tree."""
    tree_rules = export_text(clf, feature_names=feature_names, max_depth=10)
    return tree_rules


def get_feature_importance(
    clf: DecisionTreeClassifier,
    feature_names: list[str],
    top_k: int = 15,
) -> list[tuple[str, float]]:
    """Get top-k most important features."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    result = []
    for i in indices[:top_k]:
        if importances[i] > 0.001:  # Filter out negligible features
            result.append((feature_names[i], float(importances[i])))

    return result


def interpret_rules_to_insights(tree_rules: str, feature_importance: list) -> list[str]:
    """Convert raw tree rules to strategic insights."""
    insights = []

    # Analyze feature importance
    top_features = [f[0] for f in feature_importance[:10]]

    if any("hand_" in f for f in top_features):
        hand_features = [f for f in top_features if f.startswith("hand_")]
        if hand_features:
            species = [f.replace("hand_", "") for f in hand_features]
            insights.append(f"Card availability is crucial: having {', '.join(species[:3])} strongly influences decisions")

    if "queue_length" in top_features:
        insights.append("Queue length is a key decision factor - tactics change based on how full the queue is")

    if any("score" in f for f in top_features):
        insights.append("Score differential affects card choices - the AI plays differently when ahead vs behind")

    if "has_monkey_pair" in top_features:
        insights.append("Monkey pair detection: the AI actively looks for monkey synergies")

    if "queue_avg_strength" in top_features or "queue_max_strength" in top_features:
        insights.append("Queue strength composition matters - high-strength queues trigger different plays")

    if any("front_" in f for f in top_features):
        insights.append("Front card position is important - indicates queue-manipulation strategies")

    if "turn_number" in top_features:
        insights.append("Game phase matters - early, mid, and late game have different optimal plays")

    if "can_use_parrot" in top_features:
        insights.append("Parrot usage is conditional - the AI waits for the right moment to remove opponents")

    if any("opponent" in f for f in top_features):
        insights.append("Opponent awareness - the AI considers opponent's queue position and card counts")

    return insights


def analyze_species_decision_patterns(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """Analyze when each species is preferred."""
    patterns = {}

    for species in SPECIES_NAMES:
        mask = y == species
        if not mask.any():
            continue

        species_X = X[mask]
        other_X = X[~mask]

        # Find features that differ significantly
        species_means = species_X.mean(axis=0)
        other_means = other_X.mean(axis=0)
        diff = species_means - other_means

        # Get top differentiating features
        top_diff_indices = np.argsort(np.abs(diff))[::-1][:5]
        pattern_info = []
        for idx in top_diff_indices:
            if abs(diff[idx]) > 0.05:
                direction = "higher" if diff[idx] > 0 else "lower"
                pattern_info.append((feature_names[idx], diff[idx], direction))

        patterns[species] = pattern_info

    return patterns


def main():
    """Run the decision tree analysis."""
    checkpoint_path = Path("/Users/p/Desktop/v/experiments/beastybar/checkpoints/v4/final.pt")
    output_path = Path("/Users/p/Desktop/v/experiments/beastybar/_05_analysis/03_decision_rules.md")

    print("=" * 60)
    print("Decision Tree Analysis of Neural Network Strategy")
    print("=" * 60)

    # Load the neural agent
    print("\n1. Loading neural network from checkpoint...")
    agent = load_neural_agent(checkpoint_path, mode="greedy")
    print(f"   Model loaded on device: {agent.device}")

    # Generate training data (increased samples for better fidelity)
    print("\n2. Generating game states and NN decisions...")
    X, y = generate_training_data(agent, num_samples=10000, seed=42)
    print(f"   Generated {len(X)} samples")
    print(f"   Species distribution: {Counter(y).most_common()}")

    feature_names = get_feature_names()

    # Train decision trees with different depths
    print("\n3. Training decision trees...")
    results = {}
    for depth in [5, 6, 7, 8, 10, 12]:
        clf, accuracy = train_decision_tree(X, y, max_depth=depth)
        results[depth] = (clf, accuracy)
        print(f"   Depth {depth}: Fidelity = {accuracy:.1%}")

    # Use depth 8 for main analysis (good balance of accuracy and interpretability)
    best_depth = 8
    clf, fidelity = results[best_depth]

    # Extract rules
    print("\n4. Extracting decision rules...")
    tree_rules = extract_decision_rules(clf, feature_names)

    # Get feature importance
    print("\n5. Analyzing feature importance...")
    importance = get_feature_importance(clf, feature_names, top_k=20)
    for feat, imp in importance[:10]:
        print(f"   {feat}: {imp:.3f}")

    # Generate insights
    print("\n6. Generating strategic insights...")
    insights = interpret_rules_to_insights(tree_rules, importance)

    # Analyze species patterns
    print("\n7. Analyzing species-specific decision patterns...")
    species_patterns = analyze_species_decision_patterns(X, y, feature_names)

    # Write report
    print("\n8. Writing report...")

    report = f"""# Neural Network Decision Tree Analysis

## Overview

This analysis trains an interpretable decision tree to mimic the neural network's
card selection decisions, extracting human-readable strategic rules.

**Model**: `checkpoints/v4/final.pt`
**Training Samples**: {len(X):,}
**Decision Tree Depth**: {best_depth}
**Fidelity Score**: {fidelity:.1%}

The fidelity score measures how often the decision tree makes the same choice
as the neural network - a {fidelity:.1%} fidelity means the tree captures the
NN's decision-making pattern well.

---

## Species Selection Distribution

The neural network's greedy policy selected cards in the following proportions:

| Species | Count | Percentage |
|---------|-------|------------|
"""

    species_counts = Counter(y)
    total = len(y)
    for species, count in species_counts.most_common():
        pct = count / total * 100
        report += f"| {species} | {count:,} | {pct:.1f}% |\n"

    report += f"""

---

## Feature Importance Ranking

The decision tree identified these features as most important for predicting
which card the neural network will play:

| Rank | Feature | Importance |
|------|---------|------------|
"""

    for i, (feat, imp) in enumerate(importance, 1):
        report += f"| {i} | `{feat}` | {imp:.3f} |\n"

    report += f"""

---

## Key Strategic Insights

Based on the decision tree analysis, here are the main strategic principles
the neural network has learned:

"""

    for i, insight in enumerate(insights, 1):
        report += f"{i}. **{insight}**\n\n"

    report += """
### Interpretation of Top Features

"""

    # Detailed interpretation of top features
    for feat, imp in importance[:5]:
        if feat.startswith("hand_"):
            species = feat.replace("hand_", "")
            report += f"- **{feat}** (importance: {imp:.3f}): Having {species} in hand is a major decision driver. "
            if species == "lion":
                report += "Lion (strength 12) dominates the queue with its roar ability.\n"
            elif species == "monkey":
                report += "Monkey pairs can eliminate high-strength opponents.\n"
            elif species == "parrot":
                report += "Parrot provides targeted removal to That's It.\n"
            elif species == "giraffe":
                report += "Giraffe's recurring ability to move forward is valuable.\n"
            elif species == "hippo":
                report += "Hippo has recurring push ability, strong for queue control.\n"
            elif species == "zebra":
                report += "Zebra provides 4 points and stays permanently once in the bar.\n"
            elif species == "crocodile":
                report += "Crocodile's recurring eat ability threatens weaker cards.\n"
            else:
                report += f"{species.title()} has specific tactical value.\n"
        elif feat == "queue_length":
            report += f"- **queue_length** (importance: {imp:.3f}): The number of cards in queue determines available tactics and timing.\n"
        elif feat == "score_difference":
            report += f"- **score_difference** (importance: {imp:.3f}): The AI plays differently when ahead vs behind in score.\n"
        elif feat == "can_use_parrot":
            report += f"- **can_use_parrot** (importance: {imp:.3f}): Parrot availability combined with targets in queue triggers targeted removal.\n"
        elif "queue_" in feat:
            species = feat.replace("queue_", "")
            report += f"- **{feat}** (importance: {imp:.3f}): Presence of {species} in queue affects card selection.\n"
        else:
            report += f"- **{feat}** (importance: {imp:.3f}): This game state aspect influences card choices.\n"

    # Add species-specific patterns section
    report += """

---

## Species-Specific Decision Patterns

When does the neural network prefer to play each species? Here's what makes each
card choice distinctive:

"""

    for species in ["lion", "hippo", "crocodile", "giraffe", "zebra", "monkey", "parrot", "skunk"]:
        if species in species_patterns and species_patterns[species]:
            report += f"### {species.title()}\n"
            patterns = species_patterns[species]
            for feat_name, diff_val, direction in patterns[:3]:
                if "hand_" in feat_name and feat_name.replace("hand_", "") == species:
                    continue  # Skip the obvious "has this card" pattern
                report += f"- When `{feat_name}` is {direction} than average (diff: {diff_val:+.2f})\n"
            report += "\n"

    report += f"""

---

## Decision Tree Rules (Simplified)

Below is a simplified representation of the decision tree's rules. The tree
makes decisions by checking conditions from the root down to the leaves.

```
{tree_rules[:3000]}
```

*Note: Full tree rules truncated for readability. The tree has {best_depth} levels
of depth.*

---

## Fidelity Comparison by Tree Depth

| Max Depth | Fidelity | Notes |
|-----------|----------|-------|
"""

    for depth in sorted(results.keys()):
        _, acc = results[depth]
        note = "Good balance" if depth == 7 else ("Simpler" if depth < 7 else "More complex")
        report += f"| {depth} | {acc:.1%} | {note} |\n"

    report += f"""

---

## Methodology

1. **Data Generation**: Played {len(X):,} game states using the trained neural
   network with greedy action selection.

2. **Feature Engineering**: Extracted {len(feature_names)} interpretable features
   including:
   - Cards in hand (one-hot for each species)
   - Queue length and composition
   - Score differential
   - Strategic indicators (monkey pairs, blocking opportunities)
   - Game phase (turn number)

3. **Decision Tree Training**: Trained sklearn DecisionTreeClassifier with
   max_depth={best_depth}, min_samples_split=20, min_samples_leaf=10.

4. **Rule Extraction**: Used sklearn's export_text to extract if-then rules.

---

## Limitations

- The decision tree is a simplified approximation of the neural network
- Complex non-linear patterns may not be fully captured
- The {1-fidelity:.1%} of decisions where tree differs from NN represent
  nuanced situations the tree cannot express
- Feature engineering choices affect what patterns can be detected

---

*Generated by decision_tree_analysis.py*
"""

    # Write the report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\n   Report written to: {output_path}")
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
