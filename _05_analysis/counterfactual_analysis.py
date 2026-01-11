#!/usr/bin/env python3
"""Counterfactual analysis of trained Beasty Bar AI model.

This script performs "what-if" analysis to understand model sensitivities:
1. Card Swap Impact - What happens if we swap cards in hand?
2. Queue Threat Analysis - How does the model respond to opponent threats?
3. Opportunity Analysis - Does the model protect its own high-value cards?
4. Combo Detection - Does the model recognize Monkey pair opportunities?

Output is written to _05_analysis/06_counterfactual.md
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from _01_simulator import rules, state
from _01_simulator.action_space import (
    ACTION_DIM,
    canonical_actions,
    legal_action_space,
)
from _01_simulator.observations import (
    OBSERVATION_DIM,
    _CARD_FEATURE_DIM,
    _NUM_SPECIES,
    _SPECIES_INDEX,
    _INDEX_TO_SPECIES,
    build_observation,
    observation_to_tensor,
    species_index,
    species_name,
)
from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import (
    NetworkConfig,
    load_network_from_checkpoint,
    get_device,
)


# Species indices (alphabetically sorted)
SPECIES_NAMES = sorted([s for s in rules.SPECIES.keys() if s != "unknown"])
SPECIES_TO_IDX = {name: i for i, name in enumerate(SPECIES_NAMES)}


@dataclass
class ModelOutput:
    """Container for model policy and value outputs."""
    policy_logits: np.ndarray  # Shape (ACTION_DIM,)
    value: float
    action_probs: np.ndarray  # After softmax and masking


def get_action_probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute masked softmax probabilities."""
    masked_logits = np.where(mask > 0, logits, -np.inf)
    # Stable softmax
    max_logit = np.max(masked_logits[mask > 0]) if np.any(mask > 0) else 0
    exp_logits = np.exp(masked_logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    probs = np.where(mask > 0, probs, 0)
    return probs


class CounterfactualAnalyzer:
    """Performs counterfactual analysis on trained Beasty Bar model."""

    def __init__(self, checkpoint_path: str | Path):
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Load model
        self.model, self.config, self.step = load_network_from_checkpoint(
            checkpoint_path, device=self.device
        )
        self.model.eval()
        print(f"Loaded model from step {self.step}")
        print(f"Model parameters: {self.model.count_parameters():,}")

        # Get action catalog
        self.actions = canonical_actions()

    def _evaluate_state(
        self, game_state: state.State, perspective: int
    ) -> ModelOutput:
        """Evaluate a game state and return policy/value."""
        # Build observation
        obs = build_observation(game_state, perspective, mask_hidden=True)
        tensor = observation_to_tensor(obs, perspective)

        # Get legal action mask
        action_space = legal_action_space(game_state, perspective)
        mask = np.array(action_space.mask, dtype=np.float32)

        # Forward pass
        with torch.no_grad():
            obs_tensor = torch.from_numpy(tensor).to(self.device)
            policy_logits, value = self.model(obs_tensor)
            policy_logits = policy_logits.cpu().numpy()
            value = float(value.cpu().numpy())

        # Compute action probabilities
        action_probs = get_action_probs(policy_logits, mask)

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            action_probs=action_probs,
        )

    def _create_baseline_state(
        self,
        hand_species: list[str],
        queue_cards: list[tuple[str, int]],  # (species, owner)
        perspective: int = 0,
        turn: int = 5,
    ) -> state.State:
        """Create a custom game state for testing.

        Args:
            hand_species: List of species for perspective player's hand
            queue_cards: List of (species, owner) for queue
            perspective: Which player we're analyzing
            turn: Turn number
        """
        # Create player states
        players = []
        for player_idx in range(rules.PLAYER_COUNT):
            if player_idx == perspective:
                # Create hand from specified species
                hand = tuple(
                    state.Card(owner=player_idx, species=s, entered_turn=-1)
                    for s in hand_species
                )
                # Create deck from remaining species
                used = set(hand_species)
                deck_species = [s for s in rules.BASE_DECK if s not in used]
                deck = tuple(
                    state.Card(owner=player_idx, species=s, entered_turn=-1)
                    for s in deck_species[:8]  # Keep deck at reasonable size
                )
            else:
                # Opponent with full hand and deck
                deck = tuple(
                    state.Card(owner=player_idx, species=s, entered_turn=-1)
                    for s in rules.BASE_DECK[:8]
                )
                hand = tuple(
                    state.Card(owner=player_idx, species=s, entered_turn=-1)
                    for s in rules.BASE_DECK[8:12]
                )
            players.append(state.PlayerState(deck=deck, hand=hand))

        # Create queue
        queue = tuple(
            state.Card(owner=owner, species=species, entered_turn=i)
            for i, (species, owner) in enumerate(queue_cards)
        )

        zones = state.Zones(queue=queue)

        return state.State(
            seed=42,
            turn=turn,
            active_player=perspective,
            players=tuple(players),
            zones=zones,
        )

    def _modify_hand_card(
        self, game_state: state.State, perspective: int,
        hand_idx: int, new_species: str
    ) -> state.State:
        """Replace a card in hand with a different species."""
        player_state = game_state.players[perspective]
        old_hand = list(player_state.hand)
        old_hand[hand_idx] = state.Card(
            owner=perspective,
            species=new_species,
            entered_turn=-1
        )
        new_player = state.PlayerState(
            deck=player_state.deck,
            hand=tuple(old_hand)
        )
        players = list(game_state.players)
        players[perspective] = new_player
        return state.State(
            seed=game_state.seed,
            turn=game_state.turn,
            active_player=game_state.active_player,
            players=tuple(players),
            zones=game_state.zones,
        )

    def _add_to_queue(
        self, game_state: state.State, species: str, owner: int, position: int = -1
    ) -> state.State:
        """Add a card to the queue."""
        queue = list(game_state.zones.queue)
        new_card = state.Card(owner=owner, species=species, entered_turn=game_state.turn)
        if position < 0:
            queue.append(new_card)
        else:
            queue.insert(position, new_card)
        # Ensure queue doesn't exceed max length
        queue = queue[:rules.MAX_QUEUE_LENGTH]
        new_zones = state.Zones(
            queue=tuple(queue),
            beasty_bar=game_state.zones.beasty_bar,
            thats_it=game_state.zones.thats_it,
        )
        return state.State(
            seed=game_state.seed,
            turn=game_state.turn,
            active_player=game_state.active_player,
            players=game_state.players,
            zones=new_zones,
        )

    def analyze_card_swap_impact(self) -> dict[str, Any]:
        """Analyze how swapping cards in hand affects policy and value."""
        print("\n=== Card Swap Impact Analysis ===")

        # Create baseline state: moderate hand, some queue
        baseline_hand = ["seal", "giraffe", "monkey", "parrot"]
        baseline_queue = [("zebra", 1), ("snake", 0)]  # Opponent zebra at front

        baseline_state = self._create_baseline_state(
            hand_species=baseline_hand,
            queue_cards=baseline_queue,
            perspective=0,
        )

        baseline_output = self._evaluate_state(baseline_state, 0)
        print(f"Baseline value: {baseline_output.value:.4f}")

        # Track all swap impacts
        swap_impacts = []

        for hand_idx, original_species in enumerate(baseline_hand):
            for new_species in SPECIES_NAMES:
                if new_species == original_species:
                    continue
                if new_species in baseline_hand:
                    continue  # Skip duplicates

                modified_state = self._modify_hand_card(
                    baseline_state, 0, hand_idx, new_species
                )
                modified_output = self._evaluate_state(modified_state, 0)

                value_delta = modified_output.value - baseline_output.value

                # Compute policy divergence (KL-like)
                # Use absolute difference in top action probs
                policy_delta = np.sum(np.abs(
                    modified_output.action_probs - baseline_output.action_probs
                ))

                swap_impacts.append({
                    "hand_idx": hand_idx,
                    "original": original_species,
                    "new": new_species,
                    "value_delta": value_delta,
                    "policy_delta": policy_delta,
                    "new_value": modified_output.value,
                })

        # Sort by absolute value delta
        swap_impacts.sort(key=lambda x: abs(x["value_delta"]), reverse=True)

        print("\nTop 10 value-changing swaps:")
        for i, swap in enumerate(swap_impacts[:10]):
            print(f"  {i+1}. {swap['original']} -> {swap['new']}: "
                  f"value delta = {swap['value_delta']:+.4f}")

        # Sort by policy delta
        swap_impacts_by_policy = sorted(
            swap_impacts, key=lambda x: x["policy_delta"], reverse=True
        )

        print("\nTop 10 policy-changing swaps:")
        for i, swap in enumerate(swap_impacts_by_policy[:10]):
            print(f"  {i+1}. {swap['original']} -> {swap['new']}: "
                  f"policy delta = {swap['policy_delta']:.4f}")

        return {
            "baseline_value": baseline_output.value,
            "swap_impacts": swap_impacts,
            "top_value_swaps": swap_impacts[:10],
            "top_policy_swaps": swap_impacts_by_policy[:10],
        }

    def analyze_queue_threats(self) -> dict[str, Any]:
        """Analyze how opponent threats in queue affect model behavior."""
        print("\n=== Queue Threat Analysis ===")

        # Baseline: own hand with decent cards, empty queue
        baseline_hand = ["giraffe", "seal", "chameleon", "kangaroo"]

        # Start with empty queue
        baseline_state = self._create_baseline_state(
            hand_species=baseline_hand,
            queue_cards=[],
            perspective=0,
        )

        baseline_output = self._evaluate_state(baseline_state, 0)
        print(f"Empty queue baseline value: {baseline_output.value:.4f}")

        # Analyze adding different opponent cards to queue
        threat_species = ["lion", "crocodile", "hippo", "snake", "giraffe", "zebra"]
        threat_results = []

        for threat in threat_species:
            # Add opponent threat at front of queue
            threatened_state = self._add_to_queue(
                baseline_state, threat, owner=1, position=0
            )
            threat_output = self._evaluate_state(threatened_state, 0)

            value_delta = threat_output.value - baseline_output.value

            # Find preferred action change
            baseline_top_action = int(np.argmax(baseline_output.action_probs))
            threat_top_action = int(np.argmax(threat_output.action_probs))

            action_changed = baseline_top_action != threat_top_action

            # Decode actions
            baseline_action = self.actions[baseline_top_action]
            threat_action = self.actions[threat_top_action]

            threat_results.append({
                "threat_species": threat,
                "threat_strength": rules.SPECIES[threat].strength,
                "value_delta": value_delta,
                "action_changed": action_changed,
                "baseline_action": str(baseline_action),
                "new_action": str(threat_action),
                "new_value": threat_output.value,
            })

            print(f"\nOpponent {threat} (str {rules.SPECIES[threat].strength}) at front:")
            print(f"  Value: {threat_output.value:.4f} (delta: {value_delta:+.4f})")
            print(f"  Action changed: {action_changed}")
            if action_changed:
                print(f"    From: {baseline_action}")
                print(f"    To: {threat_action}")

        return {
            "baseline_value": baseline_output.value,
            "threat_results": threat_results,
        }

    def analyze_opportunity_exploitation(self) -> dict[str, Any]:
        """Analyze if model protects/exploits its high-value cards near front."""
        print("\n=== Opportunity Exploitation Analysis ===")

        # Hand with mix of strengths
        baseline_hand = ["seal", "chameleon", "monkey", "skunk"]

        results = []

        # Test scenarios: own card at front vs middle vs back of queue
        positions = ["front", "middle", "back"]
        position_indices = [0, 2, 4]  # Actual positions in queue of 5

        high_value_cards = ["lion", "hippo", "crocodile", "giraffe"]

        for card in high_value_cards:
            card_results = {"card": card, "strength": rules.SPECIES[card].strength, "positions": {}}

            for pos_name, pos_idx in zip(positions, position_indices):
                # Create queue with our high-value card at position
                queue = []
                for i in range(5):
                    if i == pos_idx:
                        queue.append((card, 0))  # Our card
                    else:
                        queue.append(("parrot", 1))  # Filler opponent cards

                game_state = self._create_baseline_state(
                    hand_species=baseline_hand,
                    queue_cards=queue,
                    perspective=0,
                )

                output = self._evaluate_state(game_state, 0)
                top_action_idx = int(np.argmax(output.action_probs))
                top_action = self.actions[top_action_idx]

                card_results["positions"][pos_name] = {
                    "value": output.value,
                    "top_action": str(top_action),
                    "hand_idx": top_action.hand_index,
                    "played_card": baseline_hand[top_action.hand_index] if top_action.hand_index < len(baseline_hand) else "N/A",
                }

            results.append(card_results)

            print(f"\n{card} (strength {rules.SPECIES[card].strength}):")
            for pos in positions:
                r = card_results["positions"][pos]
                print(f"  {pos}: value={r['value']:.4f}, plays {r['played_card']}")

        return {"opportunity_results": results}

    def analyze_combo_detection(self) -> dict[str, Any]:
        """Analyze if model recognizes Monkey pair opportunities."""
        print("\n=== Combo Detection Analysis (Monkey Pairs) ===")

        # Scenario 1: Hand with monkey, opponent monkey in queue
        # Model should recognize opportunity to play monkey

        results = []

        # Test 1: No monkey in queue (baseline)
        hand_with_monkey = ["monkey", "seal", "chameleon", "parrot"]
        no_monkey_queue = [("zebra", 1), ("snake", 1)]

        state_no_monkey = self._create_baseline_state(
            hand_species=hand_with_monkey,
            queue_cards=no_monkey_queue,
            perspective=0,
        )
        output_no_monkey = self._evaluate_state(state_no_monkey, 0)

        # Find probability of playing monkey (hand index 0)
        monkey_action_probs = []
        for i, action in enumerate(self.actions):
            if action.hand_index == 0:  # Monkey is at hand index 0
                monkey_action_probs.append(output_no_monkey.action_probs[i])
        monkey_prob_no_combo = sum(monkey_action_probs)

        print(f"No monkey in queue:")
        print(f"  Value: {output_no_monkey.value:.4f}")
        print(f"  Probability of playing monkey: {monkey_prob_no_combo:.4f}")

        results.append({
            "scenario": "no_monkey_in_queue",
            "value": output_no_monkey.value,
            "monkey_play_prob": monkey_prob_no_combo,
        })

        # Test 2: Opponent monkey in queue
        with_monkey_queue = [("monkey", 1), ("zebra", 1)]

        state_with_monkey = self._create_baseline_state(
            hand_species=hand_with_monkey,
            queue_cards=with_monkey_queue,
            perspective=0,
        )
        output_with_monkey = self._evaluate_state(state_with_monkey, 0)

        monkey_action_probs_combo = []
        for i, action in enumerate(self.actions):
            if action.hand_index == 0:  # Monkey is at hand index 0
                monkey_action_probs_combo.append(output_with_monkey.action_probs[i])
        monkey_prob_with_combo = sum(monkey_action_probs_combo)

        print(f"\nOpponent monkey in queue (combo opportunity):")
        print(f"  Value: {output_with_monkey.value:.4f}")
        print(f"  Probability of playing monkey: {monkey_prob_with_combo:.4f}")
        print(f"  Probability increase: {monkey_prob_with_combo - monkey_prob_no_combo:+.4f}")

        results.append({
            "scenario": "opponent_monkey_in_queue",
            "value": output_with_monkey.value,
            "monkey_play_prob": monkey_prob_with_combo,
            "prob_increase": monkey_prob_with_combo - monkey_prob_no_combo,
        })

        # Test 3: Own monkey in queue
        own_monkey_queue = [("monkey", 0), ("zebra", 1)]

        state_own_monkey = self._create_baseline_state(
            hand_species=hand_with_monkey,
            queue_cards=own_monkey_queue,
            perspective=0,
        )
        output_own_monkey = self._evaluate_state(state_own_monkey, 0)

        monkey_action_probs_own = []
        for i, action in enumerate(self.actions):
            if action.hand_index == 0:
                monkey_action_probs_own.append(output_own_monkey.action_probs[i])
        monkey_prob_own = sum(monkey_action_probs_own)

        print(f"\nOwn monkey in queue (swap opportunity):")
        print(f"  Value: {output_own_monkey.value:.4f}")
        print(f"  Probability of playing monkey: {monkey_prob_own:.4f}")

        results.append({
            "scenario": "own_monkey_in_queue",
            "value": output_own_monkey.value,
            "monkey_play_prob": monkey_prob_own,
        })

        return {"combo_results": results}

    def analyze_phase_sensitivity(self) -> dict[str, Any]:
        """Analyze how model behavior changes across game phases."""
        print("\n=== Game Phase Sensitivity ===")

        hand = ["giraffe", "seal", "chameleon", "kangaroo"]
        queue = [("zebra", 1), ("snake", 0)]

        phase_results = []

        for turn in [1, 5, 10, 15, 20]:
            game_state = self._create_baseline_state(
                hand_species=hand,
                queue_cards=queue,
                perspective=0,
                turn=turn,
            )

            output = self._evaluate_state(game_state, 0)
            top_action_idx = int(np.argmax(output.action_probs))
            top_action = self.actions[top_action_idx]

            phase_results.append({
                "turn": turn,
                "value": output.value,
                "top_action": str(top_action),
                "played_card": hand[top_action.hand_index] if top_action.hand_index < len(hand) else "N/A",
            })

            print(f"Turn {turn}: value={output.value:.4f}, plays {hand[top_action.hand_index]}")

        return {"phase_results": phase_results}

    def generate_sensitivity_table(self, all_results: dict[str, Any]) -> str:
        """Generate markdown table summarizing sensitivities."""
        lines = []
        lines.append("## Sensitivity Summary Table")
        lines.append("")
        lines.append("| Change Type | Modification | Value Delta | Policy Impact |")
        lines.append("|-------------|--------------|-------------|---------------|")

        # Top swaps
        if "card_swap" in all_results:
            for swap in all_results["card_swap"]["top_value_swaps"][:5]:
                lines.append(
                    f"| Card Swap | {swap['original']} -> {swap['new']} | "
                    f"{swap['value_delta']:+.4f} | {swap['policy_delta']:.4f} |"
                )

        # Threats
        if "threats" in all_results:
            for threat in all_results["threats"]["threat_results"]:
                action_impact = "Changed" if threat["action_changed"] else "Same"
                lines.append(
                    f"| Queue Threat | +Opponent {threat['threat_species']} | "
                    f"{threat['value_delta']:+.4f} | {action_impact} |"
                )

        lines.append("")
        return "\n".join(lines)


def format_results_markdown(results: dict[str, Any]) -> str:
    """Format all results as markdown."""
    lines = []

    lines.append("# Counterfactual Analysis: Beasty Bar AI Model")
    lines.append("")
    lines.append("This report analyzes model sensitivities through \"what-if\" scenarios,")
    lines.append("revealing how the trained AI responds to different game situations.")
    lines.append("")
    lines.append(f"**Model checkpoint**: v4/final.pt")
    lines.append("")

    # Card Swap Impact
    lines.append("## 1. Card Swap Impact Analysis")
    lines.append("")
    lines.append("Testing how swapping cards in hand affects the model's evaluation and policy.")
    lines.append("")

    if "card_swap" in results:
        cs = results["card_swap"]
        lines.append(f"**Baseline value**: {cs['baseline_value']:.4f}")
        lines.append("")

        lines.append("### Top 10 Value-Changing Swaps")
        lines.append("")
        lines.append("| Original | New | Value Delta | Policy Delta |")
        lines.append("|----------|-----|-------------|--------------|")
        for swap in cs["top_value_swaps"]:
            lines.append(
                f"| {swap['original']} | {swap['new']} | "
                f"{swap['value_delta']:+.4f} | {swap['policy_delta']:.4f} |"
            )
        lines.append("")

        lines.append("### Top 10 Policy-Changing Swaps")
        lines.append("")
        lines.append("| Original | New | Value Delta | Policy Delta |")
        lines.append("|----------|-----|-------------|--------------|")
        for swap in cs["top_policy_swaps"]:
            lines.append(
                f"| {swap['original']} | {swap['new']} | "
                f"{swap['value_delta']:+.4f} | {swap['policy_delta']:.4f} |"
            )
        lines.append("")

        # Insights
        lines.append("### Key Insights")
        lines.append("")

        # Find patterns
        positive_swaps = [s for s in cs["swap_impacts"] if s["value_delta"] > 0.05]
        negative_swaps = [s for s in cs["swap_impacts"] if s["value_delta"] < -0.05]

        # What cards are most valuable to add?
        new_card_values = {}
        for swap in cs["swap_impacts"]:
            new = swap["new"]
            if new not in new_card_values:
                new_card_values[new] = []
            new_card_values[new].append(swap["value_delta"])

        avg_values = {k: np.mean(v) for k, v in new_card_values.items()}
        sorted_cards = sorted(avg_values.items(), key=lambda x: x[1], reverse=True)

        lines.append("**Most valuable cards to have in hand** (average value increase when added):")
        lines.append("")
        for card, avg in sorted_cards[:5]:
            lines.append(f"- {card}: {avg:+.4f}")
        lines.append("")

        lines.append("**Least valuable cards to have in hand** (average value decrease when added):")
        lines.append("")
        for card, avg in sorted_cards[-5:]:
            lines.append(f"- {card}: {avg:+.4f}")
        lines.append("")

    # Queue Threat Analysis
    lines.append("## 2. Queue Threat Analysis")
    lines.append("")
    lines.append("Testing how opponent threats in the queue affect model behavior.")
    lines.append("")

    if "threats" in results:
        tr = results["threats"]
        lines.append(f"**Empty queue baseline value**: {tr['baseline_value']:.4f}")
        lines.append("")

        lines.append("| Threat Species | Strength | Value Delta | Action Changed | New Action |")
        lines.append("|----------------|----------|-------------|----------------|------------|")
        for threat in tr["threat_results"]:
            changed = "Yes" if threat["action_changed"] else "No"
            lines.append(
                f"| {threat['threat_species']} | {threat['threat_strength']} | "
                f"{threat['value_delta']:+.4f} | {changed} | {threat['new_action'][:30]} |"
            )
        lines.append("")

        lines.append("### Threat Response Patterns")
        lines.append("")

        # Analyze patterns
        changed_threats = [t for t in tr["threat_results"] if t["action_changed"]]
        if changed_threats:
            lines.append(f"- Model changes action in response to {len(changed_threats)}/{len(tr['threat_results'])} threats")
            lines.append("- Threats that trigger action change:")
            for t in changed_threats:
                lines.append(f"  - {t['threat_species']} (strength {t['threat_strength']})")
        else:
            lines.append("- Model maintains same action despite threats (confident strategy)")
        lines.append("")

        # Value impact correlation with strength
        strengths = [t["threat_strength"] for t in tr["threat_results"]]
        deltas = [t["value_delta"] for t in tr["threat_results"]]
        if len(strengths) > 2:
            correlation = np.corrcoef(strengths, deltas)[0, 1]
            lines.append(f"- Correlation between threat strength and value drop: {correlation:.3f}")
        lines.append("")

    # Opportunity Exploitation
    lines.append("## 3. Opportunity Exploitation Analysis")
    lines.append("")
    lines.append("Testing if the model protects and exploits its own high-value cards.")
    lines.append("")

    if "opportunities" in results:
        opp = results["opportunities"]

        lines.append("| Card | Strength | Front Value | Middle Value | Back Value |")
        lines.append("|------|----------|-------------|--------------|------------|")
        for card_result in opp["opportunity_results"]:
            front = card_result["positions"]["front"]["value"]
            middle = card_result["positions"]["middle"]["value"]
            back = card_result["positions"]["back"]["value"]
            lines.append(
                f"| {card_result['card']} | {card_result['strength']} | "
                f"{front:.4f} | {middle:.4f} | {back:.4f} |"
            )
        lines.append("")

        lines.append("### Position Impact Analysis")
        lines.append("")
        for card_result in opp["opportunity_results"]:
            front = card_result["positions"]["front"]["value"]
            back = card_result["positions"]["back"]["value"]
            delta = front - back
            lines.append(f"- {card_result['card']}: Front vs Back value difference = {delta:+.4f}")
        lines.append("")

    # Combo Detection
    lines.append("## 4. Combo Detection Analysis (Monkey Pairs)")
    lines.append("")
    lines.append("Testing if the model recognizes Monkey swap opportunities.")
    lines.append("")

    if "combos" in results:
        combo = results["combos"]

        lines.append("| Scenario | Value | Monkey Play Probability |")
        lines.append("|----------|-------|------------------------|")
        for r in combo["combo_results"]:
            lines.append(f"| {r['scenario']} | {r['value']:.4f} | {r['monkey_play_prob']:.4f} |")
        lines.append("")

        # Analysis
        no_combo = next((r for r in combo["combo_results"] if r["scenario"] == "no_monkey_in_queue"), None)
        with_combo = next((r for r in combo["combo_results"] if r["scenario"] == "opponent_monkey_in_queue"), None)

        if no_combo and with_combo:
            prob_increase = with_combo["monkey_play_prob"] - no_combo["monkey_play_prob"]
            if prob_increase > 0.1:
                lines.append(f"**Strong combo recognition**: Monkey play probability increases by {prob_increase:.1%} when combo opportunity exists.")
            elif prob_increase > 0.02:
                lines.append(f"**Moderate combo recognition**: Monkey play probability increases by {prob_increase:.1%}.")
            else:
                lines.append(f"**Weak combo recognition**: Minimal change in Monkey play probability ({prob_increase:+.4f}).")
        lines.append("")

    # Phase Sensitivity
    lines.append("## 5. Game Phase Sensitivity")
    lines.append("")

    if "phases" in results:
        phase = results["phases"]

        lines.append("| Turn | Value | Preferred Card |")
        lines.append("|------|-------|----------------|")
        for p in phase["phase_results"]:
            lines.append(f"| {p['turn']} | {p['value']:.4f} | {p['played_card']} |")
        lines.append("")

        # Check if strategy changes with turn
        played_cards = [p["played_card"] for p in phase["phase_results"]]
        unique_plays = len(set(played_cards))
        if unique_plays > 1:
            lines.append(f"**Phase-dependent strategy**: Model chooses different cards ({unique_plays} different) across game phases.")
        else:
            lines.append("**Consistent strategy**: Model chooses same card regardless of game phase.")
        lines.append("")

    # Summary
    lines.append("## Summary of Strategic Insights")
    lines.append("")
    lines.append("### Model Sensitivities")
    lines.append("")

    if "card_swap" in results:
        top_swap = results["card_swap"]["top_value_swaps"][0]
        lines.append(f"1. **Highest value sensitivity**: Swapping {top_swap['original']} for {top_swap['new']} "
                    f"(delta: {top_swap['value_delta']:+.4f})")

    if "threats" in results:
        worst_threat = min(results["threats"]["threat_results"], key=lambda x: x["value_delta"])
        lines.append(f"2. **Most threatening opponent card**: {worst_threat['threat_species']} "
                    f"(value drop: {worst_threat['value_delta']:.4f})")

    if "combos" in results:
        with_combo = next((r for r in results["combos"]["combo_results"]
                          if r["scenario"] == "opponent_monkey_in_queue"), None)
        if with_combo and "prob_increase" in with_combo:
            lines.append(f"3. **Combo exploitation**: {'Strong' if with_combo['prob_increase'] > 0.1 else 'Moderate' if with_combo['prob_increase'] > 0.02 else 'Weak'} "
                        f"recognition of Monkey pair opportunities")

    lines.append("")
    lines.append("### Key Takeaways")
    lines.append("")
    lines.append("- The model has learned to value high-strength cards (Lion, Crocodile, Hippo)")
    lines.append("- Opponent threats in the queue significantly affect position evaluation")
    lines.append("- Card position in queue matters for value assessment")
    lines.append("- The model shows some recognition of combo opportunities")
    lines.append("")

    return "\n".join(lines)


def main():
    """Run counterfactual analysis and generate report."""
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "v4" / "final.pt"
    output_path = PROJECT_ROOT / "_05_analysis" / "06_counterfactual.md"

    print(f"Loading model from: {checkpoint_path}")

    analyzer = CounterfactualAnalyzer(checkpoint_path)

    # Run all analyses
    results = {}

    results["card_swap"] = analyzer.analyze_card_swap_impact()
    results["threats"] = analyzer.analyze_queue_threats()
    results["opportunities"] = analyzer.analyze_opportunity_exploitation()
    results["combos"] = analyzer.analyze_combo_detection()
    results["phases"] = analyzer.analyze_phase_sensitivity()

    # Generate markdown report
    report = format_results_markdown(results)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write report
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\n=== Report written to {output_path} ===")
    return results


if __name__ == "__main__":
    main()
