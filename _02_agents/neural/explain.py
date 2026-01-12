"""Move explanation utilities for AI transparency.

This module provides utilities for explaining why the neural network
chose a particular move, including:
- Feature importance rankings
- Alternative move analysis
- Value prediction breakdown
- Confidence scoring

Example:
    from _02_agents.neural.explain import explain_move, MoveExplainer

    explainer = MoveExplainer(network)
    explanation = explainer.explain(state, action, legal_actions)
    print(explanation.summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ActionExplanation:
    """Explanation for a single action.

    Attributes:
        action_index: Index of the action in legal actions list.
        hand_index: Card position in hand.
        card_species: Species of the card played.
        probability: Probability assigned by policy.
        log_probability: Log probability.
        value_estimate: Value estimate if this action were taken.
        rank: Rank among legal actions (1 = best).
    """

    action_index: int
    hand_index: int
    card_species: str
    probability: float
    log_probability: float
    value_estimate: float | None
    rank: int


@dataclass
class FeatureImportance:
    """Feature importance for a decision.

    Attributes:
        feature_name: Human-readable feature name.
        importance_score: Importance score (higher = more important).
        feature_value: Current value of the feature.
        direction: Whether feature pushed toward (+) or away from (-) action.
    """

    feature_name: str
    importance_score: float
    feature_value: float
    direction: str  # "positive", "negative", "neutral"


@dataclass
class MoveExplanation:
    """Complete explanation for a move decision.

    Attributes:
        chosen_action: The action that was chosen.
        alternatives: Top alternative actions considered.
        confidence: Confidence in the chosen action (0-1).
        value_prediction: Predicted game outcome value.
        top_features: Most important features for this decision.
        reasoning: Human-readable reasoning summary.
        raw_policy: Full policy distribution.
        raw_value: Raw value output.
    """

    chosen_action: ActionExplanation
    alternatives: list[ActionExplanation]
    confidence: float
    value_prediction: float
    top_features: list[FeatureImportance]
    reasoning: str
    raw_policy: list[float]
    raw_value: float


# Feature names for interpretation
ZONE_FEATURES = {
    "queue": "Queue position analysis",
    "bar": "Bar (Heaven's Gate) analysis",
    "thats_it": "That's It zone analysis",
    "hand": "Hand cards analysis",
    "opponent_hand": "Opponent hand estimate",
}

SCALAR_FEATURES = {
    0: "Own score",
    1: "Opponent score",
    2: "Score difference",
    3: "Cards in deck",
    4: "Turn number",
    5: "Queue fullness",
    6: "Game phase",
}

CARD_IMPORTANCE = {
    "kangaroo": "High mobility, can hop over cards",
    "hippo": "Pushes back other cards",
    "crocodile": "Eats cards at front of queue",
    "snake": "Reorders queue cards",
    "seal": "Duplicates abilities",
    "chameleon": "Copies other species",
    "zebra": "Cannot be pushed",
    "parrot": "Repeats last action",
    "monkey": "Works well in pairs",
    "giraffe": "Advances to front",
    "lion": "High strength, king of queue",
    "skunk": "Clears the queue",
}


class MoveExplainer:
    """Explains neural network move decisions.

    Provides interpretable explanations for why the AI chose
    a particular move over alternatives.

    Attributes:
        network: The neural network to explain.
        device: Device for inference.

    Example:
        >>> explainer = MoveExplainer(network)
        >>> explanation = explainer.explain(state, chosen_action, legal_actions)
        >>> print(explanation.reasoning)
    """

    def __init__(
        self,
        network: nn.Module,
        device: str | None = None,
    ) -> None:
        """Initialize the explainer.

        Args:
            network: Neural network model.
            device: Device for inference (auto-detected if None).
        """
        import torch

        self.network = network
        self.network.eval()

        if device is None:
            try:
                self.device = next(network.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def explain(
        self,
        observation: np.ndarray | torch.Tensor,
        chosen_action_idx: int,
        legal_action_indices: list[int],
        action_labels: list[str] | None = None,
        card_species: list[str] | None = None,
        num_alternatives: int = 3,
    ) -> MoveExplanation:
        """Generate explanation for a move.

        Args:
            observation: Game observation tensor (988-dim).
            chosen_action_idx: Index of the chosen action.
            legal_action_indices: List of legal action indices.
            action_labels: Human-readable labels for actions.
            card_species: Species of cards for each action.
            num_alternatives: Number of alternative actions to analyze.

        Returns:
            MoveExplanation with full analysis.
        """
        import torch
        import torch.nn.functional as f

        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.from_numpy(observation).float()
        else:
            obs_tensor = observation.float()

        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        obs_tensor = obs_tensor.to(self.device)

        # Get network output
        with torch.no_grad():
            policy_logits, value = self.network(obs_tensor)

        policy_logits = policy_logits.squeeze(0)
        value = value.squeeze().item()

        # Create action mask
        mask = torch.full((policy_logits.size(0),), float("-inf"), device=self.device)
        for idx in legal_action_indices:
            mask[idx] = 0

        # Apply mask and softmax
        masked_logits = policy_logits + mask
        probabilities = f.softmax(masked_logits, dim=0)

        # Get probabilities for legal actions
        legal_probs = [(idx, probabilities[idx].item()) for idx in legal_action_indices]
        legal_probs.sort(key=lambda x: x[1], reverse=True)

        # Rank actions
        rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(legal_probs)}

        # Build action explanations
        def make_action_explanation(action_idx: int, rank: int) -> ActionExplanation:
            prob = probabilities[action_idx].item()
            log_prob = torch.log(probabilities[action_idx] + 1e-10).item()

            # Determine hand index and species from action index
            hand_idx = action_idx // 31  # Simplified mapping
            species = "unknown"
            if card_species and hand_idx < len(card_species):
                species = card_species[hand_idx]

            return ActionExplanation(
                action_index=action_idx,
                hand_index=hand_idx,
                card_species=species,
                probability=prob,
                log_probability=log_prob,
                value_estimate=None,  # Would require forward pass with action
                rank=rank,
            )

        chosen_explanation = make_action_explanation(chosen_action_idx, rank_map[chosen_action_idx])

        # Get top alternatives
        alternatives = []
        for idx, _prob in legal_probs[:num_alternatives + 1]:
            if idx != chosen_action_idx:
                alternatives.append(make_action_explanation(idx, rank_map[idx]))
            if len(alternatives) >= num_alternatives:
                break

        # Calculate confidence
        chosen_prob = probabilities[chosen_action_idx].item()
        if len(legal_probs) > 1:
            second_best_prob = legal_probs[1][1] if legal_probs[0][0] == chosen_action_idx else legal_probs[0][1]
            confidence = chosen_prob / (chosen_prob + second_best_prob)
        else:
            confidence = 1.0

        # Extract feature importance (simplified gradient-based)
        top_features = self._analyze_features(observation, chosen_action_idx)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            chosen_explanation,
            alternatives,
            confidence,
            value,
            top_features,
        )

        return MoveExplanation(
            chosen_action=chosen_explanation,
            alternatives=alternatives,
            confidence=confidence,
            value_prediction=value,
            top_features=top_features,
            reasoning=reasoning,
            raw_policy=probabilities.cpu().tolist(),
            raw_value=value,
        )

    def _analyze_features(
        self,
        observation: np.ndarray | torch.Tensor,
        action_idx: int,
    ) -> list[FeatureImportance]:
        """Analyze feature importance for an action.

        Uses a simplified perturbation-based approach.
        """
        import torch

        if isinstance(observation, np.ndarray):
            obs = torch.from_numpy(observation).float()
        else:
            obs = observation.float().clone()

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        obs = obs.to(self.device)

        # Get baseline policy
        with torch.no_grad():
            baseline_logits, _ = self.network(obs)
            baseline_prob = torch.softmax(baseline_logits, dim=-1)[0, action_idx].item()

        # Feature groups to analyze
        feature_groups = [
            ("Queue cards (positions 0-84)", 0, 85),
            ("Bar cards (positions 85-492)", 85, 493),
            ("That's It (positions 493-900)", 493, 901),
            ("Hand cards (positions 901-968)", 901, 969),
            ("Opponent hand (positions 969-980)", 969, 981),
            ("Game state (positions 981-987)", 981, 988),
        ]

        importances = []

        for name, start, end in feature_groups:
            # Perturb this feature group
            perturbed = obs.clone()
            perturbed[0, start:end] = 0  # Zero out feature group

            with torch.no_grad():
                perturbed_logits, _ = self.network(perturbed)
                perturbed_prob = torch.softmax(perturbed_logits, dim=-1)[0, action_idx].item()

            # Importance = change in probability
            importance = abs(baseline_prob - perturbed_prob)
            direction = "positive" if perturbed_prob < baseline_prob else "negative"

            importances.append(
                FeatureImportance(
                    feature_name=name,
                    importance_score=importance,
                    feature_value=obs[0, start:end].mean().item(),
                    direction=direction,
                )
            )

        # Sort by importance
        importances.sort(key=lambda x: x.importance_score, reverse=True)

        return importances[:5]  # Return top 5

    def _generate_reasoning(
        self,
        chosen: ActionExplanation,
        alternatives: list[ActionExplanation],
        confidence: float,
        value: float,
        features: list[FeatureImportance],
    ) -> str:
        """Generate human-readable reasoning."""
        lines = []

        # Confidence assessment
        if confidence > 0.8:
            lines.append(f"High confidence ({confidence:.0%}) in playing {chosen.card_species}.")
        elif confidence > 0.6:
            lines.append(f"Moderate confidence ({confidence:.0%}) in playing {chosen.card_species}.")
        else:
            lines.append(f"Low confidence ({confidence:.0%}) - close decision between options.")

        # Value assessment
        if value > 0.3:
            lines.append(f"Position looks favorable (value: {value:+.2f}).")
        elif value < -0.3:
            lines.append(f"Position is challenging (value: {value:+.2f}).")
        else:
            lines.append(f"Position is balanced (value: {value:+.2f}).")

        # Card-specific reasoning
        species = chosen.card_species.lower()
        if species in CARD_IMPORTANCE:
            lines.append(f"{chosen.card_species}: {CARD_IMPORTANCE[species]}")

        # Feature importance
        if features:
            top_feature = features[0]
            lines.append(f"Key factor: {top_feature.feature_name} (importance: {top_feature.importance_score:.2f})")

        # Alternatives
        if alternatives and alternatives[0].probability > 0.2:
            alt = alternatives[0]
            lines.append(
                f"Considered alternative: {alt.card_species} ({alt.probability:.0%} probability)"
            )

        return " ".join(lines)


def explain_move(
    network: nn.Module,
    observation: np.ndarray | torch.Tensor,
    chosen_action_idx: int,
    legal_action_indices: list[int],
    **kwargs,
) -> MoveExplanation:
    """Convenience function to explain a single move.

    Args:
        network: Neural network model.
        observation: Game observation.
        chosen_action_idx: Index of chosen action.
        legal_action_indices: Legal action indices.
        **kwargs: Additional arguments passed to MoveExplainer.explain().

    Returns:
        MoveExplanation for the move.

    Example:
        >>> explanation = explain_move(network, obs, action_idx, legal_actions)
        >>> print(explanation.reasoning)
    """
    explainer = MoveExplainer(network)
    return explainer.explain(observation, chosen_action_idx, legal_action_indices, **kwargs)


def format_explanation_for_api(explanation: MoveExplanation) -> dict:
    """Format explanation for API response.

    Args:
        explanation: MoveExplanation to format.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    return {
        "chosen_action": {
            "action_index": explanation.chosen_action.action_index,
            "hand_index": explanation.chosen_action.hand_index,
            "card_species": explanation.chosen_action.card_species,
            "probability": round(explanation.chosen_action.probability, 4),
            "rank": explanation.chosen_action.rank,
        },
        "alternatives": [
            {
                "action_index": alt.action_index,
                "hand_index": alt.hand_index,
                "card_species": alt.card_species,
                "probability": round(alt.probability, 4),
                "rank": alt.rank,
            }
            for alt in explanation.alternatives
        ],
        "confidence": round(explanation.confidence, 4),
        "value_prediction": round(explanation.value_prediction, 4),
        "top_factors": [
            {
                "feature": f.feature_name,
                "importance": round(f.importance_score, 4),
                "direction": f.direction,
            }
            for f in explanation.top_features
        ],
        "reasoning": explanation.reasoning,
    }


__all__ = [
    "ActionExplanation",
    "FeatureImportance",
    "MoveExplainer",
    "MoveExplanation",
    "explain_move",
    "format_explanation_for_api",
]
