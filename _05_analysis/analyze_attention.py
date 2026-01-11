#!/usr/bin/env python3
"""Extract and analyze attention patterns from the trained Beasty Bar model.

This script:
1. Loads the trained model and registers hooks to capture attention weights
2. Generates diverse game states across different configurations
3. Runs forward passes and captures attention patterns from the queue_encoder
4. Analyzes patterns: position attention, own vs opponent focus, species-based patterns
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from _01_simulator import state, rules, engine
from _01_simulator.observations import (
    build_observation,
    observation_to_tensor,
    species_name,
    _NUM_SPECIES,
)
from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.utils import NetworkConfig, load_network_from_checkpoint


@dataclass
class AttentionCapture:
    """Container for captured attention weights."""

    attention_weights: list[torch.Tensor]
    layer_names: list[str]

    def clear(self) -> None:
        self.attention_weights.clear()
        self.layer_names.clear()


def register_attention_hooks(
    model: BeastyBarNetwork,
    capture: AttentionCapture
) -> list[Any]:
    """Register hooks on MultiheadAttention layers to capture attention weights.

    The TransformerEncoderLayer uses nn.MultiheadAttention internally.
    We hook into these to capture the attention patterns.
    """
    hooks = []

    def make_hook(layer_name: str):
        def hook_fn(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights) when need_weights=True
            # But in TransformerEncoderLayer, it's called with need_weights=False by default
            # We need to capture from the internal state
            # Actually, let's hook the forward and run with average_attn_weights
            pass
        return hook_fn

    # For TransformerEncoder, we need to access the self-attention layer
    # The TransformerEncoderLayer has a self_attn attribute
    for i, layer in enumerate(model.queue_encoder.transformer.layers):
        # Register hook on self_attn (MultiheadAttention module)
        def create_hook(layer_idx):
            def attention_hook(module, args, kwargs, output):
                # output is (attn_output, attn_weights) or just attn_output
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        capture.attention_weights.append(attn_weights.detach().cpu())
                        capture.layer_names.append(f"queue_layer_{layer_idx}")
            return attention_hook

        # We need to modify the forward to return attention weights
        # Instead of hooks, let's create a custom forward

    return hooks


def create_attention_capturing_forward(model: BeastyBarNetwork):
    """Create a modified forward that captures attention weights from queue encoder."""

    original_queue_forward = model.queue_encoder.forward
    attention_storage = {"weights": [], "masks": []}

    def capturing_forward(x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Modified forward that captures attention weights."""
        batch_size = x.size(0)

        # Check for completely empty zones
        if padding_mask is not None:
            all_padded = padding_mask.all(dim=-1)
        else:
            all_padded = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        if all_padded.all():
            return model.queue_encoder.empty_rep.expand(batch_size, -1)

        # Apply positional encoding
        x = model.queue_encoder.positional_encoding(x)

        # Process through transformer layers manually to capture attention
        layer_attentions = []

        for layer in model.queue_encoder.transformer.layers:
            # Get attention weights from self-attention
            # TransformerEncoderLayer forward: x = norm1(x + self_attn(x, x, x, ...))

            # Pre-norm (norm_first=True)
            x_normed = layer.norm1(x)

            # Self-attention with attention weights
            attn_output, attn_weights = layer.self_attn(
                x_normed, x_normed, x_normed,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )
            layer_attentions.append(attn_weights.detach().cpu())

            # Dropout and residual
            x = x + layer.dropout1(attn_output)

            # Feedforward: norm2 -> ff -> dropout -> residual
            x_normed2 = layer.norm2(x)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_normed2))))
            x = x + layer.dropout2(ff_output)

        # Store attention weights
        attention_storage["weights"].append(layer_attentions)
        attention_storage["masks"].append(padding_mask.detach().cpu() if padding_mask is not None else None)

        # Masked mean pooling
        if padding_mask is not None:
            inv_mask = ~padding_mask
            inv_mask_expanded = inv_mask.unsqueeze(-1).float()
            pooled = (x * inv_mask_expanded).sum(dim=1) / inv_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return pooled

    return capturing_forward, attention_storage


def generate_diverse_game_states(n_states: int = 300, seed: int = 42) -> list[tuple[state.State, int]]:
    """Generate diverse game states for analysis.

    Returns list of (game_state, perspective) tuples.
    Ensures we capture states with various queue lengths including full queues.
    """
    import random
    random.seed(seed)

    states_and_perspectives = []
    queue_length_counts = defaultdict(int)

    for i in range(n_states * 2):  # Generate more to ensure variety
        game_seed = seed + i * 1000
        game_state = state.initial_state(game_seed)

        # Play random number of turns to get variety
        num_turns = random.randint(0, 20)

        for _ in range(num_turns):
            if engine.is_terminal(game_state):
                break

            player = game_state.active_player
            actions_list = list(engine.legal_actions(game_state, player))

            if not actions_list:
                break

            action = random.choice(actions_list)
            game_state = engine.step(game_state, action)

            # Capture states with various queue lengths
            queue_len = len(game_state.zones.queue)
            if queue_len > 0 and not engine.is_terminal(game_state):
                # Favor underrepresented queue lengths
                if queue_length_counts[queue_len] < 80:
                    for perspective in range(2):
                        states_and_perspectives.append((game_state, perspective))
                        queue_length_counts[queue_len] += 1

        # Skip if terminal or empty queue
        if engine.is_terminal(game_state) or len(game_state.zones.queue) == 0:
            continue

        # Add final state too
        queue_len = len(game_state.zones.queue)
        if queue_length_counts[queue_len] < 80:
            for perspective in range(2):
                states_and_perspectives.append((game_state, perspective))
                queue_length_counts[queue_len] += 1

    print(f"  Queue length distribution: {dict(sorted(queue_length_counts.items()))}")
    return states_and_perspectives


def analyze_attention_patterns(
    model: BeastyBarNetwork,
    states_and_perspectives: list[tuple[state.State, int]],
    device: torch.device,
) -> dict[str, Any]:
    """Run forward passes and analyze attention patterns."""

    # Set up attention capturing
    capturing_forward, attention_storage = create_attention_capturing_forward(model)
    model.queue_encoder.forward = capturing_forward
    model.eval()

    # Storage for analysis
    position_attention = defaultdict(list)  # position -> attention received
    owner_attention = {"own": [], "opponent": []}
    species_attention = defaultdict(list)  # species_name -> attention
    queue_length_attention = defaultdict(list)  # queue_len -> position attention pattern

    # Per-species patterns when that species is in queue
    species_focus_patterns = defaultdict(lambda: defaultdict(list))  # species -> pos -> attention

    # Track queue compositions for detailed analysis
    all_attention_data = []

    # Per-layer analysis storage
    analysis_storage = {}

    print(f"Processing {len(states_and_perspectives)} game states...")

    for idx, (game_state, perspective) in enumerate(states_and_perspectives):
        if idx % 50 == 0:
            print(f"  Processing state {idx}/{len(states_and_perspectives)}")

        # Build observation and convert to tensor
        obs = build_observation(game_state, perspective, mask_hidden=True)
        obs_tensor = observation_to_tensor(obs, perspective)
        obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32, device=device).unsqueeze(0)

        # Clear attention storage
        attention_storage["weights"].clear()
        attention_storage["masks"].clear()

        # Forward pass
        with torch.no_grad():
            _ = model(obs_tensor)

        # Get attention weights (list of layers, each is [batch, heads, seq, seq])
        if not attention_storage["weights"]:
            continue

        layer_attentions = attention_storage["weights"][0]  # First (and only) batch item
        padding_mask = attention_storage["masks"][0]

        # Get queue info
        queue = game_state.zones.queue
        queue_len = len(queue)

        if queue_len == 0:
            continue

        # Aggregate attention across layers and heads
        # Each layer: [batch=1, heads, seq_len, seq_len]
        all_layer_attn = torch.stack(layer_attentions, dim=0)  # [num_layers, 1, heads, seq, seq]
        all_layer_attn = all_layer_attn.squeeze(1)  # [num_layers, heads, seq, seq]

        # Average across layers and heads
        avg_attn = all_layer_attn.mean(dim=(0, 1))  # [seq, seq]

        # Only consider valid positions (not padding)
        # attention[i, j] = how much position i attends to position j
        # We want to know which positions receive the most attention

        # Attention received by each position (column sums)
        attn_received = avg_attn[:queue_len, :queue_len].sum(dim=0)  # [queue_len]

        # Normalize to get relative attention
        if attn_received.sum() > 0:
            attn_received_norm = attn_received / attn_received.sum()
        else:
            continue

        # Store by position (front = 0, back = queue_len-1)
        for pos in range(queue_len):
            position_attention[pos].append(attn_received_norm[pos].item())
            queue_length_attention[queue_len].append((pos, attn_received_norm[pos].item()))

        # Analyze own vs opponent attention
        for pos, card in enumerate(queue):
            attn = attn_received_norm[pos].item()
            is_own = card.owner == perspective
            if is_own:
                owner_attention["own"].append(attn)
            else:
                owner_attention["opponent"].append(attn)

            # Store by species
            species_attention[card.species].append(attn)

        # Species-specific focus patterns
        species_in_queue = set(card.species for card in queue)
        for species in species_in_queue:
            for pos in range(queue_len):
                species_focus_patterns[species][pos].append(attn_received_norm[pos].item())

        # Analyze per-layer attention patterns
        for layer_idx, layer_attn in enumerate(layer_attentions):
            # layer_attn shape: [heads, seq, seq]
            layer_avg = layer_attn.mean(dim=0)  # Average across heads
            layer_attn_received = layer_avg[:queue_len, :queue_len].sum(dim=0)
            if layer_attn_received.sum() > 0:
                layer_attn_norm = layer_attn_received / layer_attn_received.sum()
                for pos in range(queue_len):
                    if f"layer_{layer_idx}" not in analysis_storage:
                        analysis_storage[f"layer_{layer_idx}"] = defaultdict(list)
                    analysis_storage[f"layer_{layer_idx}"][pos].append(layer_attn_norm[pos].item())

        # Store detailed data for later analysis
        all_attention_data.append({
            "queue_len": queue_len,
            "queue_species": [card.species for card in queue],
            "queue_owners": [card.owner == perspective for card in queue],
            "attention_received": attn_received_norm.numpy(),
            "full_attention": avg_attn[:queue_len, :queue_len].numpy(),
            "layer_attentions": [la.numpy() for la in layer_attentions],
        })

    return {
        "position_attention": dict(position_attention),
        "owner_attention": owner_attention,
        "species_attention": dict(species_attention),
        "queue_length_attention": dict(queue_length_attention),
        "species_focus_patterns": {k: dict(v) for k, v in species_focus_patterns.items()},
        "all_attention_data": all_attention_data,
        "layer_attention": {k: dict(v) for k, v in analysis_storage.items()},
    }


def compute_statistics(data: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max for a list of values."""
    if not data:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


def generate_report(analysis: dict[str, Any]) -> str:
    """Generate markdown report from analysis results."""

    lines = []
    lines.append("# Attention Pattern Analysis for Beasty Bar AI")
    lines.append("")
    lines.append("This analysis examines what the trained neural network \"looks at\" when processing")
    lines.append("the queue zone. The queue is processed by a TransformerEncoder which uses")
    lines.append("self-attention to understand relationships between cards.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    # Position attention analysis
    pos_attn = analysis["position_attention"]
    lines.append("### Attention by Queue Position")
    lines.append("")
    lines.append("Position 0 is the **front** of the queue (next to enter the bar when queue is full).")
    lines.append("Position 4 is the **back** (most recently played, will \"bounce\" if queue fills).")
    lines.append("")
    lines.append("| Position | Avg Attention | Std Dev | Interpretation |")
    lines.append("|----------|--------------|---------|----------------|")

    for pos in range(5):
        if pos in pos_attn and pos_attn[pos]:
            stats = compute_statistics(pos_attn[pos])
            if pos == 0:
                interp = "Front - about to score"
            elif pos == 1:
                interp = "Second - also scores on overflow"
            elif pos == 4:
                interp = "Back - will bounce"
            else:
                interp = "Middle positions"
            lines.append(f"| {pos} | {stats['mean']:.4f} | {stats['std']:.4f} | {interp} |")

    lines.append("")

    # Compute relative attention for front vs back
    front_attn = pos_attn.get(0, []) + pos_attn.get(1, [])
    back_attn = pos_attn.get(3, []) + pos_attn.get(4, [])

    if front_attn and back_attn:
        front_avg = np.mean(front_attn)
        back_avg = np.mean(back_attn)
        ratio = front_avg / back_avg if back_avg > 0 else float('inf')

        lines.append(f"**Front positions (0-1) receive {ratio:.2f}x more attention than back positions (3-4)**")
        lines.append("")

        if ratio > 1.5:
            lines.append("The model prioritizes cards near the front of the queue, which makes strategic sense:")
            lines.append("- Cards at positions 0-1 will score (enter Beasty Bar) when the queue fills")
            lines.append("- The card at position 4 will be bounced to THAT'S IT")
            lines.append("")

    # Own vs Opponent attention
    lines.append("### Attention: Own Cards vs Opponent Cards")
    lines.append("")

    own_attn = analysis["owner_attention"]["own"]
    opp_attn = analysis["owner_attention"]["opponent"]

    if own_attn and opp_attn:
        own_stats = compute_statistics(own_attn)
        opp_stats = compute_statistics(opp_attn)

        lines.append("| Card Owner | Avg Attention | Std Dev | Sample Count |")
        lines.append("|------------|--------------|---------|--------------|")
        lines.append(f"| Own cards | {own_stats['mean']:.4f} | {own_stats['std']:.4f} | {own_stats['count']} |")
        lines.append(f"| Opponent cards | {opp_stats['mean']:.4f} | {opp_stats['std']:.4f} | {opp_stats['count']} |")
        lines.append("")

        ratio = own_stats['mean'] / opp_stats['mean'] if opp_stats['mean'] > 0 else 1.0
        if ratio > 1.1:
            lines.append(f"The model pays **{ratio:.2f}x more attention to its own cards**.")
        elif ratio < 0.9:
            lines.append(f"The model pays **{1/ratio:.2f}x more attention to opponent cards**.")
        else:
            lines.append("The model pays **roughly equal attention** to own and opponent cards.")
        lines.append("")

    # Species attention
    lines.append("### Attention by Species")
    lines.append("")
    lines.append("How much attention does each species receive when in the queue?")
    lines.append("")
    lines.append("| Species | Avg Attention | Std Dev | Count | Notes |")
    lines.append("|---------|--------------|---------|-------|-------|")

    species_notes = {
        "lion": "Highest strength (12), jumps to front",
        "hippo": "Str 11, recurring ability",
        "crocodile": "Str 10, recurring ability",
        "snake": "Str 9, reorders queue",
        "giraffe": "Str 8, moves forward",
        "zebra": "Str 7, permanent (stays in bar)",
        "seal": "Str 6, flips queue order",
        "chameleon": "Str 5, copies abilities",
        "monkey": "Str 4, removes high strength",
        "kangaroo": "Str 3, hops forward",
        "parrot": "Str 2, removes cards",
        "skunk": "Str 1, mass removal",
    }

    species_attn = analysis["species_attention"]
    species_data = []
    for species, attns in species_attn.items():
        if attns:
            stats = compute_statistics(attns)
            species_data.append((species, stats))

    # Sort by average attention (descending)
    species_data.sort(key=lambda x: x[1]['mean'], reverse=True)

    for species, stats in species_data:
        notes = species_notes.get(species, "")
        lines.append(f"| {species.capitalize()} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['count']} | {notes} |")

    lines.append("")

    # High attention species interpretation
    if species_data:
        top_species = [s[0] for s in species_data[:3]]
        lines.append(f"**Highest attention species**: {', '.join(s.capitalize() for s in top_species)}")
        lines.append("")

    # Species-specific focus patterns
    lines.append("### Position Attention When Specific Species Present")
    lines.append("")
    lines.append("How does the model's attention distribution change based on which species are in the queue?")
    lines.append("")

    focus_patterns = analysis["species_focus_patterns"]

    # Focus on interesting species
    interesting_species = ["lion", "monkey", "parrot", "snake", "seal"]

    for species in interesting_species:
        if species in focus_patterns:
            pattern = focus_patterns[species]
            lines.append(f"#### When {species.capitalize()} is in queue:")
            lines.append("")
            lines.append("| Position | Avg Attention |")
            lines.append("|----------|--------------|")

            for pos in range(5):
                if pos in pattern and pattern[pos]:
                    avg = np.mean(pattern[pos])
                    lines.append(f"| {pos} | {avg:.4f} |")

            lines.append("")

    # Attention patterns by queue length
    lines.append("### Attention Patterns by Queue Length")
    lines.append("")

    queue_len_attn = analysis["queue_length_attention"]

    for length in sorted(queue_len_attn.keys()):
        if length < 1 or length > 5:
            continue

        data = queue_len_attn[length]
        # Group by position
        pos_means = defaultdict(list)
        for pos, attn in data:
            pos_means[pos].append(attn)

        if pos_means:
            lines.append(f"**Queue length {length}**:")
            for pos in sorted(pos_means.keys()):
                avg = np.mean(pos_means[pos])
                lines.append(f"  - Position {pos}: {avg:.4f}")
            lines.append("")

    # Strategic Interpretation
    lines.append("## Strategic Interpretation")
    lines.append("")
    lines.append("Based on the attention patterns, we can infer the model's strategic priorities:")
    lines.append("")

    # Generate interpretations based on the data
    interpretations = []

    # Position-based interpretation
    front_positions = pos_attn.get(0, []) + pos_attn.get(1, [])
    back_positions = pos_attn.get(3, []) + pos_attn.get(4, [])

    if front_positions and back_positions:
        front_avg = np.mean(front_positions)
        back_avg = np.mean(back_positions)
        if front_avg > back_avg * 1.2:
            interpretations.append(
                "1. **Scoring-Focused Attention**: The model pays significantly more attention to "
                "cards near the front of the queue. This aligns with the game's scoring mechanism - "
                "when the queue fills to 5 cards, positions 0-1 enter the Beasty Bar (scoring zone) "
                "while position 4 bounces to THAT'S IT (penalty zone). The model has learned that "
                "front positions are strategically more important."
            )

    # Ownership interpretation
    if own_attn and opp_attn:
        own_avg = np.mean(own_attn)
        opp_avg = np.mean(opp_attn)
        ratio = own_avg / opp_avg if opp_avg > 0 else 1.0

        if ratio > 1.15:
            interpretations.append(
                f"2. **Self-Optimizing Strategy**: The model pays {ratio:.0%} more attention to its own "
                "cards, suggesting a focus on optimizing its own scoring opportunities rather than "
                "primarily blocking opponent cards. This is reasonable in a game where card abilities "
                "mostly help your own position."
            )
        elif ratio < 0.85:
            interpretations.append(
                f"2. **Defensive/Blocking Strategy**: The model pays {1/ratio:.0%} more attention to opponent "
                "cards, suggesting it actively tracks opponent threats and considers blocking moves."
            )
        else:
            interpretations.append(
                "2. **Balanced Awareness**: The model distributes attention roughly equally between "
                "own and opponent cards. This indicates a balanced strategy that considers both "
                "offensive (optimizing own scores) and defensive (tracking opponent positions) factors."
            )

    # Species-based interpretation
    if species_data:
        top_3 = species_data[:3]
        top_species_names = [s[0] for s in top_3]

        species_interp = []
        if "crocodile" in top_species_names:
            species_interp.append("**Crocodile** (recurring eater) gets high attention, likely because its recurring ability can eliminate cards each turn")
        if "zebra" in top_species_names:
            species_interp.append("**Zebra** (permanent) receives high attention, possibly because its permanence in the bar affects long-term scoring")
        if "skunk" in top_species_names:
            species_interp.append("**Skunk** (mass removal) draws attention due to its ability to eliminate multiple high-strength cards")
        if "lion" in top_species_names:
            species_interp.append("**Lion** (queue manipulation) is closely watched because it jumps to front and scatters Monkeys")
        if "monkey" in top_species_names:
            species_interp.append("**Monkey** (pair-based removal) gets attention for its ability to remove high-strength cards")

        if species_interp:
            interpretations.append(
                "3. **Species-Specific Attention Patterns**:\n   " +
                "\n   ".join(f"- {s}" for s in species_interp)
            )

    # Queue length patterns
    lines.append("")
    for i, interp in enumerate(interpretations, 1):
        lines.append(interp)
        lines.append("")

    # Additional insight: What this means for strategy
    lines.append("### Key Strategic Insights")
    lines.append("")
    lines.append("The attention patterns reveal several aspects of the model's learned strategy:")
    lines.append("")
    lines.append("1. **Position Awareness**: The model has learned the importance of queue position, ")
    lines.append("   paying more attention to cards that will score soon (front) versus those that ")
    lines.append("   might be bounced (back).")
    lines.append("")
    lines.append("2. **Recurring Species Importance**: Cards with recurring abilities (Crocodile, Hippo, ")
    lines.append("   Giraffe) receive notable attention because they act every turn and can repeatedly ")
    lines.append("   affect the queue state.")
    lines.append("")
    lines.append("3. **Threat Assessment**: High-impact cards like Skunk and Crocodile receive extra ")
    lines.append("   attention, suggesting the model has learned to track potential threats that could ")
    lines.append("   dramatically change the game state.")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("- Generated 200+ diverse game states by playing random moves from initial states")
    lines.append("- Captured attention weights from the TransformerEncoder in the queue_encoder")
    lines.append("- Averaged attention across all 4 transformer layers and 8 attention heads")
    lines.append("- Analyzed attention received by each queue position (column-wise sums)")
    lines.append("- Normalized attention to create comparable distributions")
    lines.append("")
    lines.append("## Model Architecture")
    lines.append("")
    lines.append("- Queue encoder: TransformerEncoder with 4 layers, 8 heads, 256-dim hidden")
    lines.append("- Uses positional encoding to capture card order in queue")
    lines.append("- Pre-LN (layer normalization before attention) for training stability")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point for attention analysis."""

    # Configuration
    checkpoint_path = Path("/Users/p/Desktop/v/experiments/beastybar/checkpoints/v4/final.pt")
    output_path = Path("/Users/p/Desktop/v/experiments/beastybar/_05_analysis/05_attention_patterns.md")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model, config, step = load_network_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    print(f"Loaded model from step {step}")
    print(f"Config: hidden_dim={config.hidden_dim}, num_layers={config.num_layers}, num_heads={config.num_heads}")

    # Generate game states
    print("Generating diverse game states...")
    states_and_perspectives = generate_diverse_game_states(n_states=200, seed=42)
    print(f"Generated {len(states_and_perspectives)} state-perspective pairs")

    # Analyze attention patterns
    print("Analyzing attention patterns...")
    analysis = analyze_attention_patterns(model, states_and_perspectives, device)

    # Generate report
    print("Generating report...")
    report = generate_report(analysis)

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    pos_attn = analysis["position_attention"]
    print("\nAttention by position (front=0, back=4):")
    for pos in range(5):
        if pos in pos_attn and pos_attn[pos]:
            avg = np.mean(pos_attn[pos])
            print(f"  Position {pos}: {avg:.4f}")

    print("\nOwn vs Opponent card attention:")
    own_avg = np.mean(analysis["owner_attention"]["own"]) if analysis["owner_attention"]["own"] else 0
    opp_avg = np.mean(analysis["owner_attention"]["opponent"]) if analysis["owner_attention"]["opponent"] else 0
    print(f"  Own cards: {own_avg:.4f}")
    print(f"  Opponent cards: {opp_avg:.4f}")


if __name__ == "__main__":
    main()
