"""Data compression utilities for activation visualization.

Compresses high-dimensional activation tensors into JSON-friendly
summary statistics for efficient WebSocket streaming.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from _04_ui.visualization.activation_capture import ActivationSnapshot


def compress_activation(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    """Compress activation tensor to summary statistics.

    Args:
        tensor: Activation tensor of shape (batch, hidden_dim) or (hidden_dim,)

    Returns:
        Dict with summary stats or None if tensor is None
    """
    if tensor is None:
        return None

    # Convert to numpy and flatten
    arr = tensor.numpy().flatten().astype(np.float64)

    # Compute statistics
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "l2_norm": float(np.linalg.norm(arr)),
        "sparsity": float(np.mean(np.abs(arr) < 0.1)),
        "top_k": _get_top_k(arr, k=5),
        "dim": len(arr),
    }


def _get_top_k(arr: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
    """Get top k activations by absolute value.

    Args:
        arr: 1D numpy array of activations
        k: Number of top activations to return

    Returns:
        List of dicts with idx and val for top k activations
    """
    k = min(k, len(arr))
    indices = np.argsort(np.abs(arr))[-k:][::-1]
    return [{"idx": int(i), "val": float(arr[i])} for i in indices]


def compress_policy(
    logits: torch.Tensor | None,
    mask: torch.Tensor | None,
    action_labels: dict[int, dict[str, Any]] | None = None,
    action_taken: int | None = None,
) -> dict[str, Any] | None:
    """Compress policy output for visualization.

    Args:
        logits: Policy logits tensor (batch, 124) or (124,)
        mask: Legal action mask tensor (batch, 124) or (124,)
        action_labels: Optional dict mapping action index to metadata
        action_taken: Index of action that was taken

    Returns:
        Dict with policy summary or None if logits is None
    """
    if logits is None:
        return None

    # Convert to numpy and flatten
    logits_np = logits.numpy().flatten().astype(np.float64)

    # Handle mask
    if mask is not None:
        mask_np = mask.numpy().flatten().astype(np.float64)
    else:
        mask_np = np.ones_like(logits_np)

    # Apply mask and compute softmax
    masked_logits = np.where(mask_np > 0.5, logits_np, -np.inf)

    # Stable softmax
    max_logit = np.max(masked_logits[np.isfinite(masked_logits)])
    exp_logits = np.exp(masked_logits - max_logit)
    probs = exp_logits / exp_logits.sum()

    # Get legal action indices
    legal_indices = np.where(mask_np > 0.5)[0].tolist()

    # Top 5 by probability
    top_indices = np.argsort(probs)[-5:][::-1]
    top_5 = []
    for i in top_indices:
        if mask_np[i] > 0.5 and np.isfinite(probs[i]):
            # Get label from action_labels dict if available
            if action_labels and int(i) in action_labels:
                meta = action_labels[int(i)]
                label = meta.get("label", f"Action {i}")
                species = meta.get("species", "unknown")
                strength = meta.get("strength", 0)
                points = meta.get("points", 0)
            else:
                label = f"Action {i}"
                species = "unknown"
                strength = 0
                points = 0

            top_5.append({
                "action_idx": int(i),
                "prob": float(probs[i]),
                "label": label,
                "species": species,
                "strength": int(strength),
                "points": int(points),
                "is_selected": bool(i == action_taken),
            })

    # Compute entropy over legal actions
    valid_probs = probs[mask_np > 0.5]
    valid_probs = valid_probs[valid_probs > 1e-10]  # Filter zeros
    entropy = -float(np.sum(valid_probs * np.log(valid_probs)))

    # Confidence (max probability)
    confidence = float(np.max(probs)) if len(probs) > 0 else 0.0

    return {
        "legal_action_count": len(legal_indices),
        "entropy": entropy,
        "confidence": confidence,
        "top_5": top_5,
        "selected_action": action_taken,
    }


def snapshot_to_dict(snapshot: ActivationSnapshot) -> dict[str, Any]:
    """Convert ActivationSnapshot to JSON-serializable dict.

    Args:
        snapshot: The activation snapshot to convert

    Returns:
        Dict suitable for JSON serialization and WebSocket transmission
    """
    return {
        "type": "activation_update",
        "turn": snapshot.turn,
        "player": snapshot.player,
        "timestamp_ms": snapshot.timestamp_ms,
        "zones": {
            "queue": compress_activation(snapshot.queue_rep),
            "bar": compress_activation(snapshot.bar_rep),
            "thats_it": compress_activation(snapshot.thats_it_rep),
            "hand": compress_activation(snapshot.hand_rep),
            "opponent": compress_activation(snapshot.opponent_hand_rep),
        },
        "fusion": {
            "fusion1": compress_activation(snapshot.fusion1_out),
            "fusion2": compress_activation(snapshot.fusion2_out),
            "fusion3": compress_activation(snapshot.fusion3_out),
        },
        "policy": compress_policy(
            snapshot.policy_logits,
            snapshot.legal_mask,
            action_labels=snapshot.action_labels,
            action_taken=snapshot.action_taken,
        ),
        "value": _compress_value(snapshot.value),
        "action": {
            "index": snapshot.action_taken,
            "label": snapshot.action_label,
        } if snapshot.action_taken is not None else None,
        "game_context": snapshot.game_context,
    }


def _compress_value(value: torch.Tensor | None) -> dict[str, Any] | None:
    """Compress value head output.

    Args:
        value: Value tensor (batch, 1) or (1,) or scalar

    Returns:
        Dict with value info or None
    """
    if value is None:
        return None

    val = float(value.numpy().flatten()[0])

    return {
        "estimate": val,
        "win_probability": (val + 1) / 2,  # Map [-1, 1] to [0, 1]
    }
