"""Factory helpers for SelfPlayRLAgent manifests."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch

from _01_simulator import observations

from . import encoders, models


def load_policy(
    *,
    checkpoint: str | Path,
    device: str = "cpu",
) -> Callable[[observations.Observation], Sequence[float]]:
    """Return a callable that maps observations to policy logits."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, _, payload = models.load_checkpoint(checkpoint_path, device=device)
    model.eval()

    torch_device = torch.device(device)

    def _run(obs: observations.Observation) -> Sequence[float]:
        with torch.no_grad():
            encoded = encoders.encode_observation(obs).to(torch_device)
            logits, _ = model(encoded)
            return logits.squeeze(0).cpu().tolist()

    return _run


__all__ = ["load_policy"]
