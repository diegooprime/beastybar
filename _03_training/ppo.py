"""Proximal Policy Optimization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .rollout import RolloutBatch


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    epochs: int = 4
    batch_size: int = 512
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True


def ppo_update(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: PPOConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run a PPO update over the collected rollout batch."""

    observations = batch.observations.to(device)
    actions = batch.actions.to(device)
    old_log_probs = batch.log_probs.to(device)
    advantages = batch.advantages.to(device)
    returns = batch.returns.to(device)
    masks = batch.action_masks.to(device)

    if config.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    num_samples = observations.shape[0]
    batch_size = max(1, min(config.batch_size, num_samples))

    losses = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }

    for _ in range(config.epochs):
        indices = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, batch_size):
            mb_idx = indices[start : start + batch_size]
            mb_obs = observations[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            mb_masks = masks[mb_idx]

            logits, values = model(mb_obs)
            new_log_probs, entropy = _log_prob_and_entropy(logits, mb_actions, mb_masks)

            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)

            surrogate1 = ratio * mb_advantages
            surrogate2 = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * mb_advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = F.mse_loss(values, mb_returns)
            entropy_loss = -entropy.mean()

            optimizer.zero_grad()
            total_loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            losses["policy_loss"] += float(policy_loss.item())
            losses["value_loss"] += float(value_loss.item())
            losses["entropy"] += float(entropy.mean().item())
            approx_kl = torch.mean((ratio - 1.0) - log_ratio).item()
            losses["approx_kl"] += approx_kl
            clip_fraction = (torch.abs(ratio - 1.0) > config.clip_coef).float().mean().item()
            losses["clip_fraction"] += clip_fraction

    updates = config.epochs * max(1, (num_samples + batch_size - 1) // batch_size)
    for key in losses:
        losses[key] /= float(updates)

    return losses


def _log_prob_and_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    masked_logits = logits.masked_fill(~masks, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.softmax(masked_logits, dim=-1)

    action_indices = actions.view(-1, 1)
    selected_log_probs = log_probs.gather(1, action_indices).squeeze(1)
    entropy = -(log_probs * probs).sum(dim=-1)
    return selected_log_probs, entropy


__all__ = ["PPOConfig", "ppo_update"]
