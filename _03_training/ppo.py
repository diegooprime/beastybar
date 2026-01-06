"""Proximal Policy Optimization (PPO) algorithm implementation.

This module provides:
- Generalized Advantage Estimation (GAE) for advantage computation
- Clipped surrogate policy loss
- Clipped value loss
- Entropy bonus for exploration
- Complete PPO update step with gradient clipping

References:
    - PPO paper: https://arxiv.org/abs/1707.06347
    - GAE paper: https://arxiv.org/abs/1506.02438
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

# Conditional PyTorch import with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    functional = None
    Tensor = None


def _ensure_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PPO training. Install with: pip install torch")


@dataclass
class PPOConfig:
    """Configuration for PPO training.

    Attributes:
        learning_rate: Optimizer learning rate.
        gamma: Discount factor for future rewards.
        gae_lambda: Lambda for GAE bias-variance tradeoff.
        clip_epsilon: Clipping parameter for policy ratio.
        value_coef: Coefficient for value loss in total loss.
        entropy_coef: Coefficient for entropy bonus in total loss.
        max_grad_norm: Maximum gradient norm for clipping.
        ppo_epochs: Number of optimization epochs per batch.
        minibatch_size: Size of minibatches for SGD.
        clip_value: Whether to clip value function updates.
        normalize_advantages: Whether to normalize advantages.
        target_kl: Optional early stopping KL divergence threshold.
    """

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 64
    clip_value: bool = True
    normalize_advantages: bool = True
    target_kl: float | None = None


@dataclass
class PPOBatch:
    """A batch of experience data for PPO training (PyTorch tensors).

    All tensors should have shape (batch_size, ...) where ... depends on the field.
    This is distinct from replay_buffer.Batch which uses numpy arrays.
    """

    observations: Tensor  # (batch_size, obs_dim)
    actions: Tensor  # (batch_size,) - action indices
    old_log_probs: Tensor  # (batch_size,) - log prob of taken action
    old_values: Tensor  # (batch_size,) - value estimates at collection time
    action_masks: Tensor  # (batch_size, num_actions) - legal action masks
    advantages: Tensor = field(default=None)  # type: ignore[assignment]  # (batch_size,) - computed GAE advantages
    returns: Tensor = field(default=None)  # type: ignore[assignment]  # (batch_size,) - target returns

    def __len__(self) -> int:
        """Return batch size."""
        return int(self.observations.shape[0])

    def to(self, device: torch.device) -> PPOBatch:
        """Move all tensors to the specified device."""
        _ensure_torch()
        return PPOBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            old_log_probs=self.old_log_probs.to(device),
            old_values=self.old_values.to(device),
            action_masks=self.action_masks.to(device),
            advantages=self.advantages.to(device) if self.advantages is not None else None,
            returns=self.returns.to(device) if self.returns is not None else None,
        )


class PolicyValueNetwork(Protocol):
    """Protocol for neural networks compatible with PPO training.

    Networks must implement forward() returning policy logits and value estimate.
    """

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass returning policy logits and value estimate.

        Args:
            observations: Batch of observation tensors.

        Returns:
            Tuple of (policy_logits, value_estimate) where:
                - policy_logits: (batch_size, num_actions) unnormalized log probs
                - value_estimate: (batch_size,) state value estimates
        """
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return iterator over model parameters."""
        ...


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides a family of advantage estimators that trade off bias and variance.
    Setting gae_lambda=1 gives high variance, low bias (Monte Carlo).
    Setting gae_lambda=0 gives low variance, high bias (1-step TD).

    Args:
        rewards: Array of rewards, shape (T,) or (T, num_envs).
        values: Array of value estimates, shape (T+1,) or (T+1, num_envs).
                The last value is the bootstrap value for the final state.
        dones: Array of done flags, shape (T,) or (T, num_envs).
               1.0 = episode ended, 0.0 = episode continues.
        gamma: Discount factor for future rewards.
        gae_lambda: GAE lambda for bias-variance tradeoff.

    Returns:
        Tuple of (advantages, returns) arrays with same shape as rewards.
        - advantages: GAE advantage estimates
        - returns: Target returns (advantages + values[:-1])

    Example:
        >>> rewards = np.array([1.0, 0.0, 1.0])
        >>> values = np.array([0.5, 0.6, 0.7, 0.0])  # includes bootstrap
        >>> dones = np.array([0.0, 0.0, 1.0])
        >>> advantages, returns = compute_gae(rewards, values, dones)
    """
    rewards = np.asarray(rewards)
    values = np.asarray(values)
    dones = np.asarray(dones)

    # Handle both 1D (single env) and 2D (multiple envs) cases
    original_shape = rewards.shape
    if rewards.ndim == 1:
        rewards = rewards[:, np.newaxis]
        values = values[:, np.newaxis]
        dones = dones[:, np.newaxis]

    num_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(rewards.shape[1])

    # Compute GAE backwards through time
    for t in reversed(range(num_steps)):
        # Mask for non-terminal states
        non_terminal = 1.0 - dones[t]

        # TD error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]

        # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    # Returns = advantages + values (target for value function)
    returns = advantages + values[:-1]

    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)

    return advantages, returns


def policy_loss(
    logits: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    action_masks: Tensor,
    clip_epsilon: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute PPO clipped surrogate policy loss.

    The clipped objective prevents large policy updates by clipping the
    probability ratio between new and old policies.

    Args:
        logits: Policy network output logits, shape (batch_size, num_actions).
        actions: Actions taken, shape (batch_size,).
        old_log_probs: Log probabilities under old policy, shape (batch_size,).
        advantages: Advantage estimates, shape (batch_size,).
        action_masks: Binary mask for legal actions, shape (batch_size, num_actions).
                      1 = legal, 0 = illegal.
        clip_epsilon: Clipping parameter epsilon.

    Returns:
        Tuple of (loss, approx_kl, clip_fraction) where:
            - loss: Scalar policy loss (to be minimized)
            - approx_kl: Approximate KL divergence for monitoring
            - clip_fraction: Fraction of samples that were clipped

    Note:
        The loss is the negative of the clipped surrogate objective because
        we minimize losses but want to maximize the objective.
    """
    _ensure_torch()

    # Apply action mask: set illegal action logits to large negative value
    masked_logits = logits.clone()
    masked_logits[action_masks == 0] = -1e8

    # Compute log probabilities under current policy
    log_probs = functional.log_softmax(masked_logits, dim=-1)
    current_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute probability ratio
    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Clipped surrogate objective
    # L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages

    # Take minimum (pessimistic bound)
    # Negative because we minimize loss but want to maximize objective
    loss = -torch.min(surrogate1, surrogate2).mean()

    # Compute diagnostics
    with torch.no_grad():
        # Approximate KL divergence: E[(r - 1) - log(r)]
        approx_kl = ((ratio - 1) - log_ratio).mean()

        # Fraction of samples where clipping was applied
        clip_fraction = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()

    return loss, approx_kl, clip_fraction


def value_loss(
    predicted_values: Tensor,
    returns: Tensor,
    old_values: Tensor | None = None,
    clip_epsilon: float = 0.2,
    clip_value: bool = True,
) -> Tensor:
    """Compute value function loss.

    Optionally clips value predictions to prevent large updates, similar to
    the policy clipping in PPO.

    Args:
        predicted_values: Value estimates from current network, shape (batch_size,).
        returns: Target returns (from GAE), shape (batch_size,).
        old_values: Value estimates from old network, shape (batch_size,).
                    Required if clip_value=True.
        clip_epsilon: Clipping parameter for value updates.
        clip_value: Whether to apply clipping to value loss.

    Returns:
        Scalar value loss (MSE or clipped MSE).

    Raises:
        ValueError: If clip_value=True but old_values is None.
    """
    _ensure_torch()

    if clip_value:
        if old_values is None:
            raise ValueError("old_values required when clip_value=True")

        # Clipped value loss: prevents large value updates
        # V_clip = V_old + clip(V - V_old, -eps, +eps)
        values_clipped = old_values + torch.clamp(predicted_values - old_values, -clip_epsilon, clip_epsilon)

        # Take maximum of unclipped and clipped losses
        value_loss_unclipped = (predicted_values - returns) ** 2
        value_loss_clipped = (values_clipped - returns) ** 2
        loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
    else:
        # Simple MSE loss
        loss = 0.5 * functional.mse_loss(predicted_values, returns)

    return loss


def entropy_bonus(
    logits: Tensor,
    action_masks: Tensor,
) -> Tensor:
    """Compute policy entropy for exploration bonus.

    Higher entropy means more exploration. The entropy bonus encourages
    the policy to remain stochastic and explore diverse actions.

    Args:
        logits: Policy network output logits, shape (batch_size, num_actions).
        action_masks: Binary mask for legal actions, shape (batch_size, num_actions).
                      1 = legal, 0 = illegal.

    Returns:
        Mean entropy across the batch (scalar).
    """
    _ensure_torch()

    # Apply action mask
    masked_logits = logits.clone()
    masked_logits[action_masks == 0] = -1e8

    # Compute probabilities
    probs = functional.softmax(masked_logits, dim=-1)
    log_probs = functional.log_softmax(masked_logits, dim=-1)

    # Entropy: H = -sum(p * log(p))
    # Only sum over legal actions (masked ones have p=0)
    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy.mean()


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    config: PPOConfig,
) -> dict[str, float]:
    """Perform a single PPO update iteration over a batch.

    This function computes and applies gradients for one minibatch of data.

    Args:
        model: Neural network implementing PolicyValueNetwork protocol.
        optimizer: PyTorch optimizer for the model.
        batch: PPOBatch of experience data with precomputed advantages and returns.
        config: PPO hyperparameters.

    Returns:
        Dictionary of loss metrics:
            - policy_loss: Policy loss value
            - value_loss: Value loss value
            - entropy: Policy entropy value
            - total_loss: Combined loss value
            - approx_kl: Approximate KL divergence
            - clip_fraction: Fraction of clipped samples
    """
    _ensure_torch()

    # Forward pass
    policy_logits, values = model.forward(batch.observations)
    values = values.squeeze(-1)  # (batch_size,)

    # Normalize advantages if configured
    advantages = batch.advantages
    if config.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute losses
    p_loss, approx_kl, clip_frac = policy_loss(
        logits=policy_logits,
        actions=batch.actions,
        old_log_probs=batch.old_log_probs,
        advantages=advantages,
        action_masks=batch.action_masks,
        clip_epsilon=config.clip_epsilon,
    )

    v_loss = value_loss(
        predicted_values=values,
        returns=batch.returns,
        old_values=batch.old_values,
        clip_epsilon=config.clip_epsilon,
        clip_value=config.clip_value,
    )

    ent_bonus = entropy_bonus(
        logits=policy_logits,
        action_masks=batch.action_masks,
    )

    # Combined loss: policy + value_coef * value - entropy_coef * entropy
    total_loss = p_loss + config.value_coef * v_loss - config.entropy_coef * ent_bonus

    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping
    if config.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

    optimizer.step()

    return {
        "policy_loss": float(p_loss.item()),
        "value_loss": float(v_loss.item()),
        "entropy": float(ent_bonus.item()),
        "total_loss": float(total_loss.item()),
        "approx_kl": float(approx_kl.item()),
        "clip_fraction": float(clip_frac.item()),
    }


@dataclass
class RolloutData:
    """Container for collected rollout data before processing.

    This is the raw data collected during self-play before computing
    advantages and returns.
    """

    observations: np.ndarray  # (T, obs_dim)
    actions: np.ndarray  # (T,)
    log_probs: np.ndarray  # (T,)
    values: np.ndarray  # (T,)
    rewards: np.ndarray  # (T,)
    dones: np.ndarray  # (T,)
    action_masks: np.ndarray  # (T, num_actions)


def process_rollout(
    rollout: RolloutData,
    last_value: float,
    config: PPOConfig,
) -> PPOBatch:
    """Process raw rollout data into a training batch.

    Computes GAE advantages and returns, then converts to PyTorch tensors.

    Args:
        rollout: Raw rollout data from self-play.
        last_value: Bootstrap value for the final state (0 if terminal).
        config: PPO configuration for GAE parameters.

    Returns:
        PPOBatch ready for PPO training.
    """
    _ensure_torch()

    # Append bootstrap value for GAE computation
    values_with_bootstrap = np.append(rollout.values, last_value)

    # Compute advantages and returns
    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=values_with_bootstrap,
        dones=rollout.dones,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    # Convert to tensors
    return PPOBatch(
        observations=torch.from_numpy(rollout.observations).float(),
        actions=torch.from_numpy(rollout.actions).long(),
        old_log_probs=torch.from_numpy(rollout.log_probs).float(),
        old_values=torch.from_numpy(rollout.values).float(),
        action_masks=torch.from_numpy(rollout.action_masks).float(),
        advantages=torch.from_numpy(advantages).float(),
        returns=torch.from_numpy(returns).float(),
    )


def iterate_minibatches(
    batch: PPOBatch,
    minibatch_size: int,
    shuffle: bool = True,
) -> Iterator[PPOBatch]:
    """Iterate over minibatches of a larger batch.

    Args:
        batch: Full PPOBatch of data.
        minibatch_size: Size of each minibatch.
        shuffle: Whether to shuffle indices before iterating.

    Yields:
        Minibatches of the requested size.
    """
    _ensure_torch()

    batch_size = len(batch)
    indices = np.arange(batch_size)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, batch_size, minibatch_size):
        end = min(start + minibatch_size, batch_size)
        mb_indices = indices[start:end]

        yield PPOBatch(
            observations=batch.observations[mb_indices],
            actions=batch.actions[mb_indices],
            old_log_probs=batch.old_log_probs[mb_indices],
            old_values=batch.old_values[mb_indices],
            action_masks=batch.action_masks[mb_indices],
            advantages=batch.advantages[mb_indices],
            returns=batch.returns[mb_indices],
        )


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    config: PPOConfig,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Execute full PPO training step over collected data.

    Performs multiple epochs of minibatch updates over the same data,
    as is standard in PPO.

    Args:
        model: Neural network implementing PolicyValueNetwork protocol.
        optimizer: PyTorch optimizer for the model.
        batch: Full PPOBatch of experience data with advantages and returns.
        config: PPO hyperparameters.
        device: Device to run training on. If None, uses model's device.

    Returns:
        Dictionary of aggregated loss metrics (averaged over all updates):
            - policy_loss: Mean policy loss
            - value_loss: Mean value loss
            - entropy: Mean policy entropy
            - total_loss: Mean combined loss
            - approx_kl: Mean approximate KL divergence
            - clip_fraction: Mean fraction of clipped samples
            - num_updates: Total number of minibatch updates performed
    """
    _ensure_torch()

    # Determine device
    if device is None:
        device = next(model.parameters()).device

    # Move batch to device
    batch = batch.to(device)

    # Accumulate metrics across all updates
    metrics_sum: dict[str, float] = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    num_updates = 0
    early_stop = False

    # Multiple epochs over the same data
    epochs_completed = 0
    for _epoch in range(config.ppo_epochs):
        if early_stop:
            break

        epochs_completed = _epoch + 1

        # Iterate over minibatches
        for minibatch in iterate_minibatches(batch, config.minibatch_size, shuffle=True):
            metrics = ppo_update(
                model=model,
                optimizer=optimizer,
                batch=minibatch,
                config=config,
            )

            # Accumulate metrics
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]
            num_updates += 1

            # Early stopping based on KL divergence
            if config.target_kl is not None and metrics["approx_kl"] > 1.5 * config.target_kl:
                early_stop = True
                break

    # Average metrics
    if num_updates > 0:
        for key in metrics_sum:
            metrics_sum[key] /= num_updates

    metrics_sum["num_updates"] = float(num_updates)
    metrics_sum["epochs_completed"] = float(epochs_completed)

    return metrics_sum


__all__ = [
    "TORCH_AVAILABLE",
    "PPOBatch",
    "PPOConfig",
    "PolicyValueNetwork",
    "RolloutData",
    "compute_gae",
    "entropy_bonus",
    "iterate_minibatches",
    "policy_loss",
    "ppo_update",
    "process_rollout",
    "train_step",
    "value_loss",
]
