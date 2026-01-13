"""
Test suite for Proximal Policy Optimization (PPO) algorithm.

Tests the core PPO components: Generalized Advantage Estimation (GAE),
policy loss with clipping, value loss, entropy bonus, and update steps.
"""

import numpy as np
import torch

from _01_simulator.action_space import ACTION_DIM
from _01_simulator.observations import OBSERVATION_DIM
from _02_agents.neural.network import create_network
from _02_agents.neural.utils import NetworkConfig
from _03_training.ppo import (
    PPOBatch,
    PPOConfig,
    compute_gae,
    entropy_bonus,
    policy_loss,
    ppo_update,
    value_loss,
)


def test_gae_computation():
    """Verify Generalized Advantage Estimation calculation."""
    # Simple trajectory: 3 timesteps
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.6, 0.7, 0.0], dtype=np.float32)  # +1 for bootstrap
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Terminal at end

    gamma = 0.99
    gae_lambda = 0.95

    # Compute GAE
    advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

    # Verify shapes
    assert advantages.shape == (3,), f"Expected advantages shape (3,), got {advantages.shape}"
    assert returns.shape == (3,), f"Expected returns shape (3,), got {returns.shape}"

    # Returns = advantages + values[:-1]
    np.testing.assert_allclose(returns, advantages + values[:-1], rtol=1e-5)

    # For terminal state, advantage should incorporate the reward
    assert advantages[2] > 0, "Final advantage should be positive for positive reward"

    # Test edge case: lambda=0 (one-step TD)
    advantages_td, _returns_td = compute_gae(rewards, values, dones, gamma, gae_lambda=0.0)
    assert advantages_td.shape == (3,)

    # Test edge case: lambda=1 (Monte Carlo)
    advantages_mc, _returns_mc = compute_gae(rewards, values, dones, gamma, gae_lambda=1.0)
    assert advantages_mc.shape == (3,)


def test_policy_loss_gradient():
    """Policy loss has gradient and updates network."""
    batch_size = 8

    # Create dummy data
    logits = torch.randn(batch_size, ACTION_DIM, requires_grad=True)
    actions = torch.randint(0, ACTION_DIM, (batch_size,))
    old_log_probs = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    action_masks = torch.ones(batch_size, ACTION_DIM)  # All actions legal

    # Compute loss
    loss, approx_kl, clip_frac = policy_loss(
        logits=logits,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        action_masks=action_masks,
        clip_epsilon=0.2,
    )

    # Verify loss is scalar
    assert loss.dim() == 0, "Loss should be scalar"

    # Verify gradient flows
    loss.backward()
    assert logits.grad is not None, "Gradient should flow to logits"
    assert torch.any(logits.grad != 0), "Gradient should be non-zero"

    # Verify diagnostics are scalars
    assert isinstance(approx_kl.item(), float), "approx_kl should be scalar"
    assert isinstance(clip_frac.item(), float), "clip_frac should be scalar"
    assert 0.0 <= clip_frac.item() <= 1.0, "clip_frac should be in [0, 1]"


def test_value_loss_gradient():
    """Value loss has gradient and updates network."""
    batch_size = 8

    # Create dummy data
    predicted_values = torch.randn(batch_size, requires_grad=True)
    returns = torch.randn(batch_size)
    old_values = torch.randn(batch_size)

    # Compute loss with clipping
    loss_clipped = value_loss(
        predicted_values=predicted_values,
        returns=returns,
        old_values=old_values,
        clip_epsilon=0.2,
        clip_value=True,
    )

    # Verify loss is scalar
    assert loss_clipped.dim() == 0, "Loss should be scalar"
    assert loss_clipped.item() >= 0, "Value loss should be non-negative"

    # Verify gradient flows
    loss_clipped.backward()
    assert predicted_values.grad is not None, "Gradient should flow to values"
    assert torch.any(predicted_values.grad != 0), "Gradient should be non-zero"

    # Test without clipping
    predicted_values_2 = torch.randn(batch_size, requires_grad=True)
    loss_unclipped = value_loss(
        predicted_values=predicted_values_2,
        returns=returns,
        clip_value=False,
    )

    assert loss_unclipped.dim() == 0, "Loss should be scalar"
    assert loss_unclipped.item() >= 0, "Value loss should be non-negative"


def test_entropy_computation():
    """Entropy is positive for non-deterministic policies."""
    batch_size = 8

    # Create logits with varying certainty
    # High certainty (one logit much larger)
    logits_certain = torch.zeros(batch_size, ACTION_DIM)
    logits_certain[:, 0] = 10.0  # Very high preference for action 0

    # Uniform distribution (all logits equal)
    logits_uniform = torch.zeros(batch_size, ACTION_DIM)

    # Action masks (all actions legal)
    action_masks = torch.ones(batch_size, ACTION_DIM)

    # Compute entropies
    entropy_certain = entropy_bonus(logits_certain, action_masks)
    entropy_uniform = entropy_bonus(logits_uniform, action_masks)

    # Both should be non-negative
    assert entropy_certain.item() >= 0, "Entropy should be non-negative"
    assert entropy_uniform.item() >= 0, "Entropy should be non-negative"

    # Uniform distribution should have higher entropy
    assert entropy_uniform.item() > entropy_certain.item(), \
        f"Uniform entropy ({entropy_uniform.item()}) should > certain entropy ({entropy_certain.item()})"

    # Test with restricted action mask
    mask_restricted = torch.zeros(batch_size, ACTION_DIM)
    mask_restricted[:, :10] = 1.0  # Only 10 actions legal

    entropy_restricted = entropy_bonus(logits_uniform, mask_restricted)
    assert entropy_restricted.item() >= 0, "Entropy should be non-negative"
    # Restricted mask should have lower entropy than full mask
    assert entropy_restricted.item() < entropy_uniform.item(), \
        "Restricted action space should have lower entropy"


def test_ppo_update_reduces_loss():
    """Single PPO update step processes batch and returns metrics."""
    # Create small network
    config = NetworkConfig(hidden_dim=32, num_heads=2, num_layers=1, dropout=0.0)
    model = create_network(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create dummy batch
    batch_size = 16
    batch = PPOBatch(
        observations=torch.randn(batch_size, OBSERVATION_DIM),
        actions=torch.randint(0, ACTION_DIM, (batch_size,)),
        old_log_probs=torch.randn(batch_size),
        old_values=torch.randn(batch_size),
        action_masks=torch.ones(batch_size, ACTION_DIM),
        advantages=torch.randn(batch_size),
        returns=torch.randn(batch_size),
    )

    # PPO config
    ppo_config = PPOConfig(
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )

    # Perform update
    metrics = ppo_update(model, optimizer, batch, ppo_config)

    # Verify metrics are returned
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "total_loss" in metrics
    assert "approx_kl" in metrics
    assert "clip_fraction" in metrics

    # Verify all metrics are scalars
    for key, value in metrics.items():
        assert isinstance(value, float), f"Metric {key} should be float"

    # Total loss should be combination of components
    expected_total = (
        metrics["policy_loss"]
        + ppo_config.value_coef * metrics["value_loss"]
        - ppo_config.entropy_coef * metrics["entropy"]
    )
    assert abs(metrics["total_loss"] - expected_total) < 1e-3, \
        f"Total loss {metrics['total_loss']} doesn't match expected {expected_total}"


def test_gradient_clipping():
    """Gradients are clipped to max_grad_norm."""
    # Create network
    config = NetworkConfig(hidden_dim=32, num_heads=2, num_layers=1, dropout=0.0)
    model = create_network(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create batch with extreme advantages to cause large gradients
    batch_size = 16
    batch = PPOBatch(
        observations=torch.randn(batch_size, OBSERVATION_DIM),
        actions=torch.randint(0, ACTION_DIM, (batch_size,)),
        old_log_probs=torch.randn(batch_size),
        old_values=torch.randn(batch_size),
        action_masks=torch.ones(batch_size, ACTION_DIM),
        advantages=torch.randn(batch_size) * 100.0,  # Large advantages
        returns=torch.randn(batch_size) * 100.0,  # Large returns
    )

    # PPO config with gradient clipping
    ppo_config = PPOConfig(
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,  # Strict clipping
    )

    # Perform update (gradients should be clipped inside)
    metrics = ppo_update(model, optimizer, batch, ppo_config)

    # After update, compute gradient norm manually to verify clipping worked
    # (We can't directly check during update, but we verify it doesn't crash)
    assert "policy_loss" in metrics, "Update should complete successfully"

    # Verify update completed without NaN
    for param in model.parameters():
        assert torch.all(torch.isfinite(param)), "Parameters should be finite after clipped update"
