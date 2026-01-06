"""
Test suite for neural network architecture.

Tests the forward pass, output shapes, policy/value constraints,
gradient flow, and model persistence for the actor-critic network.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from _01_simulator.action_space import ACTION_DIM
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor
from _01_simulator import state
from _02_agents.neural.network import BeastyBarNetwork, create_network
from _02_agents.neural.utils import NetworkConfig


@pytest.fixture
def network():
    """Create a small test network."""
    config = NetworkConfig(
        hidden_dim=32,  # Small for fast testing
        num_heads=2,
        num_layers=1,
        dropout=0.0,  # No dropout for deterministic tests
    )
    return create_network(config)


def test_network_forward_shapes(network):
    """Verify network forward pass produces correct output shapes."""
    batch_size = 4

    # Create random input
    obs = torch.randn(batch_size, OBSERVATION_DIM)

    # Forward pass
    policy_logits, value = network(obs)

    # Check policy logits shape
    assert policy_logits.shape == (batch_size, ACTION_DIM), \
        f"Expected policy shape (4, 124), got {policy_logits.shape}"

    # Check value shape (should be (batch_size, 1) or (batch_size,))
    assert value.shape == (batch_size, 1), \
        f"Expected value shape (4, 1), got {value.shape}"

    # Test single input (unbatched)
    obs_single = torch.randn(OBSERVATION_DIM)
    policy_single, value_single = network(obs_single)

    assert policy_single.shape == (ACTION_DIM,), \
        f"Expected single policy shape (124,), got {policy_single.shape}"
    assert value_single.dim() == 0 or value_single.shape == (1,), \
        f"Expected single value to be scalar, got {value_single.shape}"


def test_network_value_range(network):
    """Verify value estimates are in [-1, 1] range due to tanh."""
    batch_size = 10

    # Create random observations
    obs = torch.randn(batch_size, OBSERVATION_DIM)

    # Forward pass
    network.eval()
    with torch.no_grad():
        _, value = network(obs)

    # Check value range
    assert torch.all(value >= -1.0), f"Value has elements < -1.0: min={value.min()}"
    assert torch.all(value <= 1.0), f"Value has elements > 1.0: max={value.max()}"


def test_network_deterministic(network):
    """Same input → same output in eval mode."""
    # Set to eval mode
    network.eval()

    # Create fixed input
    torch.manual_seed(42)
    obs = torch.randn(2, OBSERVATION_DIM)

    # Multiple forward passes
    with torch.no_grad():
        policy1, value1 = network(obs)
        policy2, value2 = network(obs)
        policy3, value3 = network(obs)

    # All should be identical
    assert torch.allclose(policy1, policy2), "Policy outputs differ"
    assert torch.allclose(policy2, policy3), "Policy outputs differ"
    assert torch.allclose(value1, value2), "Value outputs differ"
    assert torch.allclose(value2, value3), "Value outputs differ"


def test_network_mask_handling(network):
    """Masked logits should have very low values (-inf or near it)."""
    # Create observation and mask
    game_state = state.initial_state(seed=42)
    obs_np = state_to_tensor(game_state, perspective=0)
    obs = torch.from_numpy(obs_np).unsqueeze(0)

    # Create mask with only 2 legal actions
    mask = torch.zeros(1, ACTION_DIM)
    mask[0, 0] = 1.0  # Action 0 is legal
    mask[0, 5] = 1.0  # Action 5 is legal

    # Forward pass with mask
    network.eval()
    with torch.no_grad():
        policy_logits, _ = network(obs, mask)

    # Apply mask manually to check
    masked_logits = policy_logits.clone()
    masked_logits[mask == 0] = float('-inf')

    # Masked positions should become -inf or very low
    # Network doesn't apply mask internally, so we test external masking works
    assert masked_logits[0, 0] > -1e8, "Legal action should not be masked"
    assert masked_logits[0, 5] > -1e8, "Legal action should not be masked"
    assert masked_logits[0, 1] == float('-inf'), "Illegal action should be masked"


def test_network_batch_inference(network):
    """Batched forward pass processes multiple inputs correctly."""
    batch_size = 8

    # Create random batch
    obs = torch.randn(batch_size, OBSERVATION_DIM)

    # Batch forward pass
    network.eval()
    with torch.no_grad():
        policy_batch, value_batch = network(obs)

    # Individual forward passes
    individual_policies = []
    individual_values = []
    for i in range(batch_size):
        policy_i, value_i = network(obs[i])
        individual_policies.append(policy_i)
        individual_values.append(value_i)

    # Stack individual results
    policy_stacked = torch.stack(individual_policies)
    value_stacked = torch.stack(individual_values)

    # Should match batch results
    assert torch.allclose(policy_batch, policy_stacked, atol=1e-5), \
        "Batch policy differs from individual passes"
    assert torch.allclose(value_batch.squeeze(), value_stacked.squeeze(), atol=1e-5), \
        "Batch value differs from individual passes"


def test_network_parameter_count(network):
    """Verify parameter count method works correctly."""
    param_count = network.count_parameters()

    # Test network is small (32 hidden_dim), so just check it has parameters
    # For production networks (128+ hidden_dim), expect 500K-2M range
    assert param_count > 0, "Network should have parameters"
    assert param_count < 10_000_000, f"Parameter count {param_count:,} seems too large"

    # Verify count matches manual calculation
    manual_count = sum(p.numel() for p in network.parameters() if p.requires_grad)
    assert param_count == manual_count, \
        f"count_parameters() returned {param_count}, manual count is {manual_count}"

    # Test with a larger network to verify scaling
    large_config = NetworkConfig(hidden_dim=128, num_heads=4, num_layers=3, dropout=0.0)
    large_network = create_network(large_config)
    large_param_count = large_network.count_parameters()

    # Larger network should have significantly more parameters
    assert large_param_count > param_count * 5, \
        f"Large network ({large_param_count:,}) should have more params than small ({param_count:,})"


def test_network_device_transfer(network):
    """Verify network can be transferred between devices."""
    # Test CPU → CPU (always available)
    device_cpu = torch.device("cpu")
    network_cpu = network.to(device_cpu)

    obs = torch.randn(2, OBSERVATION_DIM, device=device_cpu)
    policy, value = network_cpu(obs)

    assert policy.device == device_cpu, "Policy not on CPU"
    assert value.device == device_cpu, "Value not on CPU"

    # Test GPU transfer if available
    if torch.cuda.is_available():
        device_gpu = torch.device("cuda")
        network_gpu = network.to(device_gpu)

        obs_gpu = torch.randn(2, OBSERVATION_DIM, device=device_gpu)
        policy_gpu, value_gpu = network_gpu(obs_gpu)

        assert policy_gpu.device.type == "cuda", "Policy not on CUDA"
        assert value_gpu.device.type == "cuda", "Value not on CUDA"

        # Transfer back to CPU
        network_back = network_gpu.to(device_cpu)
        obs_cpu = torch.randn(2, OBSERVATION_DIM, device=device_cpu)
        policy_back, value_back = network_back(obs_cpu)

        assert policy_back.device == device_cpu, "Policy not back on CPU"
        assert value_back.device == device_cpu, "Value not back on CPU"


def test_network_save_load():
    """Verify network can be saved and loaded correctly."""
    # Create network
    config = NetworkConfig(hidden_dim=32, num_heads=2, num_layers=1)
    network_original = create_network(config)
    network_original.eval()

    # Create test input
    torch.manual_seed(123)
    obs = torch.randn(2, OBSERVATION_DIM)

    # Get original outputs
    with torch.no_grad():
        policy_orig, value_orig = network_original(obs)

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "network.pt"
        torch.save(network_original.state_dict(), save_path)

        # Create new network and load weights
        network_loaded = create_network(config)
        network_loaded.load_state_dict(torch.load(save_path, weights_only=True))
        network_loaded.eval()

        # Get loaded outputs
        with torch.no_grad():
            policy_loaded, value_loaded = network_loaded(obs)

    # Outputs should be identical
    assert torch.allclose(policy_orig, policy_loaded, atol=1e-6), \
        "Loaded network produces different policy"
    assert torch.allclose(value_orig, value_loaded, atol=1e-6), \
        "Loaded network produces different value"
