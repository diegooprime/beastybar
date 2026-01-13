"""
Performance benchmarks for neural network training components.

These tests measure latency, throughput, and efficiency of key operations.
Run with: pytest test_benchmarks.py -m benchmark -v
"""

import time

import pytest
import torch

from _01_simulator import state
from _01_simulator.action_space import legal_action_mask_tensor
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor
from _02_agents.neural.network import create_network
from _02_agents.neural.utils import NetworkConfig
from _03_training.self_play import generate_games, play_game


@pytest.mark.benchmark
def test_inference_latency():
    """Measure time per action selection.

    Target: <10ms per action on CPU
    """
    # Create network
    config = NetworkConfig(hidden_dim=64, num_heads=4, num_layers=2)
    network = create_network(config)
    network.eval()

    # Prepare test data
    game_state = state.initial_state(seed=42)
    obs_np = state_to_tensor(game_state, perspective=0)
    obs = torch.from_numpy(obs_np).unsqueeze(0)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = network(obs)

    # Benchmark
    num_inferences = 100
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_inferences):
            _policy_logits, _value = network(obs)

    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (elapsed / num_inferences) * 1000

    print(f"\nInference latency: {avg_latency_ms:.2f}ms per action")

    # Target: <10ms on CPU (may vary by hardware)
    # This is a soft check - we mainly report the result
    if avg_latency_ms > 20:
        pytest.fail(f"Inference latency ({avg_latency_ms:.2f}ms) is too high")


@pytest.mark.benchmark
def test_game_generation_throughput():
    """Measure games per second.

    Target: 10+ games/second on CPU
    """
    # Create network
    config = NetworkConfig(hidden_dim=32, num_heads=2, num_layers=1)
    network = create_network(config)

    # Warmup
    _ = play_game(network, seed=0, temperature=1.0)

    # Benchmark
    num_games = 10
    start_time = time.perf_counter()

    generate_games(
        network=network,
        num_games=num_games,
        temperature=1.0,
        seeds=list(range(num_games)),
    )

    elapsed = time.perf_counter() - start_time
    games_per_sec = num_games / elapsed

    print(f"\nGame generation: {games_per_sec:.2f} games/second")
    print(f"Average game time: {elapsed / num_games:.3f}s")

    # Target: 10+ games/second (soft check)
    if games_per_sec < 5:
        pytest.fail(f"Game generation ({games_per_sec:.2f} games/s) is too slow")


@pytest.mark.benchmark
def test_batch_inference_speedup():
    """Verify batching provides speedup over sequential inference."""
    # Create network
    config = NetworkConfig(hidden_dim=64, num_heads=4, num_layers=2)
    network = create_network(config)
    network.eval()

    batch_size = 32

    # Prepare batch data
    obs_batch = torch.randn(batch_size, OBSERVATION_DIM)

    # Sequential inference
    with torch.no_grad():
        start_seq = time.perf_counter()
        for i in range(batch_size):
            _ = network(obs_batch[i])
        elapsed_seq = time.perf_counter() - start_seq

    # Batched inference
    with torch.no_grad():
        start_batch = time.perf_counter()
        _ = network(obs_batch)
        elapsed_batch = time.perf_counter() - start_batch

    speedup = elapsed_seq / elapsed_batch

    print(f"\nSequential time: {elapsed_seq:.4f}s")
    print(f"Batch time: {elapsed_batch:.4f}s")
    print(f"Speedup: {speedup:.2f}x")

    # Batching should provide at least 2x speedup
    assert speedup >= 2.0, \
        f"Batching speedup ({speedup:.2f}x) should be at least 2x"


@pytest.mark.benchmark
def test_observation_encoding_speed():
    """Measure state→tensor encoding time.

    Target: <1ms per encoding
    """
    # Create test states
    states = [state.initial_state(seed=i) for i in range(100)]

    # Warmup
    for game_state in states[:10]:
        _ = state_to_tensor(game_state, perspective=0)

    # Benchmark
    num_encodings = 100
    start_time = time.perf_counter()

    for i in range(num_encodings):
        _ = state_to_tensor(states[i], perspective=0)

    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (elapsed / num_encodings) * 1000

    print(f"\nObservation encoding: {avg_latency_ms:.3f}ms per encoding")

    # Target: <1ms (soft check, depends on hardware)
    if avg_latency_ms > 2.0:
        pytest.fail(f"Encoding latency ({avg_latency_ms:.3f}ms) is too high")


@pytest.mark.benchmark
def test_action_mask_generation_speed():
    """Measure action mask generation speed."""
    # Create test states
    states = [state.initial_state(seed=i) for i in range(100)]

    # Warmup
    for game_state in states[:10]:
        _ = legal_action_mask_tensor(game_state, perspective=0)

    # Benchmark
    num_masks = 100
    start_time = time.perf_counter()

    for i in range(num_masks):
        _ = legal_action_mask_tensor(states[i], perspective=0)

    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (elapsed / num_masks) * 1000

    print(f"\nAction mask generation: {avg_latency_ms:.3f}ms per mask")

    # Should be very fast
    if avg_latency_ms > 1.0:
        pytest.fail(f"Mask generation ({avg_latency_ms:.3f}ms) is too slow")
