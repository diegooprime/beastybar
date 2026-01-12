#!/usr/bin/env python3
"""Benchmark script comparing Python vs Cython game simulation performance.

This script measures the speedup from Cython optimization across:
1. Single game simulation
2. Observation encoding
3. Legal action mask generation
4. Batch operations with parallelization

Usage:
    python scripts/benchmark_cython.py

To build Cython extension first:
    pip install cython
    python _01_simulator/_cython/setup.py build_ext --inplace
"""

import argparse
import os
import sys
import time
from collections.abc import Callable

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _01_simulator import engine
from _01_simulator.action_space import ACTION_DIM, legal_action_mask_tensor
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor
from _01_simulator.state import initial_state

# Try to import Cython
try:
    from _01_simulator._cython._cython_core import (
        encode_single_observation,
        get_single_legal_mask,
        step_single,
    )

    from _01_simulator._cython import (
        GameStateArray,
        encode_observations_parallel,
        get_legal_masks_parallel,
        is_cython_available,
        python_state_to_c,
        step_batch_parallel,
    )

    CYTHON_AVAILABLE = is_cython_available()
except ImportError as e:
    print(f"Cython import error: {e}")
    CYTHON_AVAILABLE = False


def benchmark(func: Callable, iterations: int = 100, warmup: int = 10) -> tuple[float, float]:
    """Run benchmark and return (mean_time, std_time) in milliseconds."""
    # Warmup
    for _ in range(warmup):
        func()

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def benchmark_single_game_python(seed: int = 42, iterations: int = 100) -> dict:
    """Benchmark single game with Python implementation."""

    def run_game():
        state = initial_state(seed=seed)
        steps = 0
        while not engine.is_terminal(state) and steps < 50:
            actions = list(engine.legal_actions(state, state.active_player))
            if not actions:
                break
            action = actions[0]  # Deterministic
            state = engine.step(state, action)
            steps += 1
        return steps

    mean_time, std_time = benchmark(run_game, iterations)
    return {"mean_ms": mean_time, "std_ms": std_time, "iterations": iterations}


def benchmark_single_game_cython(seed: int = 42, iterations: int = 100) -> dict:
    """Benchmark single game with Cython implementation."""
    if not CYTHON_AVAILABLE:
        return {"error": "Cython not available"}

    def run_game():
        c_states = GameStateArray(1)
        c_states.init_game(0, seed)
        steps = 0
        while not c_states.is_terminal(0) and steps < 50:
            mask = get_single_legal_mask(c_states, 0)
            action_idx = int(np.argmax(mask))
            step_single(c_states, 0, action_idx)
            steps += 1
        return steps

    mean_time, std_time = benchmark(run_game, iterations)
    return {"mean_ms": mean_time, "std_ms": std_time, "iterations": iterations}


def benchmark_observation_python(num_states: int = 512, iterations: int = 50) -> dict:
    """Benchmark observation encoding with Python."""
    states = [initial_state(seed=i) for i in range(num_states)]

    def encode_all():
        for state in states:
            state_to_tensor(state, state.active_player)

    mean_time, std_time = benchmark(encode_all, iterations)
    return {
        "mean_ms": mean_time,
        "std_ms": std_time,
        "num_states": num_states,
        "per_state_us": mean_time * 1000 / num_states,
    }


def benchmark_observation_cython(num_states: int = 512, iterations: int = 50, num_threads: int = 1) -> dict:
    """Benchmark observation encoding with Cython."""
    if not CYTHON_AVAILABLE:
        return {"error": "Cython not available"}

    c_states = GameStateArray(num_states)
    for i in range(num_states):
        c_states.init_game(i, i)

    indices = np.arange(num_states, dtype=np.int64)
    output = np.zeros((num_states, OBSERVATION_DIM), dtype=np.float32)

    def encode_all():
        encode_observations_parallel(c_states, indices, output, num_threads)

    mean_time, std_time = benchmark(encode_all, iterations)
    return {
        "mean_ms": mean_time,
        "std_ms": std_time,
        "num_states": num_states,
        "num_threads": num_threads,
        "per_state_us": mean_time * 1000 / num_states,
    }


def benchmark_legal_mask_python(num_states: int = 512, iterations: int = 50) -> dict:
    """Benchmark legal action mask generation with Python."""
    states = [initial_state(seed=i) for i in range(num_states)]

    def generate_all():
        for state in states:
            legal_action_mask_tensor(state, state.active_player)

    mean_time, std_time = benchmark(generate_all, iterations)
    return {
        "mean_ms": mean_time,
        "std_ms": std_time,
        "num_states": num_states,
        "per_state_us": mean_time * 1000 / num_states,
    }


def benchmark_legal_mask_cython(num_states: int = 512, iterations: int = 50, num_threads: int = 1) -> dict:
    """Benchmark legal action mask generation with Cython."""
    if not CYTHON_AVAILABLE:
        return {"error": "Cython not available"}

    c_states = GameStateArray(num_states)
    for i in range(num_states):
        c_states.init_game(i, i)

    indices = np.arange(num_states, dtype=np.int64)
    output = np.zeros((num_states, ACTION_DIM), dtype=np.float32)

    def generate_all():
        get_legal_masks_parallel(c_states, indices, output, num_threads)

    mean_time, std_time = benchmark(generate_all, iterations)
    return {
        "mean_ms": mean_time,
        "std_ms": std_time,
        "num_states": num_states,
        "num_threads": num_threads,
        "per_state_us": mean_time * 1000 / num_states,
    }


def benchmark_batch_step_cython(num_states: int = 512, iterations: int = 50, num_threads: int = 1) -> dict:
    """Benchmark batch stepping with Cython."""
    if not CYTHON_AVAILABLE:
        return {"error": "Cython not available"}

    # Initialize games
    c_states = GameStateArray(num_states)
    for i in range(num_states):
        c_states.init_game(i, i)

    indices = np.arange(num_states, dtype=np.int64)
    masks = np.zeros((num_states, ACTION_DIM), dtype=np.float32)
    get_legal_masks_parallel(c_states, indices, masks, num_threads)

    # Pre-compute actions
    actions = np.array([int(np.argmax(masks[i])) for i in range(num_states)], dtype=np.int64)

    def step_all():
        # Re-init games for fair comparison
        for i in range(num_states):
            c_states.init_game(i, i)
        step_batch_parallel(c_states, indices, actions, num_threads)

    mean_time, std_time = benchmark(step_all, iterations)
    return {
        "mean_ms": mean_time,
        "std_ms": std_time,
        "num_states": num_states,
        "num_threads": num_threads,
        "per_state_us": mean_time * 1000 / num_states,
    }


def print_results(name: str, results: dict, baseline: dict | None = None) -> None:
    """Print benchmark results with optional speedup calculation."""
    if "error" in results:
        print(f"  {name}: {results['error']}")
        return

    speedup = ""
    if baseline and "mean_ms" in baseline:
        ratio = baseline["mean_ms"] / results["mean_ms"]
        speedup = f" (speedup: {ratio:.1f}x)"

    print(f"  {name}:")
    print(f"    Time: {results['mean_ms']:.2f} +/- {results['std_ms']:.2f} ms{speedup}")
    if "per_state_us" in results:
        print(f"    Per state: {results['per_state_us']:.1f} us")
    if "num_threads" in results:
        print(f"    Threads: {results['num_threads']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Python vs Cython performance")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for parallel ops")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads (default: all)")
    args = parser.parse_args()

    num_threads = args.threads or os.cpu_count() or 4

    print("=" * 60)
    print("BeastyBar Cython Optimization Benchmark")
    print("=" * 60)
    print(f"Cython available: {CYTHON_AVAILABLE}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Threads for parallel ops: {num_threads}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print()

    # Single game simulation
    print("Single Game Simulation (deterministic play):")
    py_single = benchmark_single_game_python(iterations=args.iterations)
    print_results("Python", py_single)

    if CYTHON_AVAILABLE:
        cy_single = benchmark_single_game_cython(iterations=args.iterations)
        print_results("Cython", cy_single, py_single)
    print()

    # Observation encoding
    print(f"Observation Encoding ({args.batch_size} states):")
    py_obs = benchmark_observation_python(num_states=args.batch_size, iterations=args.iterations)
    print_results("Python", py_obs)

    if CYTHON_AVAILABLE:
        cy_obs_1 = benchmark_observation_cython(num_states=args.batch_size, iterations=args.iterations, num_threads=1)
        print_results("Cython (1 thread)", cy_obs_1, py_obs)

        cy_obs_n = benchmark_observation_cython(
            num_states=args.batch_size, iterations=args.iterations, num_threads=num_threads
        )
        print_results(f"Cython ({num_threads} threads)", cy_obs_n, py_obs)
    print()

    # Legal action mask
    print(f"Legal Action Mask Generation ({args.batch_size} states):")
    py_mask = benchmark_legal_mask_python(num_states=args.batch_size, iterations=args.iterations)
    print_results("Python", py_mask)

    if CYTHON_AVAILABLE:
        cy_mask_1 = benchmark_legal_mask_cython(num_states=args.batch_size, iterations=args.iterations, num_threads=1)
        print_results("Cython (1 thread)", cy_mask_1, py_mask)

        cy_mask_n = benchmark_legal_mask_cython(
            num_states=args.batch_size, iterations=args.iterations, num_threads=num_threads
        )
        print_results(f"Cython ({num_threads} threads)", cy_mask_n, py_mask)
    print()

    # Batch stepping (Cython only)
    if CYTHON_AVAILABLE:
        print(f"Batch Step ({args.batch_size} games):")
        cy_step_1 = benchmark_batch_step_cython(num_states=args.batch_size, iterations=args.iterations, num_threads=1)
        print_results("Cython (1 thread)", cy_step_1)

        cy_step_n = benchmark_batch_step_cython(
            num_states=args.batch_size, iterations=args.iterations, num_threads=num_threads
        )
        print_results(f"Cython ({num_threads} threads)", cy_step_n, cy_step_1)
        print()

    # Summary
    if CYTHON_AVAILABLE:
        print("=" * 60)
        print("Summary - Expected Training Iteration Speedup")
        print("=" * 60)

        # Calculate estimated iteration time
        py_iter_time = py_obs["mean_ms"] + py_mask["mean_ms"]  # Rough estimate
        cy_iter_time = cy_obs_n["mean_ms"] + cy_mask_n["mean_ms"]

        print(f"Python batch encoding: {py_iter_time:.1f} ms")
        print(f"Cython batch encoding: {cy_iter_time:.1f} ms")
        print(f"Estimated CPU speedup: {py_iter_time/cy_iter_time:.1f}x")
        print()
        print("Note: GPU inference time unchanged. Total iteration speedup depends")
        print("on the ratio of CPU vs GPU time in your training loop.")


if __name__ == "__main__":
    main()
