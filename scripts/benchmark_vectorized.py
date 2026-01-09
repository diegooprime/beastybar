#!/usr/bin/env python3
"""Benchmark script to compare sequential vs vectorized self-play generation.

This script measures:
1. Sequential mode: 1 game at a time, 1 inference per step
2. Vectorized mode: N games in parallel, batched inference

Run on GPU:
    python scripts/benchmark_vectorized.py --device cuda

Expected results:
    - Sequential: High CPU usage, very low GPU utilization (1-18%)
    - Vectorized: Moderate CPU usage, high GPU utilization (80-95%)
"""

import argparse
import logging
import time
from dataclasses import dataclass

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    mode: str
    num_games: int
    total_time: float
    games_per_second: float
    steps_collected: int
    inference_calls: int
    avg_batch_size: float
    steps_per_second: float


def benchmark_sequential(
    network: torch.nn.Module,
    num_games: int,
    device: torch.device,
    temperature: float = 1.0,
) -> BenchmarkResult:
    """Benchmark sequential (legacy) game generation."""
    from _03_training.self_play import generate_games

    logger.info(f"Running SEQUENTIAL benchmark: {num_games} games...")
    start_time = time.time()

    # Force sequential mode
    trajectories = generate_games(
        network=network,
        num_games=num_games,
        temperature=temperature,
        device=device,
        use_vectorized=False,  # Force sequential
    )

    total_time = time.time() - start_time

    # Count total steps
    total_steps = sum(t.total_steps() for t in trajectories)
    # In sequential mode, each step is one inference call
    inference_calls = total_steps

    return BenchmarkResult(
        mode="sequential",
        num_games=num_games,
        total_time=total_time,
        games_per_second=num_games / total_time,
        steps_collected=total_steps,
        inference_calls=inference_calls,
        avg_batch_size=1.0,
        steps_per_second=total_steps / total_time,
    )


def benchmark_vectorized(
    network: torch.nn.Module,
    num_games: int,
    device: torch.device,
    temperature: float = 1.0,
) -> BenchmarkResult:
    """Benchmark vectorized (batched) game generation."""
    from _03_training.vectorized_env import generate_games_vectorized

    logger.info(f"Running VECTORIZED benchmark: {num_games} games...")
    start_time = time.time()

    trajectories, stats = generate_games_vectorized(
        network=network,
        num_games=num_games,
        temperature=temperature,
        device=device,
    )

    total_time = time.time() - start_time

    # Count total steps from trajectories
    total_steps = sum(
        len(t.steps_p0) + len(t.steps_p1)
        for t in trajectories
        if t.final_state is not None
    )

    return BenchmarkResult(
        mode="vectorized",
        num_games=num_games,
        total_time=total_time,
        games_per_second=num_games / total_time,
        steps_collected=total_steps,
        inference_calls=int(stats["inference_calls"]),
        avg_batch_size=stats["avg_batch_size"],
        steps_per_second=total_steps / total_time,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    headers = [
        "Mode", "Games", "Time(s)", "Games/s", "Steps",
        "Infer Calls", "Avg Batch", "Steps/s", "Speedup"
    ]
    print(f"{headers[0]:<12} {headers[1]:>6} {headers[2]:>8} {headers[3]:>8} {headers[4]:>8} "
          f"{headers[5]:>11} {headers[6]:>9} {headers[7]:>10} {headers[8]:>8}")
    print("-" * 80)

    # Find sequential baseline for speedup calculation
    seq_result = next((r for r in results if r.mode == "sequential"), None)
    seq_time = seq_result.total_time if seq_result else 1.0

    for r in results:
        speedup = seq_time / r.total_time if seq_result else 1.0
        print(
            f"{r.mode:<12} {r.num_games:>6} {r.total_time:>8.2f} {r.games_per_second:>8.1f} "
            f"{r.steps_collected:>8} {r.inference_calls:>11} {r.avg_batch_size:>9.1f} "
            f"{r.steps_per_second:>10.1f} {speedup:>7.1f}x"
        )

    print("=" * 80)

    # Summary
    if seq_result:
        vec_result = next((r for r in results if r.mode == "vectorized"), None)
        if vec_result:
            time_reduction = (1 - vec_result.total_time / seq_result.total_time) * 100
            inference_reduction = (1 - vec_result.inference_calls / seq_result.inference_calls) * 100
            print(f"\nSummary:")
            print(f"  Time reduction: {time_reduction:.1f}%")
            print(f"  Inference calls reduction: {inference_reduction:.1f}%")
            print(f"  Batch size improvement: {vec_result.avg_batch_size:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vectorized self-play")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to run on")
    parser.add_argument("--num-games", type=int, default=64,
                       help="Number of games per benchmark")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for action sampling")
    parser.add_argument("--skip-sequential", action="store_true",
                       help="Skip sequential benchmark (useful for large benchmarks)")
    parser.add_argument("--warmup", type=int, default=4,
                       help="Number of warmup games before benchmarking")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Create network
    from _02_agents.neural.network import BeastyBarNetwork
    from _02_agents.neural.utils import NetworkConfig

    config = NetworkConfig(hidden_dim=128, num_layers=3, num_heads=4)
    network = BeastyBarNetwork(config).to(device)
    network.eval()

    param_count = sum(p.numel() for p in network.parameters())
    logger.info(f"Network parameters: {param_count:,}")

    # Warmup
    if args.warmup > 0:
        logger.info(f"Warming up with {args.warmup} games...")
        from _03_training.vectorized_env import generate_games_vectorized
        generate_games_vectorized(network, args.warmup, device=device)

    results = []

    # Run sequential benchmark
    if not args.skip_sequential:
        seq_result = benchmark_sequential(
            network=network,
            num_games=args.num_games,
            device=device,
            temperature=args.temperature,
        )
        results.append(seq_result)

    # Run vectorized benchmark
    vec_result = benchmark_vectorized(
        network=network,
        num_games=args.num_games,
        device=device,
        temperature=args.temperature,
    )
    results.append(vec_result)

    # Print results
    print_results(results)

    # GPU utilization note
    if device.type == "cuda":
        print("\nNote: GPU utilization can be monitored with `nvidia-smi dmon -s u`")
        print("Expected: Sequential ~1-18%, Vectorized ~60-95%")


if __name__ == "__main__":
    main()
