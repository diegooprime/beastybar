"""Performance benchmarking suite for Beasty Bar AI models.

This module provides comprehensive benchmarking utilities for measuring:
- Inference latency (single and batch)
- Throughput (games per second)
- Memory usage
- Model size comparisons
- Cross-format comparisons (PyTorch, ONNX, quantized)

Example:
    from _03_training.benchmark import benchmark_model, generate_benchmark_report

    # Benchmark a PyTorch model
    results = benchmark_model(network, device="cuda")

    # Generate comprehensive report
    report = generate_benchmark_report(network, output_dir="benchmarks/")
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

# Constants
OBSERVATION_DIM = 988
ACTION_DIM = 124


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        num_iterations: Number of inference iterations for timing.
        warmup_iterations: Number of warmup iterations to stabilize timings.
        batch_sizes: List of batch sizes to test.
        include_memory: Whether to measure memory usage.
        include_games: Whether to benchmark full game simulation.
        games_per_test: Number of games to simulate for throughput test.
    """

    num_iterations: int = 1000
    warmup_iterations: int = 100
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32, 64, 128])
    include_memory: bool = True
    include_games: bool = True
    games_per_test: int = 100


@dataclass
class LatencyMetrics:
    """Latency metrics from benchmarking.

    Attributes:
        batch_size: Batch size used for this measurement.
        avg_ms: Average latency in milliseconds.
        std_ms: Standard deviation of latency.
        p50_ms: 50th percentile (median) latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        min_ms: Minimum latency observed.
        max_ms: Maximum latency observed.
    """

    batch_size: int
    avg_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class MemoryMetrics:
    """Memory usage metrics.

    Attributes:
        model_size_mb: Size of model parameters in MB.
        inference_peak_mb: Peak memory during inference.
        gpu_allocated_mb: GPU memory allocated (if applicable).
        gpu_cached_mb: GPU memory cached (if applicable).
    """

    model_size_mb: float
    inference_peak_mb: float
    gpu_allocated_mb: float | None = None
    gpu_cached_mb: float | None = None


@dataclass
class ThroughputMetrics:
    """Throughput metrics.

    Attributes:
        inferences_per_second: Raw inference calls per second.
        games_per_second: Complete games simulated per second.
        batched_throughput: Dictionary of batch_size -> throughput.
    """

    inferences_per_second: float
    games_per_second: float | None
    batched_throughput: dict[int, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark results.

    Attributes:
        model_name: Name/identifier for the model.
        model_format: Format (pytorch, onnx, quantized, etc.).
        device: Device used for benchmarking.
        latency: Latency metrics for each batch size.
        memory: Memory usage metrics.
        throughput: Throughput metrics.
        timestamp: When benchmark was run.
    """

    model_name: str
    model_format: str
    device: str
    latency: list[LatencyMetrics]
    memory: MemoryMetrics | None
    throughput: ThroughputMetrics | None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


def _measure_latency(
    inference_fn: callable,
    batch_size: int,
    num_iterations: int,
    warmup: int,
) -> LatencyMetrics:
    """Measure inference latency.

    Args:
        inference_fn: Function that takes observation tensor and returns outputs.
        batch_size: Batch size to test.
        num_iterations: Number of iterations for timing.
        warmup: Number of warmup iterations.

    Returns:
        LatencyMetrics with timing statistics.
    """
    import torch

    # Create random input
    obs = torch.randn(batch_size, OBSERVATION_DIM)

    # Try to move to same device as model
    try:
        obs = obs.cuda() if torch.cuda.is_available() else obs
    except Exception:
        pass

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = inference_fn(obs)

    # Synchronize before timing (for GPU)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = inference_fn(obs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    return LatencyMetrics(
        batch_size=batch_size,
        avg_ms=float(np.mean(latencies)),
        std_ms=float(np.std(latencies)),
        p50_ms=float(np.percentile(latencies, 50)),
        p95_ms=float(np.percentile(latencies, 95)),
        p99_ms=float(np.percentile(latencies, 99)),
        min_ms=float(np.min(latencies)),
        max_ms=float(np.max(latencies)),
    )


def _measure_memory(network: nn.Module) -> MemoryMetrics:
    """Measure memory usage.

    Args:
        network: PyTorch network to measure.

    Returns:
        MemoryMetrics with memory statistics.
    """
    import torch

    # Model parameter size
    param_size = sum(p.numel() * p.element_size() for p in network.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in network.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    # GPU memory if available
    gpu_allocated = None
    gpu_cached = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Run inference to measure peak
        obs = torch.randn(1, OBSERVATION_DIM).cuda()
        with torch.no_grad():
            _ = network(obs)

        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_cached = torch.cuda.memory_reserved() / (1024 * 1024)

    # Estimate inference peak (rough estimate)
    inference_peak = model_size_mb * 2  # Activations roughly double model size

    return MemoryMetrics(
        model_size_mb=model_size_mb,
        inference_peak_mb=inference_peak,
        gpu_allocated_mb=gpu_allocated,
        gpu_cached_mb=gpu_cached,
    )


def _measure_throughput(
    network: nn.Module,
    config: BenchmarkConfig,
) -> ThroughputMetrics:
    """Measure throughput.

    Args:
        network: PyTorch network to benchmark.
        config: Benchmark configuration.

    Returns:
        ThroughputMetrics with throughput statistics.
    """
    import torch

    # Single inference throughput
    obs = torch.randn(1, OBSERVATION_DIM)
    try:
        device = next(network.parameters()).device
        obs = obs.to(device)
    except StopIteration:
        pass

    # Warmup
    for _ in range(config.warmup_iterations):
        with torch.no_grad():
            _ = network(obs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure single inference
    start = time.perf_counter()
    for _ in range(config.num_iterations):
        with torch.no_grad():
            _ = network(obs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    inferences_per_sec = config.num_iterations / elapsed

    # Batched throughput
    batched = {}
    for batch_size in config.batch_sizes:
        obs_batch = torch.randn(batch_size, OBSERVATION_DIM)
        try:
            obs_batch = obs_batch.to(device)
        except Exception:
            pass

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = network(obs_batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        iterations = max(100, config.num_iterations // batch_size)
        for _ in range(iterations):
            with torch.no_grad():
                _ = network(obs_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        batched[batch_size] = (iterations * batch_size) / elapsed

    # Games per second (if enabled)
    games_per_sec = None
    if config.include_games:
        games_per_sec = _benchmark_games(network, config.games_per_test)

    return ThroughputMetrics(
        inferences_per_second=inferences_per_sec,
        games_per_second=games_per_sec,
        batched_throughput=batched,
    )


def _benchmark_games(
    network: nn.Module,
    num_games: int,
) -> float:
    """Benchmark full game simulation throughput.

    Args:
        network: Neural network for agent.
        num_games: Number of games to simulate.

    Returns:
        Games per second.
    """
    try:
        from _01_simulator import simulate
        from _02_agents.neural.agent import NeuralAgent
        from _02_agents.random_agent import RandomAgent

        agent = NeuralAgent(network, mode="greedy")
        opponent = RandomAgent(seed=42)

        config = simulate.SimulationConfig(
            seed=0,
            games=num_games,
            agent_a=agent,
            agent_b=opponent,
        )

        start = time.perf_counter()
        for _ in simulate.run(config):
            pass
        elapsed = time.perf_counter() - start

        return num_games / elapsed
    except Exception as e:
        logger.warning(f"Game benchmark failed: {e}")
        return 0.0


def benchmark_model(
    network: nn.Module,
    config: BenchmarkConfig | None = None,
    model_name: str = "unknown",
    model_format: str = "pytorch",
) -> BenchmarkResult:
    """Run comprehensive benchmark on a model.

    Args:
        network: PyTorch network to benchmark.
        config: Benchmark configuration.
        model_name: Name identifier for the model.
        model_format: Format of the model (pytorch, onnx, etc.).

    Returns:
        BenchmarkResult with all metrics.

    Example:
        >>> from _02_agents.neural.network import BeastyBarNetwork
        >>> network = BeastyBarNetwork()
        >>> result = benchmark_model(network, model_name="v1.0")
        >>> print(f"Latency: {result.latency[0].avg_ms:.2f}ms")
    """
    import torch

    if config is None:
        config = BenchmarkConfig()

    network.eval()

    try:
        device = next(network.parameters()).device
        device_str = str(device)
    except StopIteration:
        device_str = "cpu"

    logger.info(f"Benchmarking {model_name} on {device_str}")

    # Latency for each batch size
    latencies = []
    for batch_size in config.batch_sizes:
        latency = _measure_latency(
            inference_fn=network,
            batch_size=batch_size,
            num_iterations=config.num_iterations,
            warmup=config.warmup_iterations,
        )
        latencies.append(latency)
        logger.info(f"  Batch {batch_size}: {latency.avg_ms:.2f}ms (p95: {latency.p95_ms:.2f}ms)")

    # Memory
    memory = None
    if config.include_memory:
        memory = _measure_memory(network)
        logger.info(f"  Model size: {memory.model_size_mb:.1f} MB")

    # Throughput
    throughput = _measure_throughput(network, config)
    logger.info(f"  Throughput: {throughput.inferences_per_second:.0f} inferences/sec")
    if throughput.games_per_second:
        logger.info(f"  Games: {throughput.games_per_second:.1f} games/sec")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchmarkResult(
        model_name=model_name,
        model_format=model_format,
        device=device_str,
        latency=latencies,
        memory=memory,
        throughput=throughput,
    )


def benchmark_onnx(
    onnx_path: str | Path,
    config: BenchmarkConfig | None = None,
    model_name: str = "unknown",
) -> BenchmarkResult:
    """Benchmark an ONNX model.

    Args:
        onnx_path: Path to ONNX model file.
        config: Benchmark configuration.
        model_name: Name identifier for the model.

    Returns:
        BenchmarkResult with metrics.
    """
    from _02_agents.neural.export import ONNXInferenceSession

    if config is None:
        config = BenchmarkConfig()

    onnx_path = Path(onnx_path)
    session = ONNXInferenceSession(onnx_path)

    logger.info(f"Benchmarking ONNX model: {onnx_path}")

    # Latency for each batch size
    latencies = []
    for batch_size in config.batch_sizes:
        obs = np.random.randn(batch_size, OBSERVATION_DIM).astype(np.float32)

        # Warmup
        for _ in range(config.warmup_iterations):
            _ = session(obs)

        # Benchmark
        times = []
        for _ in range(config.num_iterations):
            start = time.perf_counter()
            _ = session(obs)
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)
        latencies.append(
            LatencyMetrics(
                batch_size=batch_size,
                avg_ms=float(np.mean(times)),
                std_ms=float(np.std(times)),
                p50_ms=float(np.percentile(times, 50)),
                p95_ms=float(np.percentile(times, 95)),
                p99_ms=float(np.percentile(times, 99)),
                min_ms=float(np.min(times)),
                max_ms=float(np.max(times)),
            )
        )
        logger.info(f"  Batch {batch_size}: {latencies[-1].avg_ms:.2f}ms")

    # Memory (model file size)
    model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    memory = MemoryMetrics(
        model_size_mb=model_size_mb,
        inference_peak_mb=model_size_mb * 2,  # Estimate
    )

    # Throughput
    obs = np.random.randn(1, OBSERVATION_DIM).astype(np.float32)
    start = time.perf_counter()
    for _ in range(config.num_iterations):
        _ = session(obs)
    elapsed = time.perf_counter() - start
    throughput = ThroughputMetrics(
        inferences_per_second=config.num_iterations / elapsed,
        games_per_second=None,
        batched_throughput={},
    )

    return BenchmarkResult(
        model_name=model_name,
        model_format="onnx",
        device="cpu",  # ONNX Runtime handles device internally
        latency=latencies,
        memory=memory,
        throughput=throughput,
    )


def generate_benchmark_report(
    results: list[BenchmarkResult],
    output_path: str | Path | None = None,
) -> str:
    """Generate a formatted benchmark report.

    Args:
        results: List of benchmark results to include.
        output_path: Optional path to save report.

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK REPORT")
    lines.append("=" * 80)
    lines.append("")

    for result in results:
        lines.append(f"Model: {result.model_name}")
        lines.append(f"Format: {result.model_format}")
        lines.append(f"Device: {result.device}")
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append("")

        # Latency table
        lines.append("Latency (ms):")
        lines.append("-" * 60)
        lines.append(f"{'Batch':<10} {'Avg':<10} {'P50':<10} {'P95':<10} {'P99':<10}")
        lines.append("-" * 60)
        for lat in result.latency:
            lines.append(
                f"{lat.batch_size:<10} {lat.avg_ms:<10.2f} {lat.p50_ms:<10.2f} "
                f"{lat.p95_ms:<10.2f} {lat.p99_ms:<10.2f}"
            )
        lines.append("")

        # Memory
        if result.memory:
            lines.append("Memory:")
            lines.append(f"  Model size: {result.memory.model_size_mb:.1f} MB")
            if result.memory.gpu_allocated_mb:
                lines.append(f"  GPU allocated: {result.memory.gpu_allocated_mb:.1f} MB")
        lines.append("")

        # Throughput
        if result.throughput:
            lines.append("Throughput:")
            lines.append(f"  Single inference: {result.throughput.inferences_per_second:.0f}/sec")
            if result.throughput.games_per_second:
                lines.append(f"  Full games: {result.throughput.games_per_second:.1f}/sec")
            if result.throughput.batched_throughput:
                lines.append("  Batched:")
                for batch, tput in result.throughput.batched_throughput.items():
                    lines.append(f"    Batch {batch}: {tput:.0f}/sec")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info(f"Benchmark report saved to {output_path}")

    return report


def compare_models(
    pytorch_network: nn.Module,
    onnx_path: str | Path | None = None,
    quantized_path: str | Path | None = None,
    config: BenchmarkConfig | None = None,
) -> str:
    """Compare performance across model formats.

    Args:
        pytorch_network: Original PyTorch model.
        onnx_path: Path to ONNX model (optional).
        quantized_path: Path to quantized model (optional).
        config: Benchmark configuration.

    Returns:
        Comparison report string.
    """
    results = []

    # Benchmark PyTorch
    results.append(
        benchmark_model(
            pytorch_network,
            config=config,
            model_name="PyTorch",
            model_format="pytorch",
        )
    )

    # Benchmark ONNX if provided
    if onnx_path and Path(onnx_path).exists():
        results.append(
            benchmark_onnx(
                onnx_path,
                config=config,
                model_name="ONNX",
            )
        )

    # Benchmark quantized if provided
    if quantized_path and Path(quantized_path).exists():
        results.append(
            benchmark_onnx(
                quantized_path,
                config=config,
                model_name="Quantized",
            )
        )

    return generate_benchmark_report(results)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "LatencyMetrics",
    "MemoryMetrics",
    "ThroughputMetrics",
    "benchmark_model",
    "benchmark_onnx",
    "compare_models",
    "generate_benchmark_report",
]
