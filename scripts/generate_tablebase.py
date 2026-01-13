#!/usr/bin/env python3
"""Generate endgame tablebase on high-core-count machine.

This script generates a complete endgame tablebase using retrograde
analysis and parallel processing. Designed for AWS c7i.48xlarge
(192 vCPUs) but works on any machine.

Usage:
    # Local development (small tablebase)
    python scripts/generate_tablebase.py --max-cards 4 --output tb_4cards.tb

    # Local validation (6 cards)
    python scripts/generate_tablebase.py --max-cards 6 --output tb_6cards.tb --validate

    # AWS production (10 cards)
    python scripts/generate_tablebase.py \
        --max-cards 10 \
        --workers 192 \
        --output tablebase_10cards.tb \
        --checkpoint-interval 300

Examples:
    # Quick test with 3 cards
    python scripts/generate_tablebase.py --max-cards 3 --output test.tb

    # Benchmark parallel scaling
    python scripts/generate_tablebase.py --max-cards 5 --workers 1 --output tb1.tb
    python scripts/generate_tablebase.py --max-cards 5 --workers 4 --output tb4.tb
    python scripts/generate_tablebase.py --max-cards 5 --workers 8 --output tb8.tb
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from _02_agents.tablebase.enumerate import PositionEnumerator, count_positions_estimate
from _02_agents.tablebase.parallel import ParallelConfig, ParallelTablebaseGenerator
from _02_agents.tablebase.retrograde import RetrogradeConfig, RetrogradeTablebase
from _02_agents.tablebase.storage import MMapTablebase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_tablebase")


def estimate_positions(max_cards: int) -> None:
    """Print position count estimate."""
    logger.info("Estimating position counts...")

    enumerator = PositionEnumerator()

    # Quick estimate
    estimate = count_positions_estimate(max_cards)
    logger.info("Quick estimate for %d cards: ~%d positions", max_cards, estimate)

    # Actual count for small values
    if max_cards <= 5:
        logger.info("Counting exact positions (may take a moment)...")
        count = enumerator.count_positions(max_cards)
        logger.info("Exact count for %d cards: %d positions", max_cards, count)


def generate_retrograde(
    max_cards: int,
    output: Path,
    validate: bool = False,
) -> None:
    """Generate tablebase using single-threaded retrograde analysis."""
    logger.info("Using retrograde analysis (single-threaded)")

    config = RetrogradeConfig(
        max_cards=max_cards,
        log_interval=1,
    )

    tablebase = RetrogradeTablebase(config)
    stats = tablebase.generate()

    logger.info(
        "Generation complete: %d/%d positions in %.1fs",
        stats.solved_positions,
        stats.total_positions,
        stats.elapsed_seconds,
    )

    # Validate if requested
    if validate:
        logger.info("Validating against forward search...")
        validation = tablebase.validate_against_forward(sample_size=min(100, stats.total_positions))
        logger.info(
            "Validation: %d matches, %d mismatches, %d errors",
            validation["matches"],
            validation["mismatches"],
            validation["errors"],
        )

    # Save to disk
    logger.info("Saving tablebase to %s", output)
    num_positions = stats.total_positions

    with MMapTablebase(output, num_positions) as storage:
        tablebase._transfer_to_storage(storage)

    file_size = output.stat().st_size / (1024 * 1024)
    logger.info("Saved: %.2f MB", file_size)


def generate_parallel(
    max_cards: int,
    output: Path,
    workers: int,
    checkpoint_path: Path | None,
    checkpoint_interval: int,
    validate: bool = False,
) -> None:
    """Generate tablebase using parallel processing."""
    logger.info("Using parallel generation with %d workers", workers)

    config = ParallelConfig(
        max_cards=max_cards,
        num_workers=workers,
        checkpoint_interval=checkpoint_interval,
        log_interval=5,
    )

    generator = ParallelTablebaseGenerator(config)

    stats = generator.generate(
        output_path=output,
        checkpoint_path=checkpoint_path,
    )

    logger.info(
        "Generation complete: %d/%d positions in %.1fs (%.1f pos/s)",
        stats.solved_positions,
        stats.total_positions,
        stats.elapsed_seconds,
        stats.positions_per_second,
    )

    # Print final stats
    if output.exists():
        file_size = output.stat().st_size / (1024 * 1024)
        logger.info("Output file: %.2f MB", file_size)

        # Load and check stats
        with MMapTablebase(output, stats.total_positions, readonly=True) as tb:
            tb_stats = tb.get_stats()
            logger.info(
                "Tablebase stats: WIN=%d, LOSS=%d, DRAW=%d, UNKNOWN=%d",
                tb_stats["win"],
                tb_stats["loss"],
                tb_stats["draw"],
                tb_stats["unknown"],
            )

    # Validation
    if validate:
        logger.info("Validating against forward search...")
        # Use retrograde for validation
        retro = RetrogradeTablebase(RetrogradeConfig(max_cards=max_cards))
        retro.generate()
        validation = retro.validate_against_forward(sample_size=100)
        logger.info(
            "Validation: %d matches, %d mismatches, %d errors",
            validation["matches"],
            validation["mismatches"],
            validation["errors"],
        )


def benchmark(max_cards: int, max_workers: int) -> None:
    """Benchmark parallel scaling."""
    logger.info("Benchmarking parallel scaling for %d cards", max_cards)

    import tempfile

    results = []
    worker_counts = [1, 2, 4, 8]
    if max_workers > 8:
        worker_counts.extend([16, 32, 64])
    if max_workers > 64:
        worker_counts.extend([128, 192])

    worker_counts = [w for w in worker_counts if w <= max_workers]

    for workers in worker_counts:
        with tempfile.NamedTemporaryFile(suffix=".tb", delete=True) as f:
            output = Path(f.name)

            config = ParallelConfig(
                max_cards=max_cards,
                num_workers=workers,
            )

            generator = ParallelTablebaseGenerator(config)
            stats = generator.generate(output_path=output)

            results.append({
                "workers": workers,
                "elapsed": stats.elapsed_seconds,
                "positions_per_second": stats.positions_per_second,
            })

            logger.info(
                "Workers=%d: %.1fs (%.1f pos/s)",
                workers,
                stats.elapsed_seconds,
                stats.positions_per_second,
            )

    # Print summary
    print("\nBenchmark Results:")
    print("-" * 50)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Pos/s':<12} {'Speedup':<10}")
    print("-" * 50)

    baseline = results[0]["elapsed"]
    for r in results:
        speedup = baseline / r["elapsed"] if r["elapsed"] > 0 else 0
        print(f"{r['workers']:<10} {r['elapsed']:<12.1f} {r['positions_per_second']:<12.1f} {speedup:<10.2f}x")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate endgame tablebase for Beasty Bar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--max-cards",
        type=int,
        default=6,
        help="Maximum cards in play (default: 6)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="Output file path for tablebase",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=300,
        help="Seconds between checkpoints (default: 300)",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file path",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate results against forward search",
    )

    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Only estimate position counts",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run parallel scaling benchmark",
    )

    parser.add_argument(
        "--single-threaded",
        action="store_true",
        help="Use single-threaded retrograde (for debugging)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle estimate mode
    if args.estimate:
        estimate_positions(args.max_cards)
        return

    # Handle benchmark mode
    if args.benchmark:
        max_workers = args.workers or os.cpu_count() or 8
        benchmark(args.max_cards, max_workers)
        return

    # Normal generation - require output path
    if args.output is None:
        args.output = Path(f"tablebase_{args.max_cards}cards.tb")
        logger.info("Using default output path: %s", args.output)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Choose generation method
    if args.single_threaded:
        generate_retrograde(
            args.max_cards,
            args.output,
            args.validate,
        )
    else:
        workers = args.workers or os.cpu_count() or 1
        generate_parallel(
            args.max_cards,
            args.output,
            workers,
            args.checkpoint,
            args.checkpoint_interval,
            args.validate,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
