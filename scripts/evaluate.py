#!/usr/bin/env python
"""Evaluation CLI for Beasty Bar neural agent.

Usage:
    python scripts/evaluate.py --model checkpoint.pt --opponent mcts-500 --games 100
    python scripts/evaluate.py --model checkpoint.pt --opponents random,heuristic,mcts-500
    python scripts/evaluate.py --model checkpoint.pt --both-sides --output results.json

This script evaluates a trained neural network agent against baseline opponents
and reports detailed statistics including win rates, confidence intervals, and
ELO estimates.

Examples:
    # Evaluate against single opponent
    python scripts/evaluate.py --model checkpoints/iter_001000.pt --opponent mcts-500

    # Evaluate against multiple opponents
    python scripts/evaluate.py --model checkpoints/final.pt \\
        --opponents random,heuristic,mcts-500,mcts-1000 --games 200

    # Save results to JSON file
    python scripts/evaluate.py --model checkpoints/best.pt \\
        --opponents random,heuristic,mcts-500 \\
        --output evaluation_results.json

    # Play as both sides for fairness
    python scripts/evaluate.py --model checkpoints/iter_001000.pt \\
        --opponent mcts-500 --games 100 --both-sides
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from _02_agents.neural.agent import load_neural_agent
from _03_training.evaluation import (
    EvaluationConfig,
    create_evaluation_report,
    estimate_elo,
    evaluate_agent,
)

logger = logging.getLogger(__name__)


def parse_opponents(opponents_str: str) -> list[str]:
    """Parse comma-separated list of opponent names.

    Args:
        opponents_str: Comma-separated opponent names.

    Returns:
        List of opponent names.

    Examples:
        >>> parse_opponents("random,heuristic")
        ['random', 'heuristic']
        >>> parse_opponents("mcts-500")
        ['mcts-500']
    """
    return [opp.strip() for opp in opponents_str.split(",") if opp.strip()]


def evaluation_results_to_dict(results: list) -> list[dict[str, Any]]:
    """Convert evaluation results to JSON-serializable format.

    Args:
        results: List of EvaluationResult objects.

    Returns:
        List of dictionaries with serializable data.
    """
    return [asdict(result) for result in results]


def main() -> int:
    """Main evaluation CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Beasty Bar neural agent against baseline opponents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )

    # Opponent selection (mutually exclusive)
    opponent_group = parser.add_mutually_exclusive_group(required=True)
    opponent_group.add_argument(
        "--opponent",
        type=str,
        help="Single opponent to evaluate against (e.g., 'mcts-500')",
    )
    opponent_group.add_argument(
        "--opponents",
        type=str,
        help="Comma-separated list of opponents (e.g., 'random,heuristic,mcts-500')",
    )

    # Evaluation settings
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play per opponent (default: 100)",
    )

    parser.add_argument(
        "--both-sides",
        action="store_true",
        help="Play as both player 0 and player 1 for fairness",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of fixed seeds for reproducibility (optional)",
    )

    # Model settings
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device for model inference (default: auto)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["greedy", "stochastic", "temperature"],
        default="greedy",
        help="Inference mode for neural agent (default: greedy)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for 'temperature' mode (default: 1.0)",
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file (optional)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game-by-game results",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Parse opponents
        opponents = [args.opponent] if args.opponent else parse_opponents(args.opponents)

        logger.info(f"Evaluating against opponents: {', '.join(opponents)}")

        # Parse seeds if provided
        seeds = None
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
            logger.info(f"Using {len(seeds)} fixed seeds")

        # Load neural agent
        logger.info(f"Loading model from {args.model}")
        agent = load_neural_agent(
            checkpoint_path=args.model,
            mode=args.mode,
            temperature=args.temperature,
            device=args.device,
        )
        logger.info(f"Model loaded: {agent.name}, device={agent.device}")

        # Create evaluation config
        eval_config = EvaluationConfig(
            games_per_opponent=args.games,
            opponents=opponents,
            play_both_sides=args.both_sides,
            seeds=seeds,
        )

        # Run evaluation
        logger.info("Starting evaluation...")
        logger.info(f"  Games per opponent: {args.games}")
        logger.info(f"  Play both sides: {args.both_sides}")
        logger.info("=" * 80)

        results = evaluate_agent(agent, eval_config, device=agent.device)

        # Print results
        report = create_evaluation_report(results)
        print("\n" + report + "\n")

        # Estimate ELO rating
        try:
            elo = estimate_elo(results)
            print(f"Estimated ELO rating: {elo:.0f}\n")
        except Exception as e:
            logger.warning(f"Could not estimate ELO: {e}")

        # Save to JSON if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "model_checkpoint": str(args.model),
                "evaluation_config": {
                    "games_per_opponent": args.games,
                    "opponents": opponents,
                    "play_both_sides": args.both_sides,
                    "inference_mode": args.mode,
                    "temperature": args.temperature,
                },
                "results": evaluation_results_to_dict(results),
                "elo_estimate": estimate_elo(results),
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Results saved to {output_path}")

        # Return success/failure based on performance
        # Consider it a success if agent beats random opponent
        random_results = [r for r in results if r.opponent_name.lower() == "random"]
        if random_results and random_results[0].win_rate >= 0.95:
            logger.info("Evaluation passed: Agent beats random >95%")
            return 0
        elif random_results:
            logger.warning(
                f"Evaluation concern: Agent only beats random {random_results[0].win_rate * 100:.1f}%"
            )
            return 0  # Still return success, just a warning
        else:
            return 0  # No random opponent evaluated, still success

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1

    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
