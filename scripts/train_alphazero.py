#!/usr/bin/env python3
"""AlphaZero Training CLI for Beasty Bar neural agent.

Usage:
    python scripts/train_alphazero.py --config configs/alphazero_v2.yaml
    python scripts/train_alphazero.py --config configs/alphazero_v2.yaml --wandb
    python scripts/train_alphazero.py --config configs/alphazero_parallel.yaml --checkpoint checkpoints/iter_100.pt

This script provides a command-line interface for training using AlphaZero-style
MCTS self-play with policy targets from visit counts.

Key differences from PPO:
- Policy targets = MCTS visit distribution (not policy gradient)
- Value targets = game outcomes (not GAE estimates)
- No entropy bonus (MCTS provides exploration)
- More stable for two-player zero-sum games
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from _03_training.alphazero_trainer import (
    AlphaZeroConfig,
    AlphaZeroTrainer,
    create_trainer_from_checkpoint,
)
from _03_training.tracking import create_tracker

logger = logging.getLogger(__name__)


def load_config_from_yaml(path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config or {}


def merge_configs(
    yaml_config: dict[str, Any],
    cli_args: argparse.Namespace,
) -> dict[str, Any]:
    """Merge YAML config with CLI arguments. CLI takes precedence."""
    config = yaml_config.copy()

    if cli_args.experiment_name is not None:
        config["experiment_name"] = cli_args.experiment_name

    if cli_args.seed is not None:
        config["seed"] = cli_args.seed

    if cli_args.iterations is not None:
        config["total_iterations"] = cli_args.iterations

    if cli_args.device is not None:
        config["device"] = cli_args.device

    return config


def main() -> int:
    """Main AlphaZero training CLI entry point."""
    import torch

    # H100 GPU Optimization: Enable TF32 for ~7x speedup on float32 matmuls
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description="Train Beasty Bar neural agent with AlphaZero-style MCTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint file (.pt)",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases tracking",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="beastybar",
        help="W&B project name (default: beastybar)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment name from config",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override total iterations",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default=None,
        help="Override device",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        yaml_config = load_config_from_yaml(args.config)
        config_dict = merge_configs(yaml_config, args)

        # Create tracker
        tracker_backend = "wandb" if args.wandb else "console"
        experiment_name = config_dict.get("experiment_name", "alphazero_v2")

        tracker = create_tracker(
            backend=tracker_backend,
            project=args.wandb_project if args.wandb else "beastybar",
            run_name=experiment_name,
            config=config_dict,
        )

        # Create or resume trainer
        if args.checkpoint:
            logger.info(f"Resuming from checkpoint: {args.checkpoint}")
            trainer = create_trainer_from_checkpoint(
                path=args.checkpoint,
                tracker=tracker,
                config_overrides=config_dict,
            )
        else:
            logger.info("Creating new AlphaZero training run")
            training_config = AlphaZeroConfig.from_dict(config_dict)
            trainer = AlphaZeroTrainer(training_config, tracker=tracker)

        # Log configuration
        logger.info("=" * 80)
        logger.info("AlphaZero V2 Training Configuration:")
        logger.info(f"  Network version: {trainer.config.network_version}")
        logger.info(f"  Total iterations: {trainer.config.total_iterations}")
        logger.info(f"  Games per iteration: {trainer.config.games_per_iteration}")
        logger.info(f"  Parallel games: {trainer.config.parallel_games}")
        logger.info(f"  MCTS simulations: {trainer.config.num_simulations}")
        logger.info(f"  Learning rate: {trainer.config.learning_rate}")
        logger.info(f"  Batch size: {trainer.config.batch_size}")
        logger.info(f"  Device: {trainer.device}")
        logger.info(f"  Tablebase: {trainer.config.tablebase_path}")
        logger.info(f"  Experiment: {trainer.config.experiment_name}")
        logger.info("=" * 80)

        # Run training
        logger.info("Starting AlphaZero training...")
        trainer.train()

        logger.info("Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
