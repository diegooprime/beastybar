#!/usr/bin/env python
"""MCTS Training CLI for Beasty Bar neural agent.

Usage:
    python scripts/train_mcts.py --config configs/h100_mcts.yaml
    python scripts/train_mcts.py --config configs/h100_mcts.yaml --wandb
    python scripts/train_mcts.py --config configs/h100_mcts.yaml --checkpoint checkpoints/iter_100.pt
    python scripts/train_mcts.py --config configs/h100_mcts.yaml --seed 123 --experiment-name mcts_exp_1

This script provides a command-line interface for training the neural network
agent using AlphaZero-style MCTS self-play. Configuration is loaded from YAML
files, with optional command-line overrides.

The MCTS training approach differs from PPO:
- Uses MCTS search to improve the policy during self-play
- Trains network to match MCTS visit distributions (policy targets)
- Uses game outcomes directly (no GAE or advantage estimation)
- More stable and sample-efficient for two-player zero-sum games

Examples:
    # Train with default config
    python scripts/train_mcts.py --config configs/h100_mcts.yaml

    # Use W&B tracking
    python scripts/train_mcts.py --config configs/h100_mcts.yaml --wandb

    # Resume from checkpoint
    python scripts/train_mcts.py --config configs/h100_mcts.yaml --checkpoint checkpoints/h100_mcts_v1/iter_000100.pt

    # Override experiment name and seed
    python scripts/train_mcts.py --config configs/h100_mcts.yaml --experiment-name test_run --seed 999
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

from _03_training.mcts_trainer import MCTSTrainerConfig, create_trainer_from_checkpoint
from _03_training.tracking import create_tracker

logger = logging.getLogger(__name__)


def load_config_from_yaml(path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        Dictionary of configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
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
    """Merge YAML config with CLI arguments.

    CLI arguments take precedence over YAML values.

    Args:
        yaml_config: Configuration loaded from YAML file.
        cli_args: Parsed command-line arguments.

    Returns:
        Merged configuration dictionary.
    """
    # Start with YAML config
    config = yaml_config.copy()

    # Override with CLI args (only if explicitly set)
    if cli_args.experiment_name is not None:
        config["experiment_name"] = cli_args.experiment_name

    if cli_args.seed is not None:
        config["seed"] = cli_args.seed

    # Device override
    if hasattr(cli_args, "device") and cli_args.device is not None:
        config["device"] = cli_args.device

    return config


def create_mcts_config(config_dict: dict[str, Any]) -> MCTSTrainerConfig:
    """Create MCTSTrainerConfig from dictionary.

    Handles nested configs for network and MCTS parameters.

    Args:
        config_dict: Configuration dictionary.

    Returns:
        MCTSTrainerConfig instance.
    """
    return MCTSTrainerConfig.from_dict(config_dict)


def main() -> int:
    """Main MCTS training CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Train Beasty Bar neural agent with AlphaZero-style MCTS self-play",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Configuration file (required)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (required)",
    )

    # Resume training
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint file (.pt)",
    )

    # Experiment tracking
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases tracking (default: console)",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="beastybar",
        help="W&B project name (default: beastybar)",
    )

    # Config overrides
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
        help="Override random seed from config",
    )

    # Logging
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
        # Load configuration from YAML
        logger.info(f"Loading configuration from {args.config}")
        yaml_config = load_config_from_yaml(args.config)

        # Merge with CLI arguments
        config_dict = merge_configs(yaml_config, args)

        # Determine tracker backend
        tracker_backend = "wandb" if args.wandb else "console"
        tracker_project = args.wandb_project if args.wandb else "beastybar"
        experiment_name = config_dict.get("experiment_name", "beastybar_mcts")

        # Create experiment tracker
        tracker = create_tracker(
            backend=tracker_backend,
            project=tracker_project,
            run_name=experiment_name,
            config=config_dict,
        )

        # Create or resume trainer
        if args.checkpoint:
            logger.info(f"Resuming training from checkpoint: {args.checkpoint}")
            from _03_training.mcts_trainer import MCTSTrainer

            trainer = create_trainer_from_checkpoint(
                path=args.checkpoint,
                tracker=tracker,
                config_overrides=config_dict,
            )
        else:
            logger.info("Creating new MCTS training run")
            from _03_training.mcts_trainer import MCTSTrainer

            training_config = create_mcts_config(config_dict)
            trainer = MCTSTrainer(training_config, tracker=tracker)

        # Log configuration summary
        logger.info("=" * 80)
        logger.info("MCTS Training Configuration:")
        logger.info(f"  Total iterations: {trainer.config.total_iterations}")
        logger.info(f"  Games per iteration: {trainer.config.games_per_iteration}")
        logger.info(f"  MCTS simulations: {trainer.config.mcts_config.num_simulations}")
        logger.info(f"  MCTS temperature: {trainer.config.mcts_config.temperature}")
        logger.info(f"  MCTS c_puct: {trainer.config.mcts_config.c_puct}")
        logger.info(f"  Learning rate: {trainer.config.learning_rate}")
        logger.info(f"  Batch size: {trainer.config.batch_size}")
        logger.info(f"  Epochs per iteration: {trainer.config.epochs_per_iteration}")
        logger.info(f"  Device: {trainer.device}")
        logger.info(f"  Seed: {trainer.config.seed}")
        logger.info(f"  Experiment: {trainer.config.experiment_name}")
        logger.info(f"  Checkpoint dir: {trainer.config.checkpoint_dir}")
        logger.info(f"  Tracker: {tracker_backend}")
        logger.info("=" * 80)

        # Run training
        logger.info("Starting MCTS training...")
        trainer.train()

        logger.info("Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130  # Standard exit code for SIGINT

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
