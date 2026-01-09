#!/usr/bin/env python
"""Training CLI for Beasty Bar neural agent.

Usage:
    python scripts/train.py --config config.yaml
    python scripts/train.py --iterations 1000 --games-per-iter 256
    python scripts/train.py --resume checkpoints/checkpoint_500.pt
    python scripts/train.py --lr 0.0001 --device cuda --seed 42

This script provides a command-line interface for training the neural network
agent using PPO self-play. Configuration can be provided via YAML file or
command-line arguments (CLI arguments override YAML values).

Examples:
    # Train with default settings
    python scripts/train.py

    # Train with custom config file
    python scripts/train.py --config configs/production.yaml

    # Override specific parameters
    python scripts/train.py --config configs/default.yaml --lr 0.0001 --iterations 2000

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/experiment_1/iter_000500.pt

    # Use W&B tracking
    python scripts/train.py --tracker wandb --wandb-project beastybar-runs
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

from _02_agents.neural.utils import NetworkConfig
from _03_training.ppo import PPOConfig
from _03_training.tracking import create_tracker
from _03_training.trainer import (
    Trainer,
    TrainingConfig,
    create_trainer_from_checkpoint,
)

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
    if cli_args.iterations is not None:
        config["total_iterations"] = cli_args.iterations

    if cli_args.games_per_iter is not None:
        config["games_per_iteration"] = cli_args.games_per_iter

    if cli_args.lr is not None:
        if "ppo_config" not in config:
            config["ppo_config"] = {}
        config["ppo_config"]["learning_rate"] = cli_args.lr

    if cli_args.checkpoint_dir is not None:
        config["checkpoint_dir"] = cli_args.checkpoint_dir

    if cli_args.experiment_name is not None:
        config["experiment_name"] = cli_args.experiment_name

    if cli_args.device is not None:
        config["device"] = cli_args.device

    if cli_args.seed is not None:
        config["seed"] = cli_args.seed

    if cli_args.eval_frequency is not None:
        config["eval_frequency"] = cli_args.eval_frequency

    # Network config overrides
    if cli_args.hidden_dim is not None:
        if "network_config" not in config:
            config["network_config"] = {}
        config["network_config"]["hidden_dim"] = cli_args.hidden_dim

    if cli_args.num_layers is not None:
        if "network_config" not in config:
            config["network_config"] = {}
        config["network_config"]["num_layers"] = cli_args.num_layers

    return config


def create_training_config(config_dict: dict[str, Any]) -> TrainingConfig:
    """Create TrainingConfig from dictionary.

    Handles nested configs for network and PPO.

    Args:
        config_dict: Configuration dictionary.

    Returns:
        TrainingConfig instance.
    """
    # Handle nested network config
    if "network_config" in config_dict and isinstance(config_dict["network_config"], dict):
        config_dict["network_config"] = NetworkConfig.from_dict(config_dict["network_config"])

    # Handle nested PPO config
    if "ppo_config" in config_dict and isinstance(config_dict["ppo_config"], dict):
        config_dict["ppo_config"] = PPOConfig(**config_dict["ppo_config"])

    return TrainingConfig.from_dict(config_dict)


def main() -> int:
    """Main training CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Train Beasty Bar neural agent with PPO self-play",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint file (.pt)",
    )

    # Training schedule
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Total training iterations (default: 1000)",
    )

    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=None,
        help="Number of self-play games per iteration (default: 256)",
    )

    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=None,
        help="Evaluate every N iterations (default: 10)",
    )

    # Learning rate
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for PPO optimizer (default: 0.0003)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints (default: checkpoints)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (default: beastybar_neural)",
    )

    # Device and reproducibility
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default=None,
        help="Device for training (default: auto)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42)",
    )

    # Experiment tracking
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["console", "wandb"],
        default="console",
        help="Experiment tracking backend (default: console)",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="beastybar",
        help="W&B project name (default: beastybar)",
    )

    # Network architecture
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for network (default: 128)",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer layers (default: 1)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load config from YAML if provided
        yaml_config: dict[str, Any] = {}
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            yaml_config = load_config_from_yaml(args.config)

        # Merge with CLI arguments
        config_dict = merge_configs(yaml_config, args)

        # Create experiment tracker
        tracker_backend = args.tracker
        tracker_project = args.wandb_project if args.tracker == "wandb" else "beastybar"
        experiment_name = config_dict.get("experiment_name", "beastybar_neural")

        tracker = create_tracker(
            backend=tracker_backend,
            project=tracker_project,
            run_name=experiment_name,
            config=config_dict,
        )

        # Create or resume trainer
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer = create_trainer_from_checkpoint(
                path=args.resume,
                tracker=tracker,
                config_overrides=config_dict,
            )
        else:
            logger.info("Creating new training run")
            training_config = create_training_config(config_dict)
            trainer = Trainer(training_config, tracker=tracker)

        # Log configuration summary
        logger.info("=" * 80)
        logger.info("Training Configuration:")
        logger.info(f"  Iterations: {trainer.config.total_iterations}")
        logger.info(f"  Games per iteration: {trainer.config.games_per_iteration}")
        logger.info(f"  Learning rate: {trainer.config.ppo_config.learning_rate}")
        logger.info(f"  Device: {trainer.device}")
        logger.info(f"  Seed: {trainer.config.seed}")
        logger.info(f"  Experiment: {trainer.config.experiment_name}")
        logger.info(f"  Checkpoint dir: {trainer.config.checkpoint_dir}")
        logger.info(f"  Tracker: {tracker_backend}")
        logger.info("=" * 80)

        # Run training
        logger.info("Starting training...")
        trainer.train()

        logger.info("Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
