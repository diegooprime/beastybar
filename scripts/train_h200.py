#!/usr/bin/env python3
"""H200-optimized training script for Beasty Bar neural network.

This script is designed for RunPod deployment with H200 SMX GPUs.
Key features:
- Tracks best model by HEURISTIC win rate (not MCTS)
- Extended evaluation with more games
- Comprehensive logging to wandb
- Checkpoint selection and cleanup

Usage:
    python scripts/train_h200.py --config configs/h200_optimized.yaml
    python scripts/train_h200.py --iterations 500 --lr 0.0001

Environment:
    WANDB_API_KEY: Required for wandb logging
    CUDA_VISIBLE_DEVICES: GPU selection (default: all available)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from _02_agents.neural.utils import NetworkConfig
from _03_training.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    create_evaluation_report,
    evaluate_agent,
)
from _03_training.ppo import PPOConfig
from _03_training.tracking import WandbTracker, create_tracker
from _03_training.trainer import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class HeuristicBestModelTracker:
    """Track best model by HEURISTIC win rate (not MCTS).

    This addresses the finding that heuristic opponent is the better
    indicator of strategic depth for this game.
    """

    def __init__(self, checkpoint_dir: str, keep_top_k: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.best_heuristic_win_rate = 0.0
        self.best_checkpoint_path: str | None = None
        self.records: list[dict[str, Any]] = []
        self._state_file = self.checkpoint_dir / "heuristic_tracker_state.json"
        self._load_state()

    def update(
        self,
        checkpoint_path: str,
        iteration: int,
        eval_results: list[EvaluationResult],
    ) -> bool:
        """Update tracker with new results. Returns True if new best."""
        # Extract heuristic win rate
        heuristic_wr = 0.0
        random_wr = 0.0
        metrics = {}

        for result in eval_results:
            opponent = result.opponent_name.lower()
            if "heuristic" in opponent:
                heuristic_wr = result.win_rate
            elif "random" in opponent:
                random_wr = result.win_rate
            metrics[f"{result.opponent_name}/win_rate"] = result.win_rate
            metrics[f"{result.opponent_name}/avg_margin"] = result.avg_point_margin

        record = {
            "checkpoint_path": str(checkpoint_path),
            "iteration": iteration,
            "heuristic_win_rate": heuristic_wr,
            "random_win_rate": random_wr,
            "timestamp": time.time(),
            "metrics": metrics,
        }

        is_new_best = heuristic_wr > self.best_heuristic_win_rate

        if is_new_best:
            self.best_heuristic_win_rate = heuristic_wr
            self.best_checkpoint_path = str(checkpoint_path)
            self._copy_best_model(checkpoint_path)
            logger.info(
                f"NEW BEST MODEL at iter {iteration}: "
                f"Heuristic={heuristic_wr:.1%}, Random={random_wr:.1%}"
            )

        # Add record and sort by heuristic win rate
        self.records.append(record)
        self.records.sort(key=lambda r: r["heuristic_win_rate"], reverse=True)
        self.records = self.records[:self.keep_top_k]

        self._save_state()
        return is_new_best

    def _copy_best_model(self, checkpoint_path: str) -> None:
        """Copy checkpoint to best_model.pt."""
        src = Path(checkpoint_path)
        dst = self.checkpoint_dir / "best_model.pt"
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"Saved best model to {dst}")

    def _save_state(self) -> None:
        state = {
            "best_heuristic_win_rate": self.best_heuristic_win_rate,
            "best_checkpoint_path": self.best_checkpoint_path,
            "records": self.records,
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    state = json.load(f)
                self.best_heuristic_win_rate = state.get("best_heuristic_win_rate", 0.0)
                self.best_checkpoint_path = state.get("best_checkpoint_path")
                self.records = state.get("records", [])
                logger.info(f"Loaded tracker state: best={self.best_heuristic_win_rate:.1%}")
            except Exception as e:
                logger.warning(f"Failed to load tracker state: {e}")


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_training_config(config_dict: dict[str, Any]) -> TrainingConfig:
    """Create TrainingConfig from dictionary."""
    # Extract nested configs
    network_dict = config_dict.pop("network_config", {})
    ppo_dict = config_dict.pop("ppo_config", {})

    # Create nested config objects
    network_config = NetworkConfig.from_dict(network_dict)
    ppo_config = PPOConfig(**ppo_dict)

    # Filter to known fields
    import dataclasses
    known_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in known_fields}

    return TrainingConfig(
        network_config=network_config,
        ppo_config=ppo_config,
        **filtered,
    )


def run_extended_evaluation(
    network: torch.nn.Module,
    device: torch.device,
    num_games: int = 100,
) -> list[EvaluationResult]:
    """Run evaluation with more games for lower variance."""
    from _02_agents.neural.agent import NeuralAgent

    agent = NeuralAgent(
        model=network,
        device=device,
        mode="greedy",
    )

    eval_config = EvaluationConfig(
        games_per_opponent=num_games,
        opponents=["random", "heuristic"],
        play_both_sides=True,
    )

    return evaluate_agent(agent, eval_config, device=device)


def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"GPU {i}: {props.name} | "
                f"Memory: {props.total_memory / 1e9:.1f}GB | "
                f"Compute: {props.major}.{props.minor}"
            )
    else:
        logger.warning("No CUDA GPUs available!")


def main():
    parser = argparse.ArgumentParser(description="H200-optimized Beasty Bar training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h200_optimized.yaml",
        help="Path to config file",
    )
    parser.add_argument("--iterations", type=int, help="Override total iterations")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--games-per-iter", type=int, help="Override games per iteration")
    parser.add_argument("--eval-games", type=int, default=100, help="Games per evaluation")
    parser.add_argument("--experiment-name", type=str, help="Override experiment name")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--seed", type=int, help="Override random seed")

    args = parser.parse_args()

    # Print system info
    logger.info("=" * 60)
    logger.info("Beasty Bar H200 Training")
    logger.info("=" * 60)
    print_gpu_info()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config_dict = load_config(str(config_path))
    logger.info(f"Loaded config from {config_path}")

    # Apply command-line overrides
    if args.iterations:
        config_dict["total_iterations"] = args.iterations
    if args.lr:
        config_dict.setdefault("ppo_config", {})["learning_rate"] = args.lr
    if args.games_per_iter:
        config_dict["games_per_iteration"] = args.games_per_iter
    if args.experiment_name:
        config_dict["experiment_name"] = args.experiment_name
    if args.seed:
        config_dict["seed"] = args.seed

    # Create training config
    training_config = create_training_config(config_dict.copy())
    training_config.validate()

    # Log config summary
    logger.info(f"Experiment: {training_config.experiment_name}")
    logger.info(f"Iterations: {training_config.total_iterations}")
    logger.info(f"Games/iter: {training_config.games_per_iteration}")
    logger.info(f"Learning rate: {training_config.ppo_config.learning_rate}")
    logger.info(f"Network: {training_config.network_config.hidden_dim}d x {training_config.network_config.num_layers}L")
    logger.info(f"Entropy coef: {training_config.ppo_config.entropy_coef}")

    # Set up tracking
    if args.no_wandb:
        tracker = create_tracker(backend="console", project="beastybar")
    else:
        try:
            tracker = WandbTracker(
                project="beastybar-experiments",
                run_name=training_config.experiment_name,
                config=training_config.to_dict(),
            )
            logger.info(f"wandb run: {tracker._run.url}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Using console tracker.")
            tracker = create_tracker(backend="console", project="beastybar")

    # Create trainer
    trainer = Trainer(training_config, tracker=tracker)

    # Resume if specified
    if args.resume:
        from _03_training.trainer import load_training_checkpoint
        load_training_checkpoint(args.resume, trainer)
        logger.info(f"Resumed from {args.resume} at iteration {trainer.current_iteration}")

    # Create heuristic-based model tracker
    checkpoint_dir = Path(training_config.checkpoint_dir) / training_config.experiment_name
    model_tracker = HeuristicBestModelTracker(
        checkpoint_dir=str(checkpoint_dir),
        keep_top_k=10,
    )

    # Training loop with enhanced evaluation
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    training_start = time.time()
    iteration = trainer.current_iteration

    try:
        while iteration < training_config.total_iterations:
            iter_start = time.time()

            # Training iteration
            metrics = trainer.train_iteration()

            # Log to tracker
            tracker.log_metrics(metrics, step=iteration)

            iter_time = time.time() - iter_start
            elapsed = time.time() - training_start
            remaining_iters = training_config.total_iterations - iteration - 1
            eta = (elapsed / (iteration + 1)) * remaining_iters

            # Console logging
            if iteration % training_config.log_frequency == 0:
                logger.info(
                    f"Iter {iteration:4d}/{training_config.total_iterations} | "
                    f"Loss: {metrics.get('total_loss', 0):.4f} | "
                    f"LR: {metrics.get('learning_rate', 0):.2e} | "
                    f"Time: {iter_time:.1f}s | "
                    f"ETA: {eta/60:.1f}min"
                )

            # Checkpoint
            if (iteration + 1) % training_config.checkpoint_frequency == 0:
                checkpoint_path = checkpoint_dir / f"iter_{iteration+1:06d}.pt"
                from _03_training.trainer import save_training_checkpoint
                save_training_checkpoint(trainer, str(checkpoint_path))
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Extended evaluation
            if (iteration + 1) % training_config.eval_frequency == 0:
                logger.info(f"Running evaluation ({args.eval_games} games/opponent)...")

                eval_results = run_extended_evaluation(
                    trainer.network,
                    trainer.device,
                    num_games=args.eval_games,
                )

                # Log evaluation results
                for r in eval_results:
                    tracker.log_metrics({
                        f"eval/{r.opponent_name}/win_rate": r.win_rate,
                        f"eval/{r.opponent_name}/avg_margin": r.avg_point_margin,
                        f"eval/{r.opponent_name}/ci_lower": r.confidence_interval_95[0],
                        f"eval/{r.opponent_name}/ci_upper": r.confidence_interval_95[1],
                    }, step=iteration)

                # Print report
                report = create_evaluation_report(eval_results, iteration)
                logger.info("\n" + report)

                # Update best model tracker (by heuristic)
                checkpoint_path = checkpoint_dir / f"iter_{iteration+1:06d}.pt"
                if checkpoint_path.exists():
                    model_tracker.update(str(checkpoint_path), iteration + 1, eval_results)

            iteration += 1
            trainer._iteration = iteration

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        # Save final checkpoint
        final_path = checkpoint_dir / "final.pt"
        from _03_training.trainer import save_training_checkpoint
        save_training_checkpoint(trainer, str(final_path))
        logger.info(f"Saved final checkpoint: {final_path}")

        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = run_extended_evaluation(
            trainer.network,
            trainer.device,
            num_games=200,  # More games for final eval
        )
        report = create_evaluation_report(eval_results, iteration)
        logger.info("\n" + report)

        # Log final metrics
        for r in eval_results:
            tracker.log_metrics({
                f"final/{r.opponent_name}/win_rate": r.win_rate,
                f"final/{r.opponent_name}/avg_margin": r.avg_point_margin,
            }, step=iteration)

        # Summary
        total_time = time.time() - training_start
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Iterations completed: {iteration}")
        logger.info(f"Best heuristic win rate: {model_tracker.best_heuristic_win_rate:.1%}")
        logger.info(f"Best model: {model_tracker.best_checkpoint_path}")
        logger.info("=" * 60)

        tracker.finish()


if __name__ == "__main__":
    main()
