#!/bin/bash
# Run multiple training experiments sequentially
# Usage: ./scripts/run_experiments.sh
#
# Each experiment runs to completion, then the next starts.
# All checkpoints are saved, so you can resume if interrupted.

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Timestamp for this batch
BATCH_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/$BATCH_ID"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "Beasty Bar Training Batch: $BATCH_ID"
echo "Log directory: $LOG_DIR"
echo "============================================="

# Define experiments (add/modify as needed)
# CPU on NUC: ~43 sec/iter at 128 games. Total ~420 iterations for 5hr.
declare -a EXPERIMENTS=(
    # Format: "name:config_file:extra_args"

    # Experiment 1: Baseline production run (100 iter, ~72 min)
    "baseline:configs/production.yaml:--games-per-iter 128 --iterations 100"

    # Experiment 2: Low learning rate (80 iter, ~57 min)
    "lr_low:configs/production.yaml:--lr 0.0001 --games-per-iter 128 --iterations 80"

    # Experiment 3: High learning rate (80 iter, ~57 min)
    "lr_high:configs/production.yaml:--lr 0.0005 --games-per-iter 128 --iterations 80"

    # Experiment 4: Deep narrow network (80 iter, ~57 min)
    "deep_narrow:configs/default.yaml:--hidden-dim 64 --num-layers 3 --games-per-iter 128 --iterations 80"

    # Experiment 5: Wide shallow network (80 iter, ~57 min)
    "wide_shallow:configs/default.yaml:--hidden-dim 256 --num-layers 1 --games-per-iter 128 --iterations 80"
)

# Run each experiment
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r NAME CONFIG ARGS <<< "${EXPERIMENTS[$i]}"

    echo ""
    echo "============================================="
    echo "EXPERIMENT $((i+1))/${#EXPERIMENTS[@]}: $NAME"
    echo "Config: $CONFIG"
    echo "Args: $ARGS"
    echo "Started: $(date)"
    echo "============================================="

    LOG_FILE="$LOG_DIR/${NAME}.log"

    # Check if checkpoint exists (for resumption)
    CHECKPOINT_DIR="checkpoints/${NAME}"
    RESUME_ARG=""
    if [ -d "$CHECKPOINT_DIR" ]; then
        LATEST=$(ls -t "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "Found checkpoint: $LATEST"
            RESUME_ARG="--resume $LATEST"
        fi
    fi

    # Run training
    ~/.local/bin/uv run python scripts/train.py \
        --config "$CONFIG" \
        --experiment-name "$NAME" \
        --tracker wandb \
        --wandb-project beastybar-experiments \
        $ARGS \
        $RESUME_ARG \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Experiment $NAME completed successfully"
    else
        echo "Experiment $NAME failed with exit code $EXIT_CODE"
        echo "Continuing to next experiment..."
    fi

    echo "Finished: $(date)"
    echo ""
done

echo "============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Batch: $BATCH_ID"
echo "Logs: $LOG_DIR"
echo "============================================="
