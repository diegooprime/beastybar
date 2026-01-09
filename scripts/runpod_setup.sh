#!/bin/bash
# RunPod H200 Setup and Training Script
#
# Usage:
#   1. Upload this repo to RunPod or clone from git
#   2. Run: bash scripts/runpod_setup.sh
#
# Expected environment: RunPod with 2x H200 SMX, PyTorch pre-installed

set -e

echo "=============================================="
echo "Beasty Bar H200 Training Setup"
echo "=============================================="

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# Navigate to project directory
cd "$(dirname "$0")/.." || exit 1
PROJECT_DIR=$(pwd)
echo "Project directory: $PROJECT_DIR"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q wandb pyyaml

# Install project in editable mode (ensures all internal imports work)
pip install -q -e .

# Check if torch is available (should be pre-installed on RunPod)
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verify CUDA
if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: CUDA not available!"
    exit 1
fi

# Set up wandb (if API key provided)
if [ -n "$WANDB_API_KEY" ]; then
    echo ""
    echo "Configuring wandb..."
    wandb login --relogin "$WANDB_API_KEY"
else
    echo ""
    echo "WARNING: WANDB_API_KEY not set. Training will use console logging only."
    echo "Set it with: export WANDB_API_KEY=your_key"
fi

# Create output directories
mkdir -p checkpoints
mkdir -p logs

# Print training configuration
echo ""
echo "=============================================="
echo "Training Configuration"
echo "=============================================="
echo "Config file: configs/h200_optimized.yaml"
cat configs/h200_optimized.yaml | grep -E "^[a-z_]+:|^  [a-z_]+:" | head -20
echo "..."
echo ""

# Estimate training time
echo "Estimated throughput:"
echo "  - Games per iteration: 512"
echo "  - Est. iteration time: 7-12s (GPU-accelerated)"
echo "  - Target iterations: 500"
echo "  - Estimated total time: ~60-90 minutes"
echo ""

# Start training
echo "=============================================="
echo "Starting Training"
echo "=============================================="
echo "Log file: logs/training_$(date +%Y%m%d_%H%M%S).log"
echo ""

# Run training with logging
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

python scripts/train_h200.py \
    --config configs/h200_optimized.yaml \
    --eval-games 100 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "Checkpoints saved to: checkpoints/"
echo "Best model: checkpoints/h200_optimized_v1/best_model.pt"
echo "Logs: $LOG_FILE"
