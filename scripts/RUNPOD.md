# RunPod Training

Quick setup for training on RunPod H200 SXM GPUs.

## Setup

```bash
# Set connection
export RUNPOD_IP="<IP>"
export RUNPOD_PORT="<PORT>"
export SSH_KEY="~/.ssh/id_ed25519"

# Test connection
ssh -o StrictHostKeyChecking=no -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP "nvidia-smi"

# Install rsync (not pre-installed)
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP "apt-get update && apt-get install -y rsync"

# Sync code
rsync -avz --progress \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='checkpoints' \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY" \
  . root@$RUNPOD_IP:/workspace/beastybar/

# Install dependencies + Cython
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP << 'EOF'
cd /workspace/beastybar
pip install uv && uv sync
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install cython setuptools
python _01_simulator/_cython/setup.py build_ext --inplace
EOF
```

## Training

The hardcoded defaults in `TrainingConfig` are optimized for H200 SXM:
- 131K games/iteration, 32K minibatch
- Opponent pool ON (80% Cython, 20% Python fallback for heuristic/random)
- 16 async workers, 32 prefetch batches
- 1000 iterations

```bash
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP << 'EOF'
cd /workspace/beastybar && source .venv/bin/activate

# Set wandb API key
export WANDB_API_KEY="<your-api-key>"

# Start training with wandb tracking
nohup python scripts/train.py \
  --tracker wandb \
  --wandb-project beastybar \
  --experiment-name <run-name> \
  > training.log 2>&1 &
EOF
```

## Monitoring

```bash
# GPU usage
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP 'nvidia-smi'

# Training progress
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP 'tail -20 /workspace/beastybar/training.log'

# Check if running
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP 'ps aux | grep train'
```

## Download Results

Training produces two checkpoint types:

| File | Size | Purpose |
|------|------|---------|
| `model_inference.pt` | ~5 MB | **Download this** - for deployment/HuggingFace |
| `final.pt` | ~500 MB | Full checkpoint - only if you need to resume training |

```bash
# Download inference model only (recommended, ~5 MB)
scp -P $RUNPOD_PORT -i $SSH_KEY \
  root@$RUNPOD_IP:/workspace/beastybar/checkpoints/*/model_inference.pt ./

# Download full checkpoint (only if resuming training, ~500 MB)
scp -P $RUNPOD_PORT -i $SSH_KEY \
  root@$RUNPOD_IP:/workspace/beastybar/checkpoints/*/final.pt ./checkpoints/
```

## Storage on RunPod

With 20GB volume, you only need to keep:
- Latest `final.pt` (~500 MB) - for resuming if interrupted
- `model_inference.pt` (~5 MB) - the actual model

Delete intermediate checkpoints to save space:
```bash
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP \
  'find /workspace/beastybar/checkpoints -name "iter_*.pt" -delete'
```

## SSH Types

RunPod has two SSH methods. **Use Direct TCP for file transfer:**

| Method | Command | SCP works |
|--------|---------|-----------|
| Proxy | `ssh <pod>@ssh.runpod.io` | No |
| Direct TCP | `ssh root@<IP> -p <PORT>` | Yes |

## Gotchas

- **rsync not found:** `apt-get install -y rsync`
- **torch not found:** `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **Cython wrong arch:** Rebuild on Linux: `python _01_simulator/_cython/setup.py build_ext --inplace`

## Cython + Opponent Pool

With default opponent weights (60% current, 20% checkpoint, 10% random, 10% heuristic):
- **80% of iterations use Cython** (neural network opponents)
- **20% fall back to Python** (random/heuristic agents need State conversion)

This is the recommended setup - opponent diversity prevents self-play collapse.
To go 100% Cython (faster but may collapse), use `--config configs/runpod_h200.yaml`.
