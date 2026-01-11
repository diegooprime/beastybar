# RunPod Training

Quick setup for training on RunPod H100/H200 GPUs.

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

```bash
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP << 'EOF'
cd /workspace/beastybar && source .venv/bin/activate
nohup python scripts/train.py \
  --config configs/runpod_h200.yaml \
  --iterations 600 \
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

```bash
scp -P $RUNPOD_PORT -i $SSH_KEY \
  root@$RUNPOD_IP:/workspace/beastybar/checkpoints/*.pt ./checkpoints/
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
- **Slow training:** Use `configs/runpod_h200.yaml` with `use_opponent_pool: false` for Cython path

## Expected Performance

| Mode | Time/Iteration |
|------|----------------|
| Async + Cython | ~30-40s |
| With opponent pool | ~225s (avoid) |
