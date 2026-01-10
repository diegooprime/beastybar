# RunPod Training Runbook for Beasty Bar

**Last Updated:** 2026-01-10
**Tested On:** RunPod H200 SXM (Ubuntu 22.04)

---

## How to Start Training (For Fresh Claude Instance)

Copy this prompt to a new Claude Code session:

```
Start training on RunPod. Here's the SSH connection:

ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

Follow the runbook at scripts/RUNPOD_TRAINING_RUNBOOK.md

Resume from the latest checkpoint at checkpoints/ppo_h200_maxperf/final.pt
```

Replace `<IP>` and `<PORT>` with values from the RunPod screenshot.

---

## Quick Start (All-in-One)

### Step 1: Set Connection Variables

```bash
# Get these from RunPod console screenshot
export RUNPOD_IP="<IP>"
export RUNPOD_PORT="<PORT>"
export SSH_KEY="~/.ssh/id_ed25519"
```

### Step 2: Run Setup Script

```bash
# Test connection
ssh -o StrictHostKeyChecking=no -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP "nvidia-smi --query-gpu=name --format=csv,noheader"

# Install rsync (not pre-installed!)
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP "apt-get update && apt-get install -y rsync"

# Sync codebase
rsync -avz --progress \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  --exclude='wandb' --exclude='checkpoints/ppo_h200_v*' \
  -e "ssh -p $RUNPOD_PORT -i $SSH_KEY" \
  /Users/p/Desktop/v/experiments/beastybar/ \
  root@$RUNPOD_IP:/workspace/beastybar/

# Upload checkpoint
scp -P $RUNPOD_PORT -i $SSH_KEY \
  /Users/p/Desktop/v/experiments/beastybar/checkpoints/ppo_h200_maxperf/final.pt \
  root@$RUNPOD_IP:/workspace/beastybar/checkpoints/ppo_h200_maxperf/

# Install dependencies + build Cython
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP << 'EOF'
cd /workspace/beastybar
pip install uv && uv sync
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install cython setuptools
python _01_simulator/_cython/setup.py build_ext --inplace
echo "Setup complete!"
EOF
```

### Step 3: Start Training

```bash
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP << 'EOF'
cd /workspace/beastybar
source .venv/bin/activate

nohup python scripts/train.py \
  --config configs/runpod_h200.yaml \
  --resume checkpoints/ppo_h200_maxperf/final.pt \
  --iterations 600 \
  --experiment-name h200_continued \
  --checkpoint-dir checkpoints/h200_continued \
  > training.log 2>&1 &

echo "Training started with PID: $!"
EOF
```

### Step 4: Monitor Training

```bash
ssh -p $RUNPOD_PORT -i $SSH_KEY root@$RUNPOD_IP \
  'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv; tail -5 /workspace/beastybar/training.log | grep -E "Step|Iter"'
```

---

## Recommended Config

Use `configs/runpod_h200.yaml` which has:

| Setting | Value | Why |
|---------|-------|-----|
| `games_per_iteration` | 4096 | Sweet spot for speed vs data |
| `minibatch_size` | 2048 | GPU efficient |
| `async_game_generation` | true | Hides generation latency |
| `async_prefetch_batches` | 2 | Keeps 2 batches ready |
| `use_opponent_pool` | false | **Critical**: Enables fast Cython path |

---

## Gotchas & Failure Modes

### 1. rsync: command not found
```bash
apt-get update && apt-get install -y rsync
```

### 2. ModuleNotFoundError: torch
PyTorch not in uv.lock. Install manually:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Cython .so wrong architecture
The repo has macOS `.so`. Rebuild for Linux:
```bash
python _01_simulator/_cython/setup.py build_ext --inplace
```

### 4. ModuleNotFoundError: setuptools
```bash
uv pip install setuptools
```

### 5. Training extremely slow (single-threaded)
**Cause:** `use_opponent_pool: true` uses pure Python (5x slower)
**Fix:** Use `configs/runpod_h200.yaml` which has `use_opponent_pool: false`

### 6. nohup command doesn't work
Use heredoc syntax:
```bash
ssh ... << 'EOF'
nohup python ... > log 2>&1 &
EOF
```

### 7. Multiple training processes
Always kill before starting new:
```bash
pkill -f "python scripts/train.py"
```

---

## Expected Performance

| Mode | Iteration Time | Notes |
|------|----------------|-------|
| **Async + Cython** | ~30-40s | After 2-batch warmup |
| Sync + Cython | ~70s | No async prefetch |
| With opponent pool | ~225s+ | Pure Python, avoid |

**With async game generation**, after the initial warmup (pre-filling 2 batches), training pulls from queue instantly and generation happens in background.

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | Training entry point |
| `configs/runpod_h200.yaml` | **Recommended** RunPod config |
| `checkpoints/ppo_h200_maxperf/final.pt` | Best checkpoint (79% vs heuristic) |
| `_01_simulator/_cython/setup.py` | Cython build script |

---

## Checkpoint Management

**Latest checkpoint:** `checkpoints/ppo_h200_maxperf/final.pt`
- Iteration 434
- 79% vs heuristic, 97% vs random
- 7.1M games trained

**Download results after training:**
```bash
scp -P $RUNPOD_PORT -i $SSH_KEY \
  root@$RUNPOD_IP:/workspace/beastybar/checkpoints/h200_continued/*.pt \
  /Users/p/Desktop/v/experiments/beastybar/checkpoints/h200_continued/
```

---

## Troubleshooting

**Check if training is running:**
```bash
ssh ... 'ps aux | grep python | grep train'
```

**Check GPU utilization:**
```bash
ssh ... 'nvidia-smi'
```

**View live logs:**
```bash
ssh ... 'tail -f /workspace/beastybar/training.log'
```

**Kill stuck training:**
```bash
ssh ... 'pkill -f "python scripts/train.py"'
```
