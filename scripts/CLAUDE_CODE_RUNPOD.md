# Claude Code + RunPod Quick Reference

## TL;DR - Fast Setup Commands

```bash
# 1. Create tarball locally (excludes venv, cache, etc.)
cd /path/to/beastybar
tar --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='*.so' --exclude='*.pyc' --exclude='.DS_Store' --exclude='checkpoints' --exclude='*.egg-info' -czf /tmp/beastybar.tar.gz .

# 2. Upload via DIRECT TCP (not the proxy!)
scp -P <PORT> -i ~/.ssh/id_ed25519 /tmp/beastybar.tar.gz root@<IP>:/workspace/

# 3. Extract, install, build, run (single command)
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP> "cd /workspace && mkdir -p beastybar && cd beastybar && tar -xzf ../beastybar.tar.gz && pip install -e . && pip install cython && python _01_simulator/_cython/setup.py build_ext --inplace && nohup python scripts/train_h200.py --config configs/h200_optimized.yaml --no-wandb > training.log 2>&1 &"
```

---

## Critical: SSH Connection Types

RunPod provides TWO SSH connection methods. **Only one works for automation:**

| Method | Command | SCP/SFTP | Use Case |
|--------|---------|----------|----------|
| **Proxy SSH** | `ssh <pod-id>@ssh.runpod.io` | ❌ NO | Interactive only |
| **Direct TCP** | `ssh root@<IP> -p <PORT>` | ✅ YES | Automation, file transfer |

### Finding Direct TCP Connection
1. Go to RunPod dashboard → Your Pod → Connect tab
2. Look for "SSH over exposed TCP" section
3. Use that IP and port (e.g., `ssh root@213.181.122.175 -p 17787`)

### Common Error with Proxy SSH
```
Error: Your SSH client doesn't support PTY
subsystem request failed on channel 0
```
**Solution**: Switch to Direct TCP connection.

---

## Step-by-Step Setup

### 1. Check GPU and Environment
```bash
SSH="ssh -o StrictHostKeyChecking=no -p <PORT> -i ~/.ssh/id_ed25519 root@<IP>"
$SSH "nvidia-smi --query-gpu=name,memory.total --format=csv && python3 --version"
```

### 2. Upload Code
```bash
# Create tarball (macOS/Linux)
tar --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='*.so' --exclude='*.pyc' --exclude='.DS_Store' --exclude='checkpoints' --exclude='*.egg-info' -czf /tmp/beastybar.tar.gz .

# Upload (note: -P for port in scp)
scp -o StrictHostKeyChecking=no -P <PORT> -i ~/.ssh/id_ed25519 /tmp/beastybar.tar.gz root@<IP>:/workspace/
```

### 3. Extract and Install
```bash
$SSH "cd /workspace && mkdir -p beastybar && cd beastybar && tar -xzf ../beastybar.tar.gz"
$SSH "cd /workspace/beastybar && pip install -e ."
```

### 4. Build Cython Extension
```bash
$SSH "pip install cython && cd /workspace/beastybar && python _01_simulator/_cython/setup.py build_ext --inplace"
```

### 5. Start Training (Background)
```bash
# Use nohup so training continues after SSH disconnect
$SSH "cd /workspace/beastybar && nohup python scripts/train_h200.py --config configs/h200_optimized.yaml --no-wandb > training.log 2>&1 &"
```

---

## Monitoring

### Check Training Progress
```bash
$SSH "tail -50 /workspace/beastybar/training.log"
```

### Check if Training is Running
```bash
$SSH "ps aux | grep train_h200 | grep -v grep"
```

### Watch Live (streams output)
```bash
$SSH "tail -f /workspace/beastybar/training.log"
```

### Check GPU Usage
```bash
$SSH "nvidia-smi"
```

---

## Training Configurations

| Config | Use Case | Est. Time |
|--------|----------|-----------|
| `configs/h200_optimized.yaml` | H100/H200, full run | ~90 min |
| `configs/fast.yaml` | Quick test | ~10 min |
| `configs/production.yaml` | Long training | ~8 hrs |

### Override Parameters
```bash
python scripts/train_h200.py \
    --config configs/h200_optimized.yaml \
    --iterations 250 \
    --lr 0.0001 \
    --no-wandb
```

---

## Expected Performance (H100 80GB)

| Metric | Value |
|--------|-------|
| Iteration time | ~11-12s |
| Games/iteration | 512 |
| ETA for 500 iters | ~90 min |
| Memory usage | ~2GB VRAM |

---

## Downloading Results

```bash
# Download best model
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/beastybar/checkpoints/h200_optimized_v1/best_model.pt ./

# Download all checkpoints
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/beastybar/checkpoints/ ./
```

---

## Troubleshooting

### Training Doesn't Start
Check for Python errors:
```bash
$SSH "cd /workspace/beastybar && python scripts/train_h200.py --help"
```

### Cython Build Fails
Ensure numpy is installed first:
```bash
$SSH "pip install numpy cython && cd /workspace/beastybar && python _01_simulator/_cython/setup.py build_ext --inplace"
```

### Multiple Training Processes
Kill all and restart:
```bash
$SSH "pkill -f train_h200.py"
```

### Out of Disk Space
```bash
$SSH "df -h /workspace && rm -rf /workspace/beastybar/checkpoints/h200_optimized_v1/iter_*.pt"
```

---

## Useful One-Liners

```bash
# Quick status check
$SSH "ps aux | grep python | grep -v grep; tail -5 /workspace/beastybar/training.log 2>/dev/null"

# Get latest eval results
$SSH "grep -A10 'Evaluation Report' /workspace/beastybar/training.log | tail -15"

# Check current iteration
$SSH "grep 'Iter ' /workspace/beastybar/training.log | tail -1"

# GPU memory usage
$SSH "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
```

---

## Template: Full Setup Script

Save connection details and run:
```bash
#!/bin/bash
IP="213.181.122.175"
PORT="17787"
KEY="~/.ssh/id_ed25519"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT -i $KEY root@$IP"
SCP="scp -o StrictHostKeyChecking=no -P $PORT -i $KEY"

# Upload
tar --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='*.so' -czf /tmp/beastybar.tar.gz .
$SCP /tmp/beastybar.tar.gz root@$IP:/workspace/

# Setup and run
$SSH "cd /workspace && rm -rf beastybar && mkdir beastybar && cd beastybar && tar -xzf ../beastybar.tar.gz && pip install -e . && pip install cython && python _01_simulator/_cython/setup.py build_ext --inplace && nohup python scripts/train_h200.py --config configs/h200_optimized.yaml --no-wandb > training.log 2>&1 &"

echo "Training started! Monitor with:"
echo "$SSH \"tail -f /workspace/beastybar/training.log\""
```
