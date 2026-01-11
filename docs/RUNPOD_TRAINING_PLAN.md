# RunPod Training Plan

**Prerequisite:** All local code changes from `TRAINING_OPTIMIZATION_MASTER_PLAN.md` must be complete.

---

## RunPod Instance Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | H100 80GB | H200 141GB |
| vCPUs | 32 | 48+ |
| RAM | 128GB | 256GB |
| Storage | 50GB NVMe | 100GB NVMe |

---

## Setup Steps

### 1. Launch Instance

```bash
# SSH into RunPod instance
ssh root@<runpod-ip>
```

### 2. Clone Repository

```bash
git clone <repo-url> beastybar
cd beastybar
```

### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Build Cython extensions
python setup.py build_ext --inplace
```

### 4. Verify Cython

```bash
python -c "from _01_simulator._cython import is_cython_available; print(f'Cython: {is_cython_available()}')"
# Should print: Cython: True
```

### 5. Upload Checkpoint

```bash
# From local machine
scp checkpoints/v4/final.pt root@<runpod-ip>:~/beastybar/checkpoints/v4/
```

### 6. Verify Checkpoint Loads

```bash
python -c "
import torch
cp = torch.load('checkpoints/v4/final.pt', map_location='cuda')
print(f'Iteration: {cp[\"iteration\"]}')
print(f'Keys: {list(cp.keys())}')
"
# Should print: Iteration: 599
```

---

## Training Execution

### Start Training

```bash
python scripts/train.py \
    --config configs/iter600_to_1000.yaml \
    --resume checkpoints/v4/final.pt
```

### Monitor Training

```bash
# In separate terminal
watch -n 5 nvidia-smi

# Or use W&B dashboard if configured
```

### Expected Output

```
Iteration 600/1000 | Loss: 0.0234 | LR: 9.5e-05 | WR: 79%
Iteration 625/1000 | Loss: 0.0212 | LR: 9.2e-05 | WR: 82%
...
Iteration 1000/1000 | Loss: 0.0145 | LR: 5.0e-05 | WR: 95%+
```

---

## Evaluation

### Run Full Evaluation

```bash
python eval_report.py \
    --checkpoint checkpoints/iter600_to_1000/final.pt \
    --opponents all \
    --games 100
```

### Expected Results

| Opponent | Target |
|----------|--------|
| random | 100% |
| heuristic | 100% |
| aggressive | 100% |
| defensive | 100% |
| queue | 100% |
| skunk | 100% |
| noisy | 100% |
| online | 100% |
| outcome_heuristic | 100% |
| distilled_outcome | 100% |

---

## Download Results

```bash
# From local machine
scp -r root@<runpod-ip>:~/beastybar/checkpoints/iter600_to_1000/ ./checkpoints/
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size in config
minibatch_size: 1024  # instead of 2048
games_per_iteration: 4096  # instead of 8192
```

### Cython Not Available

```bash
# Rebuild extensions
python setup.py build_ext --inplace --force
```

### Async Workers Dying

```bash
# Check logs
tail -f logs/*.log

# Fall back to sync (slower but stable)
async_game_generation: false
```
