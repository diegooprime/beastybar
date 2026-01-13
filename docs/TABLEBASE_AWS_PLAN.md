# Endgame Tablebase: AWS Generation Plan

**Target:** 10 cards, ~400M positions, AWS c7i.48xlarge, ~1 hour, ~$10

---

## Overview

### Current State
- `_02_agents/tablebase/endgame.py` has forward minimax with alpha-beta
- Single-threaded, on-demand solving
- Config limits: 10 cards max, 1M cache

### Target State
- Retrograde analysis (backward from terminal positions)
- 192-thread parallel generation
- Memory-mapped storage for large tablebases
- Systematic position enumeration

---

## Phase 1: Local Implementation (Do First)

### 1.1 Create Retrograde Analyzer

**File:** `_02_agents/tablebase/retrograde.py`

```python
# Key components needed:
class RetrogradeTablebase:
    """Build tablebase backward from terminal positions."""

    def enumerate_positions(self, max_cards: int) -> Iterator[State]:
        """Systematically enumerate all valid endgame positions."""
        # Enumerate card distributions across zones
        # - Each player: hand (0-4), deck (0-N)
        # - Queue: ordered, 0-5 cards
        # - Bar/thats_it: unordered
        pass

    def classify_terminal(self, positions: List[State]) -> Dict[Key, Value]:
        """Phase 1: Mark all terminal positions WIN/LOSS/DRAW."""
        pass

    def propagate_backward(self, solved: Dict, unsolved: Set) -> int:
        """Phase 2: Propagate values backward until fixed point."""
        # If ANY successor is LOSS for opponent -> WIN
        # If ALL successors are WIN for opponent -> LOSS
        # Returns number of newly solved positions
        pass
```

### 1.2 Add Parallel Execution

**File:** `_02_agents/tablebase/parallel.py`

```python
from multiprocessing import Pool, shared_memory
import numpy as np

class ParallelTablebaseGenerator:
    """Generate tablebase using multiple processes."""

    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or os.cpu_count()

    def generate(self, max_cards: int) -> EndgameTablebase:
        # 1. Enumerate positions (single-threaded, fast)
        # 2. Partition by hash prefix for parallel solving
        # 3. Use shared memory for cross-partition lookups
        # 4. Merge results
        pass
```

### 1.3 Memory-Mapped Storage

**File:** `_02_agents/tablebase/storage.py`

```python
import mmap
import struct

class MMapTablebase:
    """Memory-mapped tablebase for large position counts."""

    # Format: 2 bytes per position
    # - Bits 0-1: Value (WIN/LOSS/DRAW/UNKNOWN)
    # - Bits 2-7: Depth to end (0-63)
    # - Bits 8-15: Action index (0-255)

    def __init__(self, path: Path, num_positions: int):
        self.path = path
        self.size = num_positions * 2  # 2 bytes per entry
        # Create or open memory-mapped file
        pass
```

### 1.4 Position Enumerator

**File:** `_02_agents/tablebase/enumerate.py`

```python
from itertools import combinations, permutations

class PositionEnumerator:
    """Systematically enumerate all valid endgame positions."""

    def count_positions(self, max_cards: int) -> int:
        """Count total positions without generating them."""
        pass

    def enumerate(self, max_cards: int) -> Iterator[Tuple[Key, State]]:
        """Generate all positions with their canonical keys."""
        # For each total card count 1..max_cards:
        #   For each distribution to players:
        #     For each queue arrangement (ordered):
        #       For each bar/thats_it arrangement (unordered):
        #         Yield (canonical_key, state)
        pass
```

---

## Phase 2: Local Validation

### 2.1 Test with Small Card Counts

```bash
# In project directory
cd /Users/p/Desktop/v/experiments/beastybar

# Test enumeration counts
python -c "
from _02_agents.tablebase.enumerate import PositionEnumerator
e = PositionEnumerator()
for n in range(2, 7):
    print(f'{n} cards: {e.count_positions(n):,} positions')
"

# Expected rough output:
# 2 cards: ~500 positions
# 3 cards: ~3,000 positions
# 4 cards: ~15,000 positions
# 5 cards: ~80,000 positions
# 6 cards: ~400,000 positions
```

### 2.2 Validate Retrograde vs Forward

```bash
# Solve same positions both ways, compare results
python -c "
from _02_agents.tablebase.retrograde import RetrogradeTablebase
from _02_agents.tablebase.endgame import EndgameTablebase

retro = RetrogradeTablebase()
forward = EndgameTablebase()

# Generate small tablebase
retro_tb = retro.generate(max_cards=4)
forward_tb = forward.generate_from_games(num_games=1000)

# Compare overlapping positions
mismatches = 0
for key, entry in retro_tb.items():
    if key in forward_tb and forward_tb[key].value != entry.value:
        mismatches += 1
print(f'Mismatches: {mismatches}')
"
```

### 2.3 Benchmark Parallelization Locally

```bash
# Test on NUC via SSH
ssh primenuc@prime-nuc "cd /path/to/beastybar && python scripts/benchmark_parallel.py"

# Expected: ~6-7x speedup with 8 threads on 6-card tablebase
```

---

## Phase 3: AWS Setup

### 3.1 Prerequisites

```bash
# Install AWS CLI if not present
brew install awscli  # macOS
# or: pip install awscli

# Configure credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output (json)
```

### 3.2 Launch Instance

```bash
# Launch c7i.48xlarge spot instance (cheaper, ~$3/hr vs $8.57)
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type c7i.48xlarge \
  --key-name your-key-name \
  --security-group-ids sg-xxxxxxxx \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"5.00"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tablebase-gen}]'

# Note the instance ID from output
```

### 3.3 Connect and Setup

```bash
# Get public IP
aws ec2 describe-instances --instance-ids i-xxxxxxxxx \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# SSH in
ssh -i ~/.ssh/your-key.pem ubuntu@<public-ip>

# On the instance:
sudo apt update && sudo apt install -y python3-pip python3-venv git htop

# Clone repo
git clone <your-repo-url> beastybar
cd beastybar

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

---

## Phase 4: Execute Generation

### 4.1 Run Generation Script

```bash
# On AWS instance
cd beastybar
source venv/bin/activate

# Run with monitoring
python scripts/generate_tablebase.py \
  --max-cards 10 \
  --workers 192 \
  --output tablebase_10cards.tb \
  --checkpoint-interval 300 \
  2>&1 | tee generation.log &

# Monitor progress
htop  # Watch CPU usage (should be ~95%+ across all cores)
tail -f generation.log
```

### 4.2 Generation Script

**File:** `scripts/generate_tablebase.py`

```python
#!/usr/bin/env python3
"""Generate endgame tablebase on high-core-count machine."""

import argparse
import logging
import time
from pathlib import Path

from _02_agents.tablebase.parallel import ParallelTablebaseGenerator
from _02_agents.tablebase.storage import MMapTablebase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cards', type=int, default=10)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-interval', type=int, default=300)
    args = parser.parse_args()

    logger.info(f"Starting generation: {args.max_cards} cards, {args.workers} workers")

    generator = ParallelTablebaseGenerator(num_workers=args.workers)

    start = time.time()
    tablebase = generator.generate(
        max_cards=args.max_cards,
        checkpoint_path=args.output.with_suffix('.checkpoint'),
        checkpoint_interval=args.checkpoint_interval,
    )
    elapsed = time.time() - start

    logger.info(f"Generation complete: {len(tablebase)} positions in {elapsed:.1f}s")

    # Save final tablebase
    tablebase.save(args.output)
    logger.info(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
```

---

## Phase 5: Retrieve and Cleanup

### 5.1 Download Results

```bash
# From local machine
scp -i ~/.ssh/your-key.pem ubuntu@<public-ip>:~/beastybar/tablebase_10cards.tb ./

# Also grab logs
scp -i ~/.ssh/your-key.pem ubuntu@<public-ip>:~/beastybar/generation.log ./
```

### 5.2 Terminate Instance

```bash
# IMPORTANT: Don't forget this or you keep paying!
aws ec2 terminate-instances --instance-ids i-xxxxxxxxx

# Verify termination
aws ec2 describe-instances --instance-ids i-xxxxxxxxx \
  --query 'Reservations[0].Instances[0].State.Name'
```

### 5.3 Validate Results

```bash
# Locally
python -c "
from _02_agents.tablebase.endgame import load_tablebase

tb = load_tablebase('tablebase_10cards.tb')
stats = tb.get_stats()
print(f'Positions: {stats[\"positions_cached\"]:,}')
print(f'Size: {Path(\"tablebase_10cards.tb\").stat().st_size / 1e6:.1f} MB')
"
```

---

## File Checklist

New files to create:

- [ ] `_02_agents/tablebase/retrograde.py` - Backward analysis
- [ ] `_02_agents/tablebase/parallel.py` - Multi-process generation
- [ ] `_02_agents/tablebase/storage.py` - Memory-mapped storage
- [ ] `_02_agents/tablebase/enumerate.py` - Position enumeration
- [ ] `scripts/generate_tablebase.py` - CLI runner
- [ ] `scripts/benchmark_parallel.py` - Local benchmarking

Files to modify:

- [ ] `_02_agents/tablebase/endgame.py` - Add integration points
- [ ] `_02_agents/tablebase/__init__.py` - Export new classes

---

## Cost Estimate

| Item | Cost |
|------|------|
| c7i.48xlarge spot (~1 hr) | ~$3-5 |
| 100GB gp3 storage | ~$0.50 |
| Data transfer | ~$0.10 |
| **Total** | **~$5-6** |

---

## Quick Reference Commands

```bash
# Launch spot instance
aws ec2 run-instances --instance-type c7i.48xlarge --instance-market-options MarketType=spot ...

# Check spot price
aws ec2 describe-spot-price-history --instance-types c7i.48xlarge --product-descriptions "Linux/UNIX" --max-items 5

# Monitor instance
aws ec2 describe-instances --instance-ids i-xxx --query 'Reservations[0].Instances[0].State.Name'

# Terminate
aws ec2 terminate-instances --instance-ids i-xxx
```

---

## Timeline

1. **Phase 1** (2-3 hrs): Implement retrograde + parallel locally
2. **Phase 2** (30 min): Validate on small card counts
3. **Phase 3** (15 min): AWS setup and launch
4. **Phase 4** (1 hr): Run generation
5. **Phase 5** (15 min): Download and cleanup

**Total: ~4-5 hours of work, ~1 hour of AWS compute**
