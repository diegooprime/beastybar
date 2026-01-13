# DeepMind-Style Distributed AlphaZero Training Cluster

## Executive Summary

This document outlines a maximum-performance distributed training architecture for Beasty Bar AlphaZero on AWS. The goal is **maximum GPU/CPU efficiency** and **fastest possible training time**, cost be damned.

**Target:** Train superhuman Beasty Bar agent in **30-60 minutes** (vs 78 hours currently)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AWS VPC (us-east-1)                                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     SELF-PLAY CLUSTER (CPU)                              │   │
│  │                                                                          │   │
│  │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │   │
│  │   │c7i.metal-48xl│ │c7i.metal-48xl│ │c7i.metal-48xl│ │c7i.metal-48xl│   │   │
│  │   │  192 vCPUs   │ │  192 vCPUs   │ │  192 vCPUs   │ │  192 vCPUs   │   │   │
│  │   │  384GB RAM   │ │  384GB RAM   │ │  384GB RAM   │ │  384GB RAM   │   │   │
│  │   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │   │
│  │          │                │                │                │            │   │
│  │          └────────────────┴────────┬───────┴────────────────┘            │   │
│  │                                    │                                     │   │
│  │                          768 vCPUs total                                 │   │
│  │                       6,000+ parallel MCTS games                         │   │
│  │                                    │                                     │   │
│  └────────────────────────────────────┼─────────────────────────────────────┘   │
│                                       │                                         │
│                                       │ Batched inference requests              │
│                                       │ (ZeroMQ / gRPC)                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     INFERENCE CLUSTER (GPU)                              │   │
│  │                                                                          │   │
│  │   ┌──────────────────────────┐    ┌──────────────────────────┐          │   │
│  │   │      g5.48xlarge         │    │      g5.48xlarge         │          │   │
│  │   │      8x A10G 24GB        │    │      8x A10G 24GB        │          │   │
│  │   │      192GB VRAM          │    │      192GB VRAM          │          │   │
│  │   │                          │    │                          │          │   │
│  │   │   Inference Server 0-7   │    │   Inference Server 8-15  │          │   │
│  │   └──────────────────────────┘    └──────────────────────────┘          │   │
│  │                                                                          │   │
│  │                        16 GPUs total                                     │   │
│  │                   ~500,000 inferences/sec                                │   │
│  │                                                                          │   │
│  └─────────────────────────────────────┬────────────────────────────────────┘   │
│                                        │                                        │
│                                        │ Training examples                      │
│                                        │ Model weight sync                      │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      TRAINING SERVER (GPU)                               │   │
│  │                                                                          │   │
│  │   ┌────────────────────────────────────────────────────────────────┐    │   │
│  │   │                      p5.48xlarge                                │    │   │
│  │   │                      8x H100 80GB                               │    │   │
│  │   │                      640GB VRAM                                 │    │   │
│  │   │                                                                 │    │   │
│  │   │   - Continuous training on replay buffer                       │    │   │
│  │   │   - Pushes new weights to inference servers every N steps      │    │   │
│  │   │   - Saves checkpoints to S3                                    │    │   │
│  │   └────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      COORDINATION (Redis/S3)                             │   │
│  │                                                                          │   │
│  │   - ElastiCache Redis: Replay buffer, weight versioning                 │   │
│  │   - S3: Checkpoint storage, model weights distribution                  │   │
│  │   - CloudWatch: Monitoring and logging                                  │   │
│  │                                                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Instance Specifications

### Self-Play Cluster (CPU-Optimized)

| Component | Spec |
|-----------|------|
| Instance Type | `c7i.metal-48xl` |
| Count | 4 |
| vCPUs per instance | 192 |
| Total vCPUs | **768** |
| RAM per instance | 384 GB |
| Total RAM | 1.5 TB |
| Network | 50 Gbps |
| Cost per instance | $8.16/hr |
| **Total Cost** | **$32.64/hr** |

**Why c7i.metal-48xl:**
- Bare metal = no virtualization overhead
- Intel Xeon Sapphire Rapids (best single-thread perf)
- Maximum cores per dollar for compute-bound MCTS

### Inference Cluster (GPU)

| Component | Spec |
|-----------|------|
| Instance Type | `g5.48xlarge` |
| Count | 2 |
| GPUs per instance | 8x NVIDIA A10G |
| VRAM per instance | 192 GB (8x 24GB) |
| Total GPUs | **16** |
| Total VRAM | 384 GB |
| vCPUs per instance | 192 |
| Cost per instance | $16.29/hr |
| **Total Cost** | **$32.58/hr** |

**Why A10G over H100 for inference:**
- A10G has excellent inference perf/$ ratio
- Lower latency for small batch inference
- H100 overkill for inference-only workload
- 16x A10G can serve 500K+ inferences/sec

### Training Server (GPU)

| Component | Spec |
|-----------|------|
| Instance Type | `p5.48xlarge` |
| Count | 1 |
| GPUs | 8x NVIDIA H100 80GB |
| Total VRAM | 640 GB |
| vCPUs | 192 |
| RAM | 2 TB |
| Network | 3200 Gbps EFA |
| **Cost** | **$98.32/hr** |

**Why p5.48xlarge:**
- H100 is fastest for training (large batches, backprop)
- 8 GPUs enables data-parallel training if needed
- EFA network for future multi-node scaling

### Coordination Services

| Service | Spec | Cost |
|---------|------|------|
| ElastiCache Redis | r6g.2xlarge | ~$0.50/hr |
| S3 | Standard | ~$0.02/hr |
| CloudWatch | Basic | ~$0.10/hr |
| **Total** | | **~$0.62/hr** |

---

## Total Cost Summary

| Component | Cost/hr |
|-----------|---------|
| Self-Play Cluster (4x c7i.metal-48xl) | $32.64 |
| Inference Cluster (2x g5.48xlarge) | $32.58 |
| Training Server (1x p5.48xlarge) | $98.32 |
| Coordination Services | $0.62 |
| **TOTAL** | **$164.16/hr** |

### Training Cost Projection

| Scenario | Time | Cost |
|----------|------|------|
| Full training (2000 iterations) | 30-45 min | **$82-123** |
| Extended training (5000 iterations) | 1-2 hours | **$164-328** |
| Hyperparameter search (10 runs) | 5-8 hours | **$820-1,312** |

**Compared to current setup:**
- Current: 78 hours × $2/hr = $156
- This cluster: 0.75 hours × $164/hr = **$123**

**Faster AND potentially cheaper!**

---

## Data Flow

### 1. Self-Play Loop

```python
# On each CPU worker (768 total workers)
while training_active:
    # 1. Get latest model weights from Redis
    weights = redis.get("model_weights_v{version}")

    # 2. Run MCTS game
    game_data = []
    state = env.reset()

    while not state.is_terminal():
        # 3. MCTS with batched inference
        policy, value = mcts.search(
            state,
            num_simulations=200,
            inference_client=inference_client  # Calls inference cluster
        )

        action = sample_action(policy)
        game_data.append((state, policy, None))  # Value filled at end
        state = env.step(action)

    # 4. Assign values based on game outcome
    outcome = state.get_outcome()
    for i, (s, p, _) in enumerate(game_data):
        game_data[i] = (s, p, outcome * (discount ** (len(game_data) - i)))

    # 5. Push to replay buffer
    redis.rpush("replay_buffer", serialize(game_data))
```

### 2. Inference Server

```python
# On each inference GPU (16 total)
@torch.inference_mode()
def inference_server(gpu_id):
    model = load_model().to(f"cuda:{gpu_id}")

    while True:
        # 1. Collect batch of requests (wait up to 1ms)
        batch = collect_requests(max_batch=512, timeout_ms=1)

        # 2. Run batched inference
        observations = torch.stack([r.obs for r in batch]).cuda()
        policies, values = model(observations)

        # 3. Return results to workers
        for req, policy, value in zip(batch, policies, values):
            req.respond(policy.cpu(), value.cpu())

        # 4. Check for weight updates
        if redis.get("model_version") > current_version:
            model.load_state_dict(redis.get("model_weights"))
            current_version += 1
```

### 3. Training Loop

```python
# On training server (p5.48xlarge)
def training_loop():
    model = BeastyBarNetworkV2().cuda()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    iteration = 0
    while iteration < total_iterations:
        # 1. Sample from distributed replay buffer
        batch = sample_from_redis_buffer(batch_size=16384)

        # 2. Training step
        obs, target_policy, target_value = batch
        pred_policy, pred_value = model(obs)

        loss = policy_loss(pred_policy, target_policy) + value_loss(pred_value, target_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Push new weights every N steps
        if iteration % weight_push_frequency == 0:
            redis.set("model_weights", model.state_dict())
            redis.incr("model_version")

        # 4. Save checkpoint to S3
        if iteration % checkpoint_frequency == 0:
            save_to_s3(model, f"checkpoints/iter_{iteration}.pt")

        iteration += 1
```

---

## Performance Estimates

### Throughput Calculations

**Self-Play:**
```
768 vCPUs × 0.8 efficiency = 614 effective workers
614 workers × 1 game/10 seconds = 61 games/second
61 games/sec × 3600 sec/hr = 220,000 games/hour
```

**Inference:**
```
16 A10G GPUs × 30,000 inferences/sec each = 480,000 inferences/sec
Each MCTS simulation = 1 inference
200 sims/move × 25 moves/game = 5,000 inferences/game
480,000 / 5,000 = 96 games/sec inference capacity (bottleneck check: OK)
```

**Training:**
```
H100 throughput: ~50,000 samples/sec with batch_size=16384
220,000 games/hr × 23 examples/game = 5,060,000 examples/hr
Training can easily keep up (10x headroom)
```

### Expected Timeline

| Phase | Duration |
|-------|----------|
| Infrastructure spin-up | 5 min |
| Initial buffer fill (50K examples) | 2 min |
| Main training (2000 iterations) | 25-35 min |
| Final evaluation | 5 min |
| **Total** | **~45 min** |

---

## Implementation Plan

### Phase 1: Infrastructure (Week 1)

#### 1.1 AWS Setup
- [ ] Create VPC with private subnets
- [ ] Configure security groups (internal only)
- [ ] Set up ElastiCache Redis cluster
- [ ] Create S3 bucket for checkpoints
- [ ] Create launch templates for each instance type
- [ ] Set up CloudWatch dashboards

#### 1.2 Docker Images
- [ ] Base image with PyTorch 2.x + CUDA 12.x
- [ ] Self-play worker image
- [ ] Inference server image
- [ ] Training server image

### Phase 2: Distributed Training Code (Week 1-2)

#### 2.1 Communication Layer
```
_05_distributed/
├── __init__.py
├── redis_client.py          # Redis connection + serialization
├── inference_client.py      # Client for workers to call inference
├── inference_server.py      # GPU inference server
├── replay_buffer.py         # Distributed replay buffer
└── weight_sync.py           # Model weight synchronization
```

#### 2.2 Code Changes

**New file: `_05_distributed/inference_client.py`**
```python
class InferenceClient:
    """Client for self-play workers to request batched inference."""

    def __init__(self, server_addresses: list[str]):
        self.servers = server_addresses
        self.current_server = 0

    async def infer(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Request inference from cluster (load-balanced)."""
        server = self.servers[self.current_server]
        self.current_server = (self.current_server + 1) % len(self.servers)

        response = await self._send_request(server, observation)
        return response.policy, response.value

    async def infer_batch(self, observations: list[np.ndarray]) -> list[tuple]:
        """Batch inference for efficiency."""
        # Implementation...
```

**New file: `_05_distributed/inference_server.py`**
```python
class InferenceServer:
    """GPU server that batches and serves inference requests."""

    def __init__(self, model: nn.Module, device: str, port: int):
        self.model = model.to(device).eval()
        self.device = device
        self.request_queue = asyncio.Queue()
        self.batch_size = 512
        self.batch_timeout_ms = 1

    async def run(self):
        """Main server loop."""
        while True:
            batch = await self._collect_batch()
            results = await self._process_batch(batch)
            await self._send_results(batch, results)

    @torch.inference_mode()
    async def _process_batch(self, batch):
        obs = torch.stack([r.obs for r in batch]).to(self.device)
        policy, value = self.model(obs)
        return policy.cpu().numpy(), value.cpu().numpy()
```

**New file: `_05_distributed/replay_buffer.py`**
```python
class DistributedReplayBuffer:
    """Redis-backed distributed replay buffer."""

    def __init__(self, redis_url: str, max_size: int = 5_000_000):
        self.redis = Redis.from_url(redis_url)
        self.max_size = max_size
        self.key = "replay_buffer"

    def push(self, examples: list[tuple]):
        """Push examples to buffer."""
        serialized = pickle.dumps(examples)
        self.redis.rpush(self.key, serialized)

        # Trim if over max size
        if self.redis.llen(self.key) > self.max_size:
            self.redis.ltrim(self.key, -self.max_size, -1)

    def sample(self, batch_size: int) -> tuple:
        """Sample random batch from buffer."""
        indices = np.random.randint(0, self.redis.llen(self.key), batch_size)
        # Implementation with pipelining for efficiency...
```

**Modified: `_02_agents/mcts/batch_mcts.py`**
```python
class DistributedBatchMCTS(BatchMCTS):
    """MCTS that uses distributed inference cluster."""

    def __init__(self, inference_client: InferenceClient, **kwargs):
        super().__init__(**kwargs)
        self.inference_client = inference_client

    async def _evaluate_leaves(self, leaves: list[MCTSNode]):
        """Evaluate leaves using inference cluster instead of local GPU."""
        observations = [self._get_observation(leaf) for leaf in leaves]
        results = await self.inference_client.infer_batch(observations)

        for leaf, (policy, value) in zip(leaves, results):
            leaf.expand(policy)
            leaf.backup(value)
```

#### 2.3 Training Script Changes

**New file: `scripts/train_distributed.py`**
```python
"""Distributed AlphaZero training on AWS cluster."""

import argparse
from _05_distributed import (
    DistributedReplayBuffer,
    WeightSynchronizer,
    start_inference_server,
    start_selfplay_worker,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["trainer", "inference", "selfplay"])
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    if args.role == "trainer":
        run_trainer(args)
    elif args.role == "inference":
        run_inference_server(args)
    elif args.role == "selfplay":
        run_selfplay_workers(args)
```

### Phase 3: Deployment Scripts (Week 2)

#### 3.1 Terraform Infrastructure
```hcl
# infrastructure/main.tf
module "vpc" {
  source = "./modules/vpc"
  cidr   = "10.0.0.0/16"
}

module "redis" {
  source        = "./modules/elasticache"
  instance_type = "r6g.2xlarge"
  vpc_id        = module.vpc.vpc_id
}

module "selfplay_cluster" {
  source         = "./modules/ec2_cluster"
  instance_type  = "c7i.metal-48xl"
  instance_count = 4
  ami            = var.selfplay_ami
}

module "inference_cluster" {
  source         = "./modules/ec2_cluster"
  instance_type  = "g5.48xlarge"
  instance_count = 2
  ami            = var.inference_ami
}

module "training_server" {
  source        = "./modules/ec2_instance"
  instance_type = "p5.48xlarge"
  ami           = var.training_ami
}
```

#### 3.2 Launch Script
```bash
#!/bin/bash
# scripts/launch_cluster.sh

set -e

echo "=== Launching DeepMind-Style Training Cluster ==="

# 1. Apply Terraform
cd infrastructure
terraform apply -auto-approve

# 2. Get instance IPs
REDIS_URL=$(terraform output -raw redis_url)
SELFPLAY_IPS=$(terraform output -json selfplay_ips | jq -r '.[]')
INFERENCE_IPS=$(terraform output -json inference_ips | jq -r '.[]')
TRAINER_IP=$(terraform output -raw trainer_ip)

# 3. Start inference servers
for ip in $INFERENCE_IPS; do
  ssh ubuntu@$ip "docker run -d --gpus all \
    -e REDIS_URL=$REDIS_URL \
    beastybar/inference:latest \
    --role inference --redis-url $REDIS_URL"
done

# 4. Start self-play workers
for ip in $SELFPLAY_IPS; do
  ssh ubuntu@$ip "docker run -d \
    -e REDIS_URL=$REDIS_URL \
    -e INFERENCE_SERVERS=$INFERENCE_IPS \
    beastybar/selfplay:latest \
    --role selfplay --redis-url $REDIS_URL"
done

# 5. Start trainer
ssh ubuntu@$TRAINER_IP "docker run -d --gpus all \
  -e REDIS_URL=$REDIS_URL \
  beastybar/trainer:latest \
  --role trainer --redis-url $REDIS_URL --config /configs/cluster_maxout.yaml"

echo "=== Cluster launched! ==="
echo "Monitor at: http://$TRAINER_IP:8080/dashboard"
```

### Phase 4: Monitoring & Observability (Week 2)

#### 4.1 Metrics to Track
- Games generated per second (per worker, total)
- Inference latency (p50, p95, p99)
- Training throughput (samples/sec)
- GPU utilization (all GPUs)
- CPU utilization (all workers)
- Replay buffer size
- Model version / weight sync latency
- Loss curves (policy, value, total)

#### 4.2 Grafana Dashboard
```yaml
# monitoring/grafana/dashboards/cluster.json
{
  "panels": [
    {"title": "Games/Second", "type": "graph", "targets": [...]},
    {"title": "GPU Utilization", "type": "gauge", "targets": [...]},
    {"title": "Inference Latency", "type": "heatmap", "targets": [...]},
    {"title": "Training Loss", "type": "graph", "targets": [...]},
    {"title": "Replay Buffer Size", "type": "stat", "targets": [...]}
  ]
}
```

---

## Configuration

### Cluster Configuration File

```yaml
# configs/cluster_maxout.yaml

# Cluster settings
cluster:
  redis_url: "redis://elasticache.internal:6379"
  inference_servers:
    - "inference-0.internal:50051"
    - "inference-1.internal:50051"
  weight_sync_frequency: 100  # Push weights every N training steps

# Self-play settings
selfplay:
  workers_per_node: 150  # Leave some CPU for overhead
  games_per_worker: 1000000  # Effectively infinite

# MCTS settings
mcts:
  num_simulations: 200
  c_puct: 1.5
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25
  batch_size: 64  # Inference batch size per worker

# Inference server settings
inference:
  batch_size: 512  # Max batch size
  batch_timeout_ms: 1  # Max wait time to fill batch
  gpus_per_server: 8

# Training settings
training:
  batch_size: 16384
  learning_rate: 0.0001
  weight_decay: 0.0001
  total_iterations: 2000
  checkpoint_frequency: 100
  eval_frequency: 200

# Replay buffer
buffer:
  max_size: 5000000  # 5M examples
  min_size: 50000    # Start training after this many

# Network architecture (unchanged)
network_config:
  hidden_dim: 256
  num_heads: 8
  queue_layers: 6
  bar_layers: 2
  hand_layers: 2
  fusion_layers: 4
  use_dueling: true
  use_auxiliary_heads: true
```

---

## Risk Mitigation

### Potential Issues & Solutions

| Risk | Mitigation |
|------|------------|
| Inference bottleneck | Scale inference cluster horizontally |
| Redis memory overflow | Set max memory policy, use Redis Cluster |
| Network saturation | Use placement groups, EFA for training |
| Stale weights | Implement version checking, force refresh |
| Worker crashes | Auto-restart with systemd, health checks |
| Cost overrun | Set billing alerts, auto-shutdown after N hours |

### Fallback Plan

If distributed setup is too complex, fall back to **single p5.48xlarge**:
- 8x H100 + 192 vCPUs
- Run inference on GPUs 1-7, training on GPU 0
- ~$98/hr, ~3-4 hours training
- Much simpler, still 20x faster than current

---

## Quick Start Commands

### Option A: Full Cluster (Maximum Performance)
```bash
# 1. Clone and setup
git clone https://github.com/diegooprime/beastybar.git
cd beastybar

# 2. Configure AWS credentials
aws configure

# 3. Launch cluster
./scripts/launch_cluster.sh --config configs/cluster_maxout.yaml

# 4. Monitor
open http://<trainer-ip>:8080/dashboard

# 5. Shutdown when done
./scripts/shutdown_cluster.sh
```

### Option B: Single Instance (Simpler)
```bash
# 1. Launch p5.48xlarge
aws ec2 run-instances \
  --instance-type p5.48xlarge \
  --image-id ami-xxxxx \
  --key-name my-key

# 2. SSH and run
ssh ubuntu@<instance-ip>
cd beastybar
python scripts/train_alphazero.py --config configs/h100_8gpu_maxout.yaml
```

---

## Timeline Summary

| Week | Tasks |
|------|-------|
| Week 1 | AWS infrastructure setup, Docker images, communication layer |
| Week 2 | Distributed training code, deployment scripts, testing |
| Week 3 | Monitoring, optimization, production runs |

---

## Appendix A: AWS CLI Commands

```bash
# Launch self-play instance
aws ec2 run-instances \
  --instance-type c7i.metal-48xl \
  --count 4 \
  --image-id ami-xxxxx \
  --subnet-id subnet-xxxxx \
  --security-group-ids sg-xxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Role,Value=selfplay}]'

# Launch inference instance
aws ec2 run-instances \
  --instance-type g5.48xlarge \
  --count 2 \
  --image-id ami-xxxxx \
  --subnet-id subnet-xxxxx \
  --security-group-ids sg-xxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Role,Value=inference}]'

# Launch training instance
aws ec2 run-instances \
  --instance-type p5.48xlarge \
  --count 1 \
  --image-id ami-xxxxx \
  --subnet-id subnet-xxxxx \
  --security-group-ids sg-xxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Role,Value=trainer}]'
```

---

## Appendix B: Cost Calculator

```python
# Quick cost estimation
def estimate_cost(training_hours: float) -> dict:
    hourly_costs = {
        "selfplay_cluster": 4 * 8.16,      # 4x c7i.metal-48xl
        "inference_cluster": 2 * 16.29,    # 2x g5.48xlarge
        "training_server": 98.32,          # 1x p5.48xlarge
        "coordination": 0.62,              # Redis + S3 + CloudWatch
    }

    total_hourly = sum(hourly_costs.values())
    total_cost = total_hourly * training_hours

    return {
        "hourly_breakdown": hourly_costs,
        "total_hourly": total_hourly,
        "training_hours": training_hours,
        "total_cost": total_cost,
    }

# Example: 45 minute training run
print(estimate_cost(0.75))
# {'total_hourly': 164.16, 'total_cost': 123.12}
```

---

## Next Steps

1. **Approve this plan** - Review architecture and costs
2. **Set up AWS account** - Ensure p5 instance quota (need to request)
3. **Implement Phase 1** - Infrastructure and Docker images
4. **Test with smaller cluster** - 1 of each instance type
5. **Scale up** - Full cluster for production training

---

*Document version: 1.0*
*Last updated: 2026-01-13*
*Author: Claude + Human collaboration*
