# Claude Prompt: Implement H100 GPU Optimizations

Copy everything below the line and paste into a new Claude Code session:

---

I need you to implement ALL GPU optimizations from `H100_GPU_OPTIMIZATION_PLAN.md` to max out my H100 GPU. Current utilization is only 10-15% - this is unacceptable.

**IMPORTANT**: Work aggressively. Spin up 6+ subagents in parallel to implement everything simultaneously. Do NOT ask for confirmation - just implement all changes.

## Your Tasks (Run ALL in Parallel via Subagents)

### Subagent 1: Create Optimized Config
- Create `configs/h100_maxout.yaml` with ALL the optimized settings from the plan
- batch_size: 16384, parallel_games: 512, games_per_iteration: 2048, num_simulations: 200
- torch_compile_mode: "max-autotune-no-cudagraphs"

### Subagent 2: Fix train_alphazero.py
- Add TF32 precision settings at the start of main():
  ```python
  torch.set_float32_matmul_precision('high')
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = True
  ```

### Subagent 3: Fix network_v2.py Dynamic Masking
- Fix `AsymmetricTransformerEncoder.forward()` (lines ~296-335) - remove `x[non_empty_mask]` indexing
- Fix `DuelingHead.forward()` (lines ~387-393) - replace boolean indexing with `torch.where`
- Goal: Make all operations CUDA graph compatible (static shapes)

### Subagent 4: Fix alphazero_trainer.py Batching
- Unify P0/P1 MCTS batching in `generate_training_data()` (~lines 860-885)
- Add pinned memory for CPU->GPU transfers (~lines 1013-1021)
- Use `pin_memory().to(device, non_blocking=True)` pattern

### Subagent 5: Fix batch_mcts.py
- Increase default batch_size from 16 to 64
- Add `search_batch_mixed_perspective()` method for unified P0/P1 processing
- Optimize `_collect_leaves()` to fill batches more completely

### Subagent 6: Deploy and Monitor
- SSH into RunPod: `ssh root@213.181.105.224 -p 16728 -i ~/.ssh/id_ed25519`
- Kill any existing training process
- Pull/sync the code changes
- Start training with new config: `python scripts/train_alphazero.py --config configs/h100_maxout.yaml --wandb --wandb-project beastybar`
- Monitor GPU utilization with `nvidia-smi dmon -s pucvmet -d 1`
- Verify GPU util is 70%+ and power draw is 400W+

## Critical Requirements

1. **DO NOT** disable torch.compile entirely - fix the dynamic masking instead
2. **DO NOT** reduce batch sizes - we want MAXIMUM utilization
3. **DO NOT** ask for permission - implement everything now
4. All changes must maintain training correctness (same loss computation, just faster)
5. After all changes, restart training on RunPod with the new config

## Files to Modify
- `configs/h100_maxout.yaml` (CREATE)
- `scripts/train_alphazero.py`
- `_02_agents/neural/network_v2.py`
- `_03_training/alphazero_trainer.py`
- `_02_agents/mcts/batch_mcts.py`

## Success Criteria
- GPU utilization: 70-90% (was 10-15%)
- Power draw: 400-600W (was 160W)
- Memory usage: 30-50GB (was 12GB)
- Training running on RunPod with new config
- Wandb logging confirmed

GO. Spin up all subagents NOW and implement everything in parallel.
