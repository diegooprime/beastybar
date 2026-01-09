# Training Optimization Plan

## Overview

Three major improvements to accelerate AI training:
1. **Scale Up** - Larger network, bigger batches (Easy, 1-2 hours)
2. **Fill GPU** - Maximize H100 utilization (Easy, 1 hour)
3. **Better Algorithm** - MCTS + Neural Network (Medium, 4-6 hours)

---

## Parallel Subagent Execution

**Run these subagents in parallel to complete Phase 3 faster (~2 hours instead of 4-6):**

### Launch Command (copy-paste this entire block)
```
Run these 4 tasks IN PARALLEL using the Task tool:

AGENT 1 (python-pro): "Create configs/h100_scaled.yaml and configs/h100_mcts.yaml
- h100_scaled.yaml: Copy h200_optimized.yaml, set hidden_dim=256, num_heads=8, num_layers=4, species_embedding_dim=64, minibatch_size=512, games_per_iteration=2048, num_workers=32, buffer_size=400000
- h100_mcts.yaml: Same network config + mcts_config section with num_simulations=200, c_puct=1.5, temperature=1.0, dirichlet_alpha=0.3"

AGENT 2 (python-pro): "Create _02_agents/mcts/node.py with MCTSNode class
- Properties: state, parent, children dict, visit_count, value_sum, prior
- Methods: ucb_score(), expand(), backup(), select_child(), is_expanded, is_terminal
- Use existing _01_simulator/engine.py for game logic
- Include __init__.py for the mcts module"

AGENT 3 (python-pro): "Create _02_agents/mcts/search.py with MCTS class
- __init__(network, num_simulations, c_puct, dirichlet_alpha)
- search(state) -> Dict[Action, float] with select/expand/backup loop
- Integrate with BeastyBarNetwork for policy/value predictions
- Add temperature-based action sampling"

AGENT 4 (python-pro): "Create _03_training/mcts_self_play.py and _03_training/mcts_trainer.py
- mcts_self_play.py: generate_mcts_games() function
- mcts_trainer.py: MCTSTrainer class with train_iteration()
- Loss = MSE(value, outcome) + CrossEntropy(policy, mcts_visits)
- Follow patterns from existing trainer.py and self_play.py"
```

### After Parallel Agents Complete
```
Run sequentially:

AGENT 5 (python-pro): "Create _02_agents/mcts/batch_mcts.py for optimized batched search
- BatchMCTS class that batches neural network calls across multiple trees
- Virtual loss for parallel MCTS
- Collect leaves, batch evaluate, distribute results"

AGENT 6 (test-automator): "Create tests for MCTS implementation
- Test MCTSNode expansion and backup
- Test MCTS search produces valid distributions
- Test MCTSTrainer runs without errors
- Compare MCTS agent vs random agent (should win >90%)"

AGENT 7 (python-pro): "Create scripts/train_mcts.py training script
- Load config from configs/h100_mcts.yaml
- Initialize MCTSTrainer
- Run training loop with checkpointing
- Add wandb logging support"
```

### Dependency Graph
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Agent 1    │  │  Agent 2    │  │  Agent 3    │  │  Agent 4    │
│  Configs    │  │  node.py    │  │  search.py  │  │  trainer.py │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       │                └────────┬───────┘                │
       │                         │                        │
       │                         ▼                        │
       │                ┌─────────────┐                   │
       │                │  Agent 5    │◀──────────────────┘
       │                │ batch_mcts  │
       │                └──────┬──────┘
       │                       │
       ▼                       ▼
┌─────────────────────────────────────┐
│            Agent 6 & 7              │
│     Tests + Training Script         │
└─────────────────────────────────────┘
```

---

## Phase 1: Scale Up Network (Easy)

### Goal
Increase model capacity to learn more complex strategies.

### Current vs Target

| Parameter | Current | Target | Rationale |
|-----------|---------|--------|-----------|
| `hidden_dim` | 128 | 256 | 4x parameters, still tiny for H100 |
| `num_layers` | 3 | 4 | Deeper reasoning |
| `num_heads` | 4 | 8 | More attention patterns |
| `minibatch_size` | 256 | 512 | Better gradient estimates |

### Tasks

```
[ ] Task 1.1: Create new config file
    File: configs/h100_scaled.yaml
    - Copy h200_optimized.yaml
    - Update network_config with new values
    - Increase minibatch_size to 512

[ ] Task 1.2: Verify network builds
    - Load new config
    - Print parameter count (target: ~12-15M params vs current 3.3M)
    - Verify forward pass works

[ ] Task 1.3: Test training iteration
    - Run 5 iterations with new config
    - Verify no OOM errors
    - Check iteration time (should be similar, GPU was underutilized)
```

### Config Changes
```yaml
# configs/h100_scaled.yaml
network_config:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.1
  species_embedding_dim: 64  # up from 32

ppo_config:
  minibatch_size: 512
  # rest unchanged
```

---

## Phase 2: Fill GPU - Parallel Environments (Easy)

### Goal
Generate more games per iteration to maximize GPU batch processing.

### Current vs Target

| Parameter | Current | Target | Rationale |
|-----------|---------|--------|-----------|
| `games_per_iteration` | 512 | 2048 | 4x more diverse samples |
| `num_workers` | 16 | 32 | More parallel game simulation |
| `buffer_size` | 100,000 | 400,000 | Hold more transitions |

### Tasks

```
[ ] Task 2.1: Update config for higher throughput
    File: configs/h100_scaled.yaml (continue from Phase 1)
    - games_per_iteration: 2048
    - num_workers: 32
    - buffer_size: 400000
    - min_buffer_size: 20000

[ ] Task 2.2: Profile memory usage
    - Monitor GPU memory during training
    - Monitor CPU memory (replay buffer)
    - Adjust if hitting limits

[ ] Task 2.3: Optimize Cython game generation
    File: _01_simulator/_cython/_cython_core.pyx
    - Ensure batch operations use all CPU cores
    - Profile with 2048 games per batch
```

---

## Phase 3: MCTS + Neural Network (AlphaZero-style)

### Goal
Replace pure PPO with MCTS-guided policy improvement. The network guides MCTS search, MCTS results train the network.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AlphaZero-Style Training                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SELF-PLAY (with MCTS):                                     │
│  ┌─────────────┐                                            │
│  │ Game State  │                                            │
│  └──────┬──────┘                                            │
│         ▼                                                   │
│  ┌─────────────┐     ┌─────────────┐                        │
│  │   Neural    │────▶│    MCTS     │  Run 100-800 sims      │
│  │   Network   │◀────│   Search    │  per move              │
│  └─────────────┘     └──────┬──────┘                        │
│    (prior P, value V)       │                               │
│                             ▼                               │
│                      ┌─────────────┐                        │
│                      │ Improved π  │  Visit counts → policy │
│                      └──────┬──────┘                        │
│                             │                               │
│                             ▼                               │
│                      Sample action from π                   │
│                                                             │
│  TRAINING:                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ Loss = (V - z)² + CrossEntropy(P, π) - H(P) │            │
│  │        value      policy              entropy│            │
│  │        target=game result  target=MCTS π    │            │
│  └─────────────────────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why This Is Better

| Aspect | PPO | MCTS+NN |
|--------|-----|---------|
| **Learns from** | Win/loss only | Every position's best move |
| **Search** | None (reactive) | Looks ahead N moves |
| **Sample efficiency** | Low | 10-100x better |
| **Training signal** | Sparse | Dense (every move) |

### Tasks

```
[ ] Task 3.1: Create MCTS Node structure
    File: _02_agents/mcts/node.py

    class MCTSNode:
        state: GameState
        parent: Optional[MCTSNode]
        children: Dict[Action, MCTSNode]
        visit_count: int
        value_sum: float
        prior: float  # from neural network

    Methods:
        - ucb_score(parent_visits, c_puct) -> float
        - expand(policy_priors: Dict[Action, float])
        - backup(value: float)
        - select_child() -> MCTSNode

[ ] Task 3.2: Create MCTS Search
    File: _02_agents/mcts/search.py

    class MCTS:
        def __init__(self, network, num_simulations=100, c_puct=1.5):
            ...

        def search(self, state: GameState) -> Dict[Action, float]:
            """Run MCTS and return visit count distribution."""
            root = MCTSNode(state)

            for _ in range(num_simulations):
                node = root

                # SELECT - walk down tree using UCB
                while node.is_expanded and not node.is_terminal:
                    node = node.select_child()

                # EXPAND - use network to get priors
                if not node.is_terminal:
                    policy, value = self.network(node.state)
                    node.expand(policy)
                else:
                    value = game_result(node.state)

                # BACKUP - propagate value up tree
                node.backup(value)

            # Return visit count distribution
            return {a: c.visit_count / root.visit_count
                    for a, c in root.children.items()}

    Key parameters:
        - num_simulations: 100-800 (more = stronger but slower)
        - c_puct: exploration constant (1.0-2.5)
        - temperature: for action selection (1.0 early, 0.1 late)

[ ] Task 3.3: Create MCTS-based self-play
    File: _03_training/mcts_self_play.py

    def generate_mcts_games(
        network: BeastyBarNetwork,
        num_games: int,
        num_simulations: int = 100,
        temperature: float = 1.0,
    ) -> List[MCTSTrajectory]:
        """Generate games using MCTS for move selection."""

        trajectories = []
        for _ in range(num_games):
            game = initial_state()
            trajectory = []

            while not is_terminal(game):
                # Run MCTS search
                mcts = MCTS(network, num_simulations)
                policy = mcts.search(game)

                # Sample action (with temperature)
                action = sample_action(policy, temperature)

                # Store training data
                trajectory.append(MCTSTransition(
                    state=game,
                    mcts_policy=policy,  # TARGET for policy training
                    # value filled in after game ends
                ))

                game = step(game, action)

            # Assign values based on game outcome
            result = get_result(game)
            for t in trajectory:
                t.value = result if t.player == winner else -result

            trajectories.append(trajectory)

        return trajectories

[ ] Task 3.4: Create MCTS Trainer
    File: _03_training/mcts_trainer.py

    class MCTSTrainer:
        """AlphaZero-style training loop."""

        def train_iteration(self):
            # 1. Generate games with MCTS
            trajectories = generate_mcts_games(
                self.network,
                num_games=self.config.games_per_iteration,
                num_simulations=self.config.mcts_simulations,
            )

            # 2. Train network
            for batch in iterate_batches(trajectories):
                policy_logits, values = self.network(batch.states)

                # Policy loss: match MCTS visit distribution
                policy_loss = cross_entropy(policy_logits, batch.mcts_policies)

                # Value loss: predict game outcome
                value_loss = mse(values, batch.outcomes)

                # Entropy bonus (optional, for exploration)
                entropy = -sum(p * log(p) for p in softmax(policy_logits))

                loss = policy_loss + value_loss - 0.01 * entropy
                loss.backward()
                self.optimizer.step()

[ ] Task 3.5: Optimize MCTS for speed
    File: _02_agents/mcts/batch_mcts.py

    Key optimizations:
    - Batch neural network evaluations (evaluate multiple leaves at once)
    - Tree reuse between moves
    - Virtual loss for parallel search
    - Cython implementation of tree operations

    class BatchMCTS:
        """MCTS with batched neural network calls."""

        def search_batch(self, states: List[GameState]) -> List[Dict[Action, float]]:
            """Run MCTS on multiple game states in parallel."""
            # Collect leaves from all trees
            # Batch evaluate with neural network
            # Distribute results back to trees

[ ] Task 3.6: Create MCTS config
    File: configs/h100_mcts.yaml

    # MCTS parameters
    mcts_config:
      num_simulations: 200      # sims per move
      c_puct: 1.5               # exploration constant
      temperature: 1.0          # action sampling temp
      temperature_drop_move: 15 # drop temp after move 15
      dirichlet_alpha: 0.3      # root noise for exploration
      dirichlet_epsilon: 0.25   # noise weight

    # Training (adjusted for MCTS)
    games_per_iteration: 256    # fewer games (MCTS is slower)
    total_iterations: 200       # fewer iters needed

    # Network (same as scaled)
    network_config:
      hidden_dim: 256
      num_heads: 8
      num_layers: 4

[ ] Task 3.7: Add MCTS evaluation agent
    File: _02_agents/mcts/agent.py

    class MCTSAgent(Agent):
        """Agent that uses MCTS for move selection."""

        def __init__(self, network, num_simulations=400):
            self.mcts = MCTS(network, num_simulations)

        def select_action(self, state, legal_actions):
            policy = self.mcts.search(state)
            return max(policy.keys(), key=lambda a: policy[a])

[ ] Task 3.8: Integration tests
    - Test MCTS produces valid move distributions
    - Test training loop runs without errors
    - Compare MCTS agent vs PPO agent (MCTS should be stronger with same network)
```

---

## Implementation Order

### Quick Wins (Do First)
```
Phase 1 (1-2 hours): Scale network
Phase 2 (1 hour): Fill GPU
→ Run overnight with new config
```

### Major Improvement (Do Second)
```
Phase 3 (4-6 hours): MCTS implementation
Tasks 3.1-3.4: Core MCTS (~3 hours)
Tasks 3.5-3.6: Optimization (~2 hours)
Tasks 3.7-3.8: Integration (~1 hour)
```

---

## Baseline Results (2026-01-07)

Training run on H100 80GB with `h200_optimized.yaml` (stopped @ iter 135/500):

| Metric | Value |
|--------|-------|
| **Heuristic win rate** | 32% |
| **Random win rate** | 75% |
| Iterations completed | 135 |
| Training time | ~25 min |
| Games played | 69,632 |
| Best checkpoint | `checkpoints/h200_optimized_v1/best_model.pt` |

### Checkpoints Available
```
/workspace/beastybar/checkpoints/h200_optimized_v1/
├── best_model.pt      # Best by heuristic (iter 100)
├── iter_000125.pt     # Latest
├── iter_000100.pt
├── iter_000075.pt
├── iter_000050.pt
└── iter_000025.pt
```

### Download Command
```bash
scp -P 17787 -i ~/.ssh/id_ed25519 root@213.181.122.175:/workspace/beastybar/checkpoints/h200_optimized_v1/best_model.pt ./
```

---

## Expected Results (After Optimization)

| Configuration | Heuristic Win Rate | Training Time |
|--------------|-------------------|---------------|
| Baseline (PPO, small) | 32% @ 135 iter | 25 min |
| Scaled (PPO, large) | ~40% @ 500 iter | 100 min |
| MCTS + NN | ~50%+ @ 200 iter | 120 min |

---

## Files to Create/Modify

### New Files
```
configs/h100_scaled.yaml          # Phase 1-2 config
configs/h100_mcts.yaml            # Phase 3 config
_02_agents/mcts/__init__.py       # MCTS module
_02_agents/mcts/node.py           # MCTS tree node
_02_agents/mcts/search.py         # MCTS search algorithm
_02_agents/mcts/batch_mcts.py     # Batched MCTS for speed
_02_agents/mcts/agent.py          # MCTS-based agent
_03_training/mcts_self_play.py    # MCTS game generation
_03_training/mcts_trainer.py      # AlphaZero-style trainer
scripts/train_mcts.py             # MCTS training script
```

### Modified Files
```
_02_agents/__init__.py            # Export MCTS agent
_03_training/__init__.py          # Export MCTS trainer
```

---

## Subagent Prompts

### Phase 1-2: Scaled Config
```
Create configs/h100_scaled.yaml based on configs/h200_optimized.yaml with:
- hidden_dim: 256, num_heads: 8, num_layers: 4
- species_embedding_dim: 64
- minibatch_size: 512
- games_per_iteration: 2048
- num_workers: 32
- buffer_size: 400000
- min_buffer_size: 20000
Test that the config loads and training runs for 5 iterations.
```

### Phase 3: MCTS Core
```
Implement MCTS for BeastyBar game:
1. Create _02_agents/mcts/node.py with MCTSNode class (UCB selection, expansion, backup)
2. Create _02_agents/mcts/search.py with MCTS class (search loop, neural network integration)
3. Create _02_agents/mcts/agent.py with MCTSAgent class
Use existing engine.py for game logic and BeastyBarNetwork for policy/value predictions.
Include type hints and docstrings.
```

### Phase 3: MCTS Training
```
Implement AlphaZero-style training:
1. Create _03_training/mcts_self_play.py - generate games using MCTS
2. Create _03_training/mcts_trainer.py - train network on MCTS data
3. Create scripts/train_mcts.py - training script with config loading
4. Create configs/h100_mcts.yaml - MCTS training configuration
Loss = MSE(value, outcome) + CrossEntropy(policy, mcts_visits) - entropy_bonus
```

---

## References

- [AlphaZero paper](https://arxiv.org/abs/1712.01815) - DeepMind's approach
- [MuZero paper](https://arxiv.org/abs/1911.08265) - Learned dynamics model
- [EfficientZero](https://arxiv.org/abs/2111.00210) - Sample-efficient variant
