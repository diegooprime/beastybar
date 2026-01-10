# Beasty Bar AI - Technical Reference

## Environment

- **Python**: >=3.10 via `pyproject.toml`
- **Build**: setuptools (editable install supported)
- **Package manager**: uv (recommended) or pip
- **Runtime deps**: fastapi, uvicorn, numpy, anthropic
- **Optional deps**: torch (training), cython (acceleration)
- **Dev deps**: pytest, ruff, mypy, pre-commit

## Module Architecture

### _01_simulator - Game Engine

Deterministic, side-effect-free game simulator with complete Beasty Bar rules.

#### state.py
- `Card` - Frozen dataclass: species, owner, entered_turn
- `PlayerState` - Player's deck and hand (frozen)
- `Zones` - Shared zones: queue, beasty_bar, thats_it (frozen)
- `State` - Complete game snapshot (frozen)
- `initial_state(seed)` - Creates reproducible starting state
- Mutation helpers return new instances: `draw_card()`, `remove_hand_card()`, `append_queue()`, etc.

#### engine.py
- `legal_actions(state, player)` - Enumerate all valid moves
- `step(state, action)` - Advance game by one action
- `step_with_trace(state, action)` - Step with event trace for UI
- `is_terminal(state)` - Check game end
- `score(state)` - Final point totals

Turn resolution order:
1. Play card at back of queue
2. Execute on-play effect
3. Process recurring effects (Hippo, Crocodile, Giraffe)
4. Five-animal check (front 2 enter bar, last 1 bounced)
5. Draw replacement card

#### cards.py
- `resolve_play(state, card)` - Execute species-specific on-play effect
- `process_recurring(state)` - Apply recurring effects in order
- Handler dispatch table for 12 species

Species mechanics:
- **Lion**: Scares monkeys to THAT'S IT, moves to front (or to THAT'S IT if duplicate lion exists)
- **Hippo** (recurring): Passes weaker animals toward gate, blocked by zebra
- **Crocodile** (recurring): Eats weaker animals ahead, blocked by zebra
- **Snake**: Sorts queue by strength descending (stable sort)
- **Giraffe** (recurring): Jumps over one weaker animal per turn
- **Zebra** (permanent): Blocks hippo/croc for itself and all animals in front
- **Seal**: Reverses entire queue order
- **Chameleon**: Temporarily clones another species' effect
- **Monkey**: If 2+ monkeys, bounces hippos/crocs, monkeys move to front
- **Kangaroo**: Jumps over 1-2 animals at the back
- **Parrot**: Bounces any one animal to THAT'S IT
- **Skunk**: Expels highest and second-highest strength animals

#### action_space.py
- `ACTION_DIM = 124` - Fixed action catalog for neural networks
- `action_index(action)` - Convert Action to integer [0, 123]
- `index_to_action(index, legal_actions)` - Convert back
- `legal_action_mask(state, player)` - 124-dim binary mask

Actions organized as: hand_index (4) x max_params (31) = 124 total

#### observations.py
- `OBSERVATION_DIM = 988` - Fixed input size for neural networks
- `state_to_tensor(state, perspective)` - Encode state as numpy array

Observation structure:
- Queue: 5 slots x 17 features each
- Beasty Bar: 24 slots x features
- That's It: 24 slots x features
- Own hand: 4 cards x 17 features (full visibility)
- Opponent hand: 4 slots x 3 features (masked)
- Scalars: turn, active_player, hand counts, deck counts (7 dims)

#### _cython/ - Optional Acceleration
- `_cython_core.pyx` - GIL-free parallel game simulation
- `setup.py` - Build configuration with OpenMP support
- Auto-detected at import; pure Python fallback if unavailable
- ~200x speedup for batch game generation

### _02_agents - AI Players

#### random_agent.py
- `random_agent(state, legal_actions)` - Uniform random selection
- Baseline for calibration

#### heuristic.py
- `MaterialEvaluator` - Position evaluation with configurable weights
- `HeuristicAgent` - Greedy action selection maximizing evaluation
- `create_heuristic_variants()` - Factory for parameter sweeps

Weights:
- `bar_weight=2.0` - Cards in Beasty Bar
- `queue_front_weight=1.1` - Front of queue
- `queue_back_weight=0.3` - Back of queue
- `thats_it_weight=-0.5` - Penalty for bounced cards
- `hand_weight=0.1` - Cards in hand

Optional: 1-ply and 2-ply lookahead, species-specific weighting

#### mcts/ - Monte Carlo Tree Search

**search.py**:
- `MCTSNode` - Search tree node with PUCT scoring
- `MCTS` - Tree search with neural network guidance
- PUCT formula: `Q + c_puct * P * sqrt(N_parent) / (1 + N)`
- Dirichlet noise at root for exploration

**batch_mcts.py**:
- `BatchMCTS` - Batched evaluation across parallel search trees
- Virtual loss for diverse exploration in parallel selection
- 5-10x speedup over sequential MCTS

**agent.py**:
- `MCTSAgent` - Full agent wrapper for game playing
- Temperature control, policy extraction, deterministic mode

#### neural/ - Neural Network Agent

**network.py** - `BeastyBarNetwork`:
- Transformer-based policy-value network (17M parameters)
- `CardEncoder` - Species embeddings + feature projection
- `TransformerEncoder` - For ordered queue (8 heads, 4 layers)
- `SetTransformer` - For unordered zones (beasty_bar, thats_it)
- `FusionLayer` - Combines zone encodings
- `PolicyHead` - 124-dim action logits
- `ValueHead` - Scalar value in [-1, 1]

**agent.py** - `NeuralAgent`:
- Inference modes: greedy, stochastic, temperature
- Batch inference for parallel games
- Device auto-detection (CUDA > MPS > CPU)

### _03_training - Training Infrastructure

#### trainer.py
- `TrainingConfig` - Complete hyperparameter configuration
- `Trainer` - PPO training orchestrator with opponent diversity
- Learning rate scheduling (warmup + cosine decay)
- Full checkpoint/resumption support

#### mcts_trainer.py
- `MCTSTrainerConfig` - MCTS-specific configuration
- `MCTSTrainer` - AlphaZero-style training loop
- Policy target = MCTS visit distribution (not single action)
- Value target = game outcome
- Loss = CrossEntropy(policy) + MSE(value) - entropy_bonus

#### game_generator.py
- `GameGenerator` - Self-play with opponent pool sampling
- Returns transitions, trajectories, opponent info, win rates

#### opponent_pool.py
- `OpponentType` enum: CURRENT, CHECKPOINT, RANDOM, HEURISTIC
- `OpponentPool` - Maintains diverse opponent mix
- Default: 60% current, 20% checkpoints, 10% random, 10% heuristic
- Critical for preventing self-play collapse

#### ppo.py
- `PPOConfig` - Algorithm hyperparameters
- `ppo_update()` - Single PPO step
- GAE with λ=0.95, clipped surrogate objective

#### evaluation.py
- `evaluate_agent()` - Win rate vs opponents
- `compare_agents()` - Head-to-head comparison
- `wilson_confidence_interval()` - Statistical confidence bounds
- `estimate_elo()` - Elo rating estimation

#### checkpoint_manager.py
- `CheckpointManager` - Save/load model checkpoints
- Handles network state, config, iteration counter
- Auto-cleanup of old checkpoints

### _04_ui - Web Interface

#### app.py
- FastAPI REST API for game management
- Endpoints: `/new_game`, `/legal_moves`, `/play`, `/history`
- AI opponent selection (random, heuristic, neural, MCTS)
- Rate limiting (60 req/min per client)
- WebSocket support for real-time updates

#### static/
- HTML/JS frontend for human play
- Card rendering, queue visualization
- Turn history with event traces

### _05_other - Tests & Docs

#### tests/
Comprehensive test suite:
- `test_engine.py` - Turn resolution, rule enforcement
- `test_cards.py` - Species-specific mechanics
- `test_state.py` - Immutability, mutations
- `test_observations.py` - Observation encoding
- `test_action_space.py` - Action catalog
- `test_mcts_*.py` - MCTS correctness
- `test_neural/` - Neural network tests
- `test_opponent_pool*.py` - Opponent diversity

Run with: `pytest _05_other/tests -ra`

## Configuration

### configs/

Training configuration files:
- `default.yaml` - Balanced PPO config
- `fast.yaml` - Quick testing
- `h100_mcts.yaml` - H100 GPU MCTS training
- `h200_optimized_v2.yaml` - H200 optimization

### Key Configuration Parameters

```yaml
# Network architecture
network_config:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  species_embedding_dim: 64

# MCTS settings
mcts_config:
  num_simulations: 200
  c_puct: 2.0
  batch_size: 32
  virtual_loss: 3.0

# Training
learning_rate: 0.00005
entropy_bonus_weight: 0.05
games_per_iteration: 128
batch_size: 512
epochs_per_iteration: 4
```

## Scripts

- `scripts/train.py` - PPO training
- `scripts/train_mcts.py` - MCTS training
- `scripts/evaluate.py` - Model evaluation
- `scripts/play.py` - Interactive play
- `scripts/benchmark_cython.py` - Cython performance testing
- `scripts/build_cython.sh` - Build Cython extensions

## Performance Characteristics

| Operation | Pure Python | Cython | Speedup |
|-----------|-------------|--------|---------|
| Single game | ~2ms | ~0.01ms | 200x |
| Batch (32 games) | ~64ms | ~0.3ms | 200x |

| MCTS Mode | Games/sec | Notes |
|-----------|-----------|-------|
| Sequential | ~0.03 | Network bottleneck |
| BatchMCTS (batch=8) | ~0.3 | 10x speedup |
| BatchMCTS (batch=32) | ~0.5 | GPU saturation |

## Design Principles

1. **Immutability**: All game state mutations return new objects
2. **Determinism**: Seeds enable reproducible replay
3. **Fixed-size tensors**: 988-dim observation, 124-dim action space
4. **Opponent diversity**: Mixed opponents prevent self-play collapse
5. **Modular agents**: Interchangeable via common interface
