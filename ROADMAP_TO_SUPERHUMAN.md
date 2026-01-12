# Roadmap to Superhuman: Beasty Bar Engine

**Date:** 2026-01-12
**Author:** Senior ML Engineer
**Goal:** Create the absolute best Beasty Bar engine ever made - like Stockfish for chess.

---

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Foundation | **IN PROGRESS** | All code changes complete, training pending |
| Phase 2: MCTS Integration | Not Started | - |
| Phase 3: Architecture V2 | Not Started | - |
| Phase 4: Population Training | Not Started | - |
| Phase 5: MuZero Enhancement | Not Started | - |
| Phase 6: Polish & Deployment | Not Started | - |

### Changelog
- **2026-01-12**: Phase 1 implementation complete
  - Created `configs/superhuman_phase1.yaml` with all Phase 1 settings
  - Added `ResidualBlock` class to `network.py`, upgraded value head (4K → 330K params)
  - Added `OUTCOME_HEURISTIC` opponent type to opponent pool
  - Added MCTS-100 configs with `create_mcts_100_configs()`
  - Added `eval_games_per_opponent` config field (default: 200)
  - Fixed `game_generator.py` to handle outcome heuristic opponents

---

## 1. Reality Check: Is 100% Win Rate Possible?

### Game Properties Analysis

**Beasty Bar is a perfect information, deterministic game with the following properties:**

| Property | Status | Implication |
|----------|--------|-------------|
| Hidden Information | **YES** - opponent's hand and deck are hidden | Cannot compute optimal play without modeling opponent's hidden cards |
| Randomness | **YES** - initial deck shuffle | Different games have different optimal strategies |
| Perfect Play Draws | **LIKELY** - some positions are drawn with optimal play | 100% win rate may be impossible even with perfect play |

### Theoretical Ceiling

**The honest answer: 100% win rate is likely IMPOSSIBLE.**

Here's why:

1. **Hidden Information**: The opponent has 4 cards in hand that you cannot see. You know WHICH species remain in their deck (12 total - 4 in hand), but not which specific 4 they hold. This uncertainty creates situations where no policy can guarantee wins.

2. **First-Mover Advantage/Disadvantage**: Depending on the shuffle, player 0 or player 1 may have an inherent advantage in some games.

3. **Draw Scenarios**: Beasty Bar allows ties (same score). Against a strong opponent making optimal defensive plays, some games will end in draws.

**Realistic Ceiling Estimates:**
- vs Random: **~99-100%** achievable (random makes catastrophic mistakes)
- vs Heuristic: **~95-98%** achievable (still exploitable patterns)
- vs Outcome Heuristic: **~85-90%** maximum (uses forward simulation)
- vs Perfect Play: **~50-55%** (both sides playing optimally = close to 50%)
- vs Itself (Superhuman Engine): **~50%** by definition

### What "Superhuman" Actually Means

Instead of 100% win rate, we define superhuman as:
1. **Beats every existing opponent >90% of the time** (including outcome heuristic)
2. **No exploitable weaknesses** (no opponent can achieve >40% vs it)
3. **Optimal or near-optimal in solved positions**
4. **Scales with inference compute** (MCTS improves with more simulations)

---

## 2. Immediate Actions (Current Run: iter 747/1000)

### Config Changes That CAN Help Now

The current run is healthy but plateauing. Here are immediate interventions:

#### 2.1 Nothing to Change Mid-Run
**DO NOT** modify the config mid-training. The current run is stable and will complete successfully. Breaking a training run mid-flight is almost always worse than letting it finish.

#### 2.2 Post-1000 Quick Wins

When iteration 1000 completes, immediately start a new run with these tweaks:

```yaml
# configs/superhuman_v1.yaml

# CHANGE 1: Slower entropy decay (keep exploring longer)
entropy_schedule: "linear"
entropy_start: 0.04
entropy_end: 0.02  # Was 0.01 - too aggressive

# CHANGE 2: Increase outcome heuristic exposure
opponent_config:
  current_weight: 0.40       # Reduce self-play weight
  checkpoint_weight: 0.15
  random_weight: 0.05
  heuristic_weight: 0.40     # Increase - but add outcome_heuristic variant

# CHANGE 3: Longer training
total_iterations: 2000       # Double the iterations

# CHANGE 4: Enable MCTS opponents for harder training signal
use_mcts_opponents: true
mcts_simulations: 100
mcts_weight: 0.10

# CHANGE 5: Lower temperature at end for sharper play
temperature_end: 0.3        # Was 0.5
```

#### 2.3 Evaluation Enhancement

Current evaluation uses 50 games per opponent. This is too noisy for 60-70% win rates.

```yaml
eval_frequency: 50           # Less frequent but more reliable
eval_games_per_opponent: 200 # Reduce variance significantly
```

---

## 3. Architecture Changes Required

### Current Architecture Analysis

**File:** `/Users/p/Desktop/v/experiments/beastybar/_02_agents/neural/network.py`

```python
# Current network: ~1.3M parameters
NetworkConfig:
  hidden_dim: 256        # Too wide, not deep enough
  num_heads: 8           # Good
  num_layers: 4          # Okay but uniform across zones
  dropout: 0.1           # Good
  species_embedding_dim: 64  # Good
```

**Problems:**
1. **Shared architecture for different zones** - Queue (order-sensitive) and Bar (order-invariant) use same encoder depth. Queue position is CRITICAL and needs more capacity.
2. **Value head too simple** - Single MLP after fusion. Value estimation is the limiting factor against outcome heuristic.
3. **No planning/simulation capability** - Network predicts value directly, but strong play requires looking ahead.

### Proposed Architecture: BeastyBarNetworkV2

```python
# /Users/p/Desktop/v/experiments/beastybar/_02_agents/neural/network_v2.py

class BeastyBarNetworkV2(nn.Module):
    """Enhanced architecture with dedicated planning capacity."""

    def __init__(self, config: NetworkConfigV2):
        super().__init__()

        # CHANGE 1: Asymmetric zone encoders
        # Queue is order-sensitive and critical - give it more capacity
        self.queue_encoder = TransformerZoneEncoder(
            hidden_dim=256,
            num_layers=6,      # 6 layers for queue (was 4)
            num_heads=8,
        )

        # Bar/ThatsIt are order-invariant, less critical
        self.bar_encoder = SetTransformerEncoder(
            hidden_dim=256,
            num_layers=2,      # 2 layers sufficient (was 4)
            num_heads=4,
        )

        # CHANGE 2: Deeper value head with residual connections
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # CHANGE 3: Dueling architecture (separate value and advantage streams)
        # Helps with value estimation stability
        self.advantage_head = nn.Linear(hidden_dim, action_dim)
        self.state_value_head = nn.Linear(hidden_dim, 1)

        # CHANGE 4: Auxiliary prediction heads for multi-task learning
        self.queue_position_predictor = nn.Linear(hidden_dim, 5)  # Predict final queue positions
        self.score_predictor = nn.Linear(hidden_dim, 1)  # Predict final score margin
```

### Key Architectural Insights

| Change | Why | Expected Impact |
|--------|-----|-----------------|
| Deeper queue encoder | Queue position is most predictive of wins | +5-10% vs outcome heuristic |
| Dueling architecture | Separates state value from action advantage | Better value estimation |
| Auxiliary heads | Multi-task learning improves representation | +2-5% overall |
| Residual value head | Deeper value estimation without vanishing gradients | +3-5% vs strong opponents |

### Parameter Budget

Current: ~1.3M parameters
Target: ~3-5M parameters (still inference-fast, but more capacity)

```python
# Proposed config
NetworkConfigV2:
    hidden_dim: 256
    queue_layers: 6
    bar_layers: 2
    thats_it_layers: 2
    hand_layers: 2
    fusion_layers: 4
    num_heads: 8
    dropout: 0.1
    # ~3.5M parameters
```

---

## 4. Training Methodology Overhaul

### Why PPO Self-Play Alone Won't Get Us There

Current approach: **PPO + Self-Play + Opponent Pool**

**Problems:**
1. **No explicit planning** - Network learns reactive patterns, not search
2. **Gradient signal from game outcome only** - Sparse reward at game end
3. **Value function learned jointly** - But value accuracy is critical for MCTS
4. **No iterative improvement of value estimates** - MuZero's key insight

### What AlphaZero/MuZero Did Differently

| Technique | AlphaZero | Current Beasty Bar | Gap |
|-----------|-----------|-------------------|-----|
| MCTS during training | Policy targets from MCTS visit counts | Policy from network only | **CRITICAL** |
| Value targets | Monte Carlo returns | GAE estimates | Medium |
| Policy targets | MCTS visit distribution | Policy gradient | **CRITICAL** |
| Training data | MCTS search trajectories | PPO rollouts | **CRITICAL** |

### The AlphaZero Recipe for Beasty Bar

**Phase 1: Add MCTS to Training Loop**

```python
# /Users/p/Desktop/v/experiments/beastybar/_03_training/alphazero_trainer.py

class AlphaZeroTrainer:
    """Training with MCTS-generated targets."""

    def generate_training_data(self, num_games: int) -> list[TrainingExample]:
        """Generate training data using MCTS search."""
        examples = []

        for _ in range(num_games):
            state = initial_state(seed=random.randint(0, 1_000_000))
            game_examples = []

            while not is_terminal(state):
                # Run MCTS from current position
                mcts = MCTS(self.network, num_simulations=100)
                visit_counts = mcts.search(state, state.active_player)

                # MCTS policy = normalized visit counts
                mcts_policy = self._visit_counts_to_policy(visit_counts)

                # Store example (state, mcts_policy, placeholder_value)
                obs = state_to_tensor(state, state.active_player)
                game_examples.append((obs, mcts_policy, state.active_player))

                # Sample action from MCTS policy
                action = self._sample_from_policy(mcts_policy, temperature=1.0)
                state = step(state, action)

            # Game over - assign values based on outcome
            final_value = self._compute_game_outcome(state)
            for obs, policy, player in game_examples:
                # Value from this player's perspective
                value = final_value if player == 0 else -final_value
                examples.append(TrainingExample(obs, policy, value))

        return examples

    def train_step(self, batch: list[TrainingExample]) -> dict:
        """Train on MCTS-generated targets."""
        obs = torch.stack([e.observation for e in batch])
        target_policy = torch.stack([e.mcts_policy for e in batch])
        target_value = torch.tensor([e.value for e in batch])

        # Forward pass
        pred_policy, pred_value = self.network(obs)

        # Policy loss: cross-entropy with MCTS targets
        policy_loss = F.cross_entropy(pred_policy, target_policy)

        # Value loss: MSE with game outcome
        value_loss = F.mse_loss(pred_value.squeeze(), target_value)

        # No entropy bonus needed - MCTS provides exploration
        total_loss = policy_loss + value_loss

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}
```

**Phase 2: MuZero-style Learned Dynamics (Advanced)**

For ultimate strength, implement a learned dynamics model:

```python
class MuZeroNetwork(nn.Module):
    """MuZero-style network with learned dynamics."""

    def __init__(self, config):
        super().__init__()
        # Representation: obs -> hidden state
        self.representation = RepresentationNetwork(config)

        # Dynamics: (hidden_state, action) -> next_hidden_state, reward
        self.dynamics = DynamicsNetwork(config)

        # Prediction: hidden_state -> (policy, value)
        self.prediction = PredictionNetwork(config)

    def initial_inference(self, observation):
        """From observation to hidden state, policy, value."""
        hidden = self.representation(observation)
        policy, value = self.prediction(hidden)
        return hidden, policy, value

    def recurrent_inference(self, hidden_state, action):
        """Simulate one step without real environment."""
        next_hidden, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(next_hidden)
        return next_hidden, reward, policy, value
```

This allows MCTS to search in learned representation space without environment simulation - faster search at inference time.

---

## 5. Search Integration

### Current MCTS State

**File:** `/Users/p/Desktop/v/experiments/beastybar/_02_agents/mcts/search.py`

The MCTS implementation exists and is correct, but:
1. **Not used during training** - Only for inference/evaluation
2. **No batch MCTS** - Sequential node evaluation is slow
3. **Not integrated with training loop** - PPO ignores MCTS

### Implementation Plan: MCTS-Enhanced Training

**Step 1: Fast Batch MCTS**

```python
# Already exists: /Users/p/Desktop/v/experiments/beastybar/_02_agents/mcts/batch_mcts.py

# Optimize for GPU:
# - Batch 8-16 leaves per forward pass
# - Virtual loss for parallel tree expansion
# - Mixed precision inference (FP16)
```

**Step 2: MCTS Training Integration**

```python
# /Users/p/Desktop/v/experiments/beastybar/_03_training/mcts_ppo.py

class MCTSEnhancedPPO:
    """PPO training with MCTS policy targets."""

    def __init__(self, network, mcts_simulations=100):
        self.network = network
        self.mcts = BatchMCTS(network, num_simulations=mcts_simulations)

    def generate_games(self, num_games: int) -> list[Trajectory]:
        """Generate games using MCTS for policy improvement."""
        trajectories = []

        # Use batch MCTS for efficiency
        states = [initial_state(seed=i) for i in range(num_games)]

        while any(not is_terminal(s) for s in states):
            # Batch MCTS search across all active games
            active_states = [s for s in states if not is_terminal(s)]
            mcts_policies = self.mcts.search_batch(active_states, perspective=0)

            for state, mcts_policy in zip(active_states, mcts_policies):
                # Store (state, mcts_policy) for training
                # Sample action from MCTS policy
                action = sample_from_policy(mcts_policy, temperature=1.0)
                new_state = step(state, action)
                # Update state

        return trajectories
```

**Step 3: Inference-Time Compute Scaling**

At inference time, scale MCTS simulations for stronger play:

| Simulations | Relative Strength | Use Case |
|-------------|------------------|----------|
| 0 (raw network) | Baseline | Fast games |
| 100 | +50 ELO | Training games |
| 400 | +100 ELO | Tournament play |
| 1600 | +150 ELO | Analysis mode |
| 6400 | +200 ELO | Superhuman mode |

**Implementation:**

```python
# /Users/p/Desktop/v/experiments/beastybar/_02_agents/mcts/adaptive_search.py

class AdaptiveMCTS:
    """MCTS with adaptive simulation budget."""

    def search(self, state, perspective, *,
               min_simulations=100,
               max_simulations=1600,
               time_budget_ms=None,
               uncertainty_threshold=0.1):
        """Run MCTS with adaptive stopping."""

        # Start with minimum simulations
        for _ in range(min_simulations):
            self._simulate(root, perspective)

        # Continue until confidence is high or budget exhausted
        while not self._should_stop(root, max_simulations, time_budget_ms):
            self._simulate(root, perspective)

            # Check if top action is clear winner
            if self._action_confidence(root) > uncertainty_threshold:
                break

        return self._visit_count_distribution(root)
```

---

## 6. Data & Curriculum

### What Training Data/Opponents We're Missing

| Missing Element | Why It Matters | How to Fix |
|----------------|----------------|------------|
| **Strong MCTS opponents** | Training against 100-sim MCTS hardens the policy | Add MCTS-100/500 to opponent pool |
| **Exploiter training** | Find and patch weaknesses | Implement Population-Based Training |
| **Edge case games** | Rare situations not seen in self-play | Curriculum with curated positions |
| **Late-game specialist** | Games often decided in final 4 moves | Weighted sampling of late-game positions |

### Population-Based Training (PBT)

The strongest AI systems use population-based training to avoid blind spots:

```python
# /Users/p/Desktop/v/experiments/beastybar/_03_training/population.py

class PopulationTrainer:
    """Maintain a population of diverse agents."""

    def __init__(self, population_size=8):
        self.agents = [create_agent() for _ in range(population_size)]
        self.exploiters = []  # Trained to beat current best

    def train_epoch(self):
        # 1. Self-play within population
        for agent in self.agents:
            opponent = random.choice(self.agents)
            games = generate_games(agent, opponent)
            agent.train(games)

        # 2. Train exploiters against current best
        best = max(self.agents, key=lambda a: a.elo)
        for exploiter in self.exploiters:
            games = generate_games_vs(exploiter, best)
            exploiter.train(games)

        # 3. If exploiter beats best significantly, add to population
        for exploiter in self.exploiters:
            if exploiter.win_rate_vs(best) > 0.6:
                self.agents.append(exploiter.clone())
                self.agents.sort(key=lambda a: a.elo)
                self.agents = self.agents[-population_size:]  # Keep best

        # 4. Spawn new exploiters
        self.exploiters = [ExploiterAgent(target=best) for _ in range(2)]
```

### Curriculum Learning

Current training: random game states from self-play

Better approach: structured curriculum

```python
# /Users/p/Desktop/v/experiments/beastybar/_03_training/curriculum_v2.py

class SupervisedCurriculum:
    """Structured curriculum from easy to hard."""

    phases = [
        # Phase 1: Basic card interactions (iter 0-500)
        {"opponents": ["random"], "position_filter": None},

        # Phase 2: Queue control (iter 500-1000)
        {"opponents": ["random", "heuristic"], "position_filter": "full_queue"},

        # Phase 3: Full game (iter 1000-2000)
        {"opponents": ["heuristic", "aggressive", "defensive"], "position_filter": None},

        # Phase 4: Strong opponents (iter 2000-3000)
        {"opponents": ["outcome_heuristic", "mcts-100"], "position_filter": None},

        # Phase 5: Population-based (iter 3000+)
        {"opponents": ["population"], "position_filter": None},
    ]
```

### Harder Training Signal

The outcome_heuristic ceiling at 60-70% indicates we're not generating hard enough training data.

**Solution: Outcome-Based Opponent Weighting**

```python
# After each evaluation, increase weight of opponents we're losing to

def update_opponent_weights(win_rates: dict[str, float]) -> dict[str, float]:
    """Weight opponents inversely to win rate."""
    weights = {}
    for opp, wr in win_rates.items():
        # Lower win rate = higher weight
        # Transform: 50% -> 1.0x, 70% -> 0.5x, 90% -> 0.1x
        weights[opp] = max(0.1, 1.0 - (wr - 0.5))

    # Normalize
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}
```

---

## 7. Concrete Milestones

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Beat outcome_heuristic 80%+ consistently

- [x] Run 2000-iteration training with enhanced config *(Config created: `configs/superhuman_phase1.yaml` - 2026-01-12)*
- [x] Implement value head improvements (residual blocks) *(Added `ResidualBlock` class, upgraded value head to 330K params - 2026-01-12)*
- [x] Add MCTS-100 to opponent pool *(Added `OUTCOME_HEURISTIC` type + MCTS-100 configs - 2026-01-12)*
- [x] Increase evaluation to 200 games per opponent *(Added `eval_games_per_opponent` config field - 2026-01-12)*
- [ ] **PENDING:** Run training and validate success criteria

**Success Criteria:**
- 98%+ vs random
- 92%+ vs heuristic
- 80%+ vs outcome_heuristic
- 75%+ vs distilled_outcome

### Phase 2: MCTS Integration (Weeks 3-4)
**Goal:** MCTS-enhanced training producing policy improvement

- [ ] Integrate batch MCTS into training loop
- [ ] Train with MCTS policy targets (AlphaZero-style)
- [ ] Implement adaptive MCTS for inference
- [ ] Benchmark: network + MCTS-100 vs network alone

**Success Criteria:**
- MCTS-100 adds 10%+ win rate vs strong opponents
- Training with MCTS targets converges faster
- Value prediction error < 0.2 on held-out positions

### Phase 3: Architecture V2 (Weeks 5-6)
**Goal:** New architecture with improved value estimation

- [ ] Implement BeastyBarNetworkV2 with asymmetric encoders
- [ ] Add dueling architecture
- [ ] Add auxiliary prediction heads
- [ ] Train from scratch with new architecture

**Success Criteria:**
- 85%+ vs outcome_heuristic
- 80%+ vs MCTS-100
- Value prediction error < 0.15

### Phase 4: Population Training (Weeks 7-8)
**Goal:** No exploitable weaknesses

- [ ] Implement population-based training
- [ ] Train exploiter agents against best model
- [ ] Iterative patching of weaknesses
- [ ] Comprehensive test suite for edge cases

**Success Criteria:**
- No opponent achieves >30% vs main agent
- Pass all edge case tests
- Stable performance across 1000+ game evaluation

### Phase 5: MuZero Enhancement (Weeks 9-12)
**Goal:** Learned dynamics for faster search

- [ ] Implement MuZero-style dynamics model
- [ ] Train representation + dynamics + prediction jointly
- [ ] Benchmark learned search vs real environment search
- [ ] Optimize for production deployment

**Success Criteria:**
- Learned MCTS matches real MCTS at 4x speed
- 90%+ vs all non-MCTS opponents
- Sub-100ms per move at competition strength

### Phase 6: Polish & Deployment (Weeks 13-16)
**Goal:** Production-ready superhuman engine

- [ ] Comprehensive evaluation vs all opponent types
- [ ] Optimize inference (quantization, ONNX export)
- [ ] Build analysis/explanation features
- [ ] Documentation and API

**Success Criteria:**
- Consistent 90%+ vs all heuristic opponents
- 70%+ vs MCTS-1000 (without MCTS of its own)
- With MCTS-1600: beats any opponent tested

---

## 8. Estimated Resources

### Compute Requirements

| Phase | GPU Hours | GPU Type | Estimated Cost |
|-------|-----------|----------|----------------|
| Phase 1 | 50 | A100/H100 | $100 |
| Phase 2 | 100 | A100/H100 | $200 |
| Phase 3 | 75 | A100/H100 | $150 |
| Phase 4 | 200 | A100/H100 | $400 |
| Phase 5 | 300 | H100/H200 | $600 |
| Phase 6 | 50 | Any GPU | $100 |
| **Total** | **775** | - | **~$1550** |

### Time Estimate

- **Realistic timeline:** 12-16 weeks
- **Aggressive timeline:** 8-10 weeks (with full-time focus)
- **Minimal viable (beats outcome_heuristic 90%):** 4-6 weeks

### What's Realistic vs Aspirational

| Goal | Feasibility | Confidence |
|------|-------------|------------|
| Beat random 99%+ | **Realistic** | 99% |
| Beat heuristic 95%+ | **Realistic** | 95% |
| Beat outcome_heuristic 90%+ | **Realistic** | 80% |
| Beat MCTS-100 80%+ | **Achievable** | 70% |
| Beat any opponent 90%+ | **Aspirational** | 40% |
| 100% win rate vs anything | **Impossible** | <1% |

---

## 9. Critical Missing Pieces

### What We Don't Have Yet

1. **Solved Endgames Database**
   - With 4-5 cards remaining, many positions are solvable
   - A tablebase would provide perfect play in late game
   - Implementation: retrograde analysis from terminal positions

2. **Opening Book**
   - First 2-3 moves have limited legal actions
   - Pre-computed optimal openings would speed up games
   - Could be generated from MCTS with high simulation counts

3. **Perfect Information Solver**
   - When opponent's hand is known (late game deduction), we can solve perfectly
   - Alpha-beta search with game-specific evaluation
   - Would provide ground truth for value function training

4. **Opponent Modeling**
   - Current approach treats opponent as unknown
   - Better: maintain belief distribution over opponent's hand
   - Even better: model opponent's policy and exploit weaknesses

### Implementation Priority

1. **High Priority:** MCTS training integration, value head improvements
2. **Medium Priority:** Population training, curriculum learning
3. **Lower Priority:** MuZero dynamics, opening book
4. **Future:** Endgame tablebase, opponent modeling

---

## 10. The Honest Summary

**Can we reach 100% win rate?** No. Hidden information makes this mathematically impossible.

**Can we reach 95%+ vs all existing opponents?** Yes, with the changes outlined above.

**Can we create a Stockfish-like dominant engine?** Yes, but it will:
- Be the best through superior search (MCTS) combined with superior evaluation (neural network)
- Scale strength with inference compute (more simulations = stronger play)
- Have no exploitable weaknesses (population-based training patches blind spots)
- Provide perfect or near-perfect play in solved endgame positions

**The path forward:**
1. Fix the value function (it's the bottleneck)
2. Integrate MCTS into training (AlphaZero recipe)
3. Harden with population-based exploiter training
4. Optionally: add MuZero-style learned dynamics for speed

**The current run (iter 747/1000) is doing its job.** Let it finish, collect the data, and then implement Phase 1 of this roadmap. The 60-70% ceiling against outcome_heuristic is not a failure - it's the expected limit of PPO self-play without MCTS-enhanced training.

---

## Appendix A: Key File Locations

| Purpose | File Path |
|---------|-----------|
| Neural Network | `/Users/p/Desktop/v/experiments/beastybar/_02_agents/neural/network.py` |
| MCTS Search | `/Users/p/Desktop/v/experiments/beastybar/_02_agents/mcts/search.py` |
| Batch MCTS | `/Users/p/Desktop/v/experiments/beastybar/_02_agents/mcts/batch_mcts.py` |
| PPO Training | `/Users/p/Desktop/v/experiments/beastybar/_03_training/ppo.py` |
| Trainer | `/Users/p/Desktop/v/experiments/beastybar/_03_training/trainer.py` |
| Evaluation | `/Users/p/Desktop/v/experiments/beastybar/_03_training/evaluation.py` |
| Game Engine | `/Users/p/Desktop/v/experiments/beastybar/_01_simulator/engine.py` |
| Outcome Heuristic | `/Users/p/Desktop/v/experiments/beastybar/_02_agents/outcome_heuristic.py` |
| Current Config | `/Users/p/Desktop/v/experiments/beastybar/configs/iter600_to_1000.yaml` |

## Appendix B: Recommended Next Config

```yaml
# configs/superhuman_phase1.yaml
# Phase 1: Foundation - Target 80%+ vs outcome_heuristic

network_config:
  hidden_dim: 256
  num_heads: 8
  num_layers: 4
  dropout: 0.1
  species_embedding_dim: 64

ppo_config:
  learning_rate: 0.00008      # Slightly lower for stability
  clip_epsilon: 0.2
  value_coef: 1.0             # Increase value learning
  entropy_coef: 0.03
  gamma: 0.99
  gae_lambda: 0.97            # Higher lambda for less bias
  ppo_epochs: 6               # More epochs per batch
  minibatch_size: 4096
  max_grad_norm: 0.5
  normalize_advantages: true
  clip_value: true
  target_kl: 0.015

total_iterations: 2000
games_per_iteration: 16384    # More games per iteration
checkpoint_frequency: 100
eval_frequency: 50

# Slower schedules
entropy_schedule: "cosine"
entropy_start: 0.04
entropy_end: 0.02             # Don't decay as aggressively
temperature_schedule: "cosine"
temperature_start: 1.0
temperature_end: 0.4

# Harder opponent mix
use_opponent_pool: true
use_adaptive_weights: true
min_opponent_weight: 0.08

opponent_config:
  current_weight: 0.35
  checkpoint_weight: 0.15
  random_weight: 0.05
  heuristic_weight: 0.35
  outcome_heuristic_weight: 0.10

# Enable MCTS opponents
use_mcts_opponents: true
mcts_simulations: 100

# Exploit-patch cycle
use_exploit_patch: true
exploit_patch_interval: 300

# Cython required
force_cython: true
async_game_generation: true
async_num_workers: 32

buffer_size: 2000000
seed: 42
device: "cuda"
experiment_name: "superhuman_phase1"

eval_games_per_opponent: 200
eval_opponents:
  - random
  - heuristic
  - aggressive
  - defensive
  - queue
  - skunk
  - noisy
  - online
  - outcome_heuristic
  - distilled_outcome
```

---

*Document generated: 2026-01-12*
*This roadmap is a living document. Update as results come in.*
