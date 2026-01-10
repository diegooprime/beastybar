# Opponent Pool Design Brainstorm

## Goal
Design the optimal opponent pool to train the **best Beasty Bar player of all time** - one that beats any opponent: humans, heuristics, other AIs, novel strategies.

## Context

### Current State
- PPO-trained neural network at iteration 434
- Performance: 97% vs random, 79% vs heuristic, ~50% vs human (small sample)
- Training: 7.1M self-play games over 5.7 hours on H200 GPU
- Checkpoint: `checkpoints/ppo_h200_maxperf/final.pt`

### The Problem
Pure self-play is stalling. Evidence:
- Heuristic win rate plateaued at 78-79%
- Hidden information games (like Beasty Bar) are vulnerable to self-play blind spots
- The AI assumes opponents play like itself - humans and heuristics don't
- Poker research (Libratus, Pluribus) shows diverse opponents are necessary for hidden info games

### Beasty Bar Game Properties
- 2-player card game with hidden hands
- 12 animal species with unique abilities
- Queue management + timing + bluffing elements
- Stochastic (card draws)
- ~20-30 moves per game typical

## Codebase Structure
```
_01_simulator/       # Game engine, rules, state
_02_agents/          # Agent implementations
  - random.py        # RandomAgent
  - heuristic.py     # HeuristicAgent (rule-based)
  - neural/          # NeuralAgent (network wrapper)
  - mcts/            # MCTSAgent (tree search)
_03_training/        # Training infrastructure
  - ppo_trainer.py   # Main PPO training loop
  - self_play.py     # Game generation
  - opponent_pool.py # Current opponent pool implementation
  - evaluation.py    # Win rate evaluation
```

## Current Opponent Pool (in opponent_pool.py)
```python
class OpponentType(Enum):
    CURRENT = "current"      # Current network (self-play)
    CHECKPOINT = "checkpoint" # Old checkpoint
    RANDOM = "random"        # Random legal moves
    HEURISTIC = "heuristic"  # Rule-based agent
```

Sampling weights are configurable but currently favor self-play heavily.

---

## Your Task

Think deeply about opponent pool design with the goal of creating an unbeatable Beasty Bar AI. Consider:

### 1. Opponent Types to Create
- What kinds of opponents would expose blind spots?
- Should we create specialized "exploiter" agents?
- What about agents with specific playstyles (aggressive, defensive, species-specialist)?
- How do we create opponents that play DIFFERENTLY from the main agent?

### 2. Opponent Creation Methods
- Train independent agents from scratch with different hyperparameters?
- Use behavioral cloning from human games?
- Procedurally generate rule-based agents with varied heuristics?
- Use MCTS with different exploration parameters?
- Fine-tune copies of the main agent on specific objectives?

### 3. Opponent Pool Composition
- How many distinct opponents?
- What sampling distribution? (uniform? curriculum? adaptive?)
- Should weak opponents (random) be included or are they wasted compute?
- How often to refresh the pool with new opponents?

### 4. Population-Based Approaches
- Should we train multiple independent agents simultaneously?
- How would they interact? Round-robin? ELO-based matching?
- What's the compute tradeoff?

### 5. Exploit-and-Patch Cycle
- Train an "exploiter" to find weaknesses in the main agent
- Retrain main agent against the exploiter
- How to automate this cycle?

### 6. Implementation Considerations
- What changes to `opponent_pool.py` and `ppo_trainer.py`?
- How to manage multiple checkpoints efficiently?
- Memory/compute constraints on H200 (143GB VRAM)?

---

## Deliverable
Produce a concrete plan:
1. List of opponent types to implement (with creation method for each)
2. Recommended pool composition and sampling strategy
3. Implementation approach (what files to modify, new code needed)
4. Training schedule (how to phase in new opponents)

Think step by step. Use the sequential thinking tool for complex reasoning. Read the existing opponent_pool.py and self_play.py to understand current implementation before proposing changes.
