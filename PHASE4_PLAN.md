# Phase 4: Population Training - Implementation Plan

**Date:** 2026-01-12

---

## Overview

Population-Based Training (PBT) creates a diverse population of agents that compete and evolve together. Key insight: exploiter agents find weaknesses in the best model, and when they succeed, they're added to the population - iteratively patching blind spots.

---

## Architecture

### Core Components

1. **PopulationTrainer** (`_03_training/population.py`)
   - Maintains population of N agents (default: 8)
   - Orchestrates self-play within population
   - Tracks ELO ratings for each agent
   - Manages exploiter spawning and integration

2. **ExploiterAgent** (`_03_training/exploiter.py`)
   - Specialized training to beat a specific target agent
   - Uses reward shaping: extra reward for beating target
   - Fast iteration cycle (fewer games, more updates)

3. **PopulationConfig** (`configs/population.yaml`)
   - Population size, exploiter count
   - ELO update parameters
   - Exploit-patch thresholds

4. **Edge Case Test Suite** (`tests/test_edge_cases.py`)
   - Known tricky positions
   - Adversarial scenarios
   - Regression tests for patched weaknesses

---

## Implementation Details

### PopulationTrainer Flow

```
1. Initialize population with current best + variants
2. For each epoch:
   a. Self-play tournament within population
   b. Update ELO ratings
   c. Train each agent on games against population
   d. Spawn exploiters against current best
   e. Train exploiters aggressively
   f. If exploiter beats best >60%: add to population
   g. Cull weakest agents to maintain population size
3. Return best agent
```

### Exploiter Training

```
1. Clone current best agent
2. Modify loss: add bonus for winning against target
3. Train with high learning rate for fast adaptation
4. Evaluate every N iterations against target
5. If win rate > threshold: success, add to population
6. If no progress after M iterations: abandon
```

---

## File Structure

```
_03_training/
├── population.py          # PopulationTrainer class
├── exploiter.py           # ExploiterAgent class
├── elo.py                 # ELO rating system
└── population_config.py   # Configuration dataclass

configs/
└── population.yaml        # Default population config

tests/
└── test_edge_cases.py     # Edge case test suite
```

---

## Success Criteria

- [ ] No opponent achieves >30% win rate vs main agent
- [ ] Pass all edge case tests
- [ ] Stable performance across 1000+ game evaluation
- [ ] Integration with existing training infrastructure

---

## Dependencies

- AlphaZero trainer (Phase 2) - ✅ Implemented
- Network architecture (Phase 3) - 🔄 In Progress by other agents
- Evaluation infrastructure - ✅ Exists

---

## Scope Boundaries

**Will modify:**
- `_03_training/` - new files for population training
- `configs/` - new configuration file
- `tests/` - new test file

**Will NOT modify:**
- `_02_agents/neural/network*.py` - Phase 3 scope
- Existing trainer files (will extend, not modify)
