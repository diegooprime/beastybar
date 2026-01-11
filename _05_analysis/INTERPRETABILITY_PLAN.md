# Interpretability Plan: From 2.6% to 40%+ Win Rate

**Problem:** DistilledStrategy achieves only 2.6% win rate against the PPO model it was extracted from.

**Goal:** Create interpretable heuristics that achieve 40%+ win rate against PPO.

---

## Root Cause Analysis: Why 2.6% is So Low

### Problem 1: We Evaluated Cards, Not Actions

The PPO model outputs **124 action logits**, not 12 card preferences:
- 4 hand indices × (1 no-param + 5 single-param + 25 two-param) = 124 actions

DistilledStrategy scores **cards** (species):
```python
BASE_VALUES = {"kangaroo": 0.107, "parrot": 0.026, ...}
```

But the model evaluates:
- "Play Parrot targeting position 0" vs "Play Parrot targeting position 3"
- "Play Kangaroo with hop=1" vs "Play Kangaroo with hop=2"
- "Play Chameleon copying Crocodile's ability" vs "copying Giraffe's ability"

**Impact:** We ignored action parameters entirely. Parrot's value depends 100% on *which card it removes*. Our rules can't express this.

### Problem 2: No Forward Simulation

The existing `HeuristicAgent` does one-step lookahead:
```python
next_state = engine.step(game_state, action)
base_score = self._evaluator(next_state, player)
```

`DistilledStrategy` does **zero lookahead**:
```python
base_value = self.BASE_VALUES.get(species, 0.0)
return base_value * timing_multiplier * threat_multiplier  # Never simulates!
```

The PPO model implicitly learns multi-turn consequences through its value head. Playing Giraffe at turn 1 sets up 10+ turns of recurring advancement. We captured none of this.

### Problem 3: Linear Rules Can't Capture Non-Linear Interactions

The neural network uses:
- Transformer attention across queue positions (cross-card reasoning)
- 3 fusion blocks with residual connections (non-linear feature combination)
- Learned representations that encode complex interactions

Our distilled rules are linear:
```python
score = (base_value × timing × threat) + position_bonus + threat_assessment
```

This cannot express:
- "Crocodile at position 3 threatens MY Giraffe at position 2" (cross-position)
- "Seal is valuable IF my cards are stuck in back positions" (conditional)
- "Monkey combo exists AND would remove their Crocodile" (logical AND)

### Problem 4: Decision Tree Predicted Species, Not Actions

The 46% fidelity decision tree predicts **which species to play**, not which action:
```python
species_label = action_to_simplified_label(action_idx, ...)  # Returns "parrot", not "parrot(target=2)"
```

With only 12 output classes, the tree cannot learn action-specific nuances even at 100% fidelity.

### Problem 5: Value Probing is Correlational, Not Causal

"Parrot in hand → +0.026 value" is a correlation. It doesn't tell us:
- When to play Parrot (timing)
- Which card to target (parameters)
- What outcome we're trying to achieve (consequences)

The value exists because Parrot *can* remove threats. But value only materializes with correct targeting.

---

## Solution: Action-Level Distillation with Simulation

### Technique 1: Outcome-Based Action Scoring (Priority: HIGHEST)

**Key insight:** Don't score cards. Score the *results* of playing actions.

```python
def score_action(state, action, player):
    # Simulate the action
    next_state = engine.step(state, action)

    # Score the outcome
    my_points_gained = bar_points(next_state, player) - bar_points(state, player)
    opp_points_lost = thats_it_points(next_state, 1-player) - thats_it_points(state, 1-player)
    position_improvement = queue_value(next_state, player) - queue_value(state, player)
    threats_removed = threat_count(state, player) - threat_count(next_state, player)

    return (
        my_points_gained * W1 +
        opp_points_lost * W2 +
        position_improvement * W3 +
        threats_removed * W4
    )
```

**Weight extraction:**
1. Collect 50K (state, chosen_action, alternative_actions) from PPO
2. Compute outcome metrics for chosen vs alternatives
3. Use logistic regression to find weights that predict PPO's choices

**Why this helps:** We're now evaluating *what the action does*, not just *what card it uses*.

### Technique 2: Action-Specific Policy Distillation (Priority: HIGH)

Train separate models for each action type:

| Action Type | What to Learn |
|-------------|---------------|
| Parrot(target=i) | Which queue position to target? |
| Kangaroo(hop=n) | How far to hop? |
| Chameleon(copy=i) | Which card to copy? |
| Basic plays | When to play each card? |

**Implementation:**
1. Generate 100K (state, action_probs) tuples from PPO
2. For parameterized actions, train classifiers on:
   - Target card's species, strength, points, owner, position
   - Simulated outcome of targeting that position
   - Relative value vs other targets
3. Use gradient boosting (XGBoost) for interpretable feature importances

**Success metric:** Per-action-type fidelity should reach 60%+

### Technique 3: Attention-Weighted Position Evaluation (Priority: MEDIUM)

From our attention analysis, positions get different focus:
```
Position 0 (front): 53% attention
Position 1: 37% attention
Position 2: 25% attention
Position 3: 27% attention
Position 4 (back): ~10% attention
```

Use these as evaluation weights:
```python
POSITION_WEIGHTS = [0.53, 0.37, 0.25, 0.27, 0.10]

def queue_value(state, player):
    value = 0
    for i, card in enumerate(state.zones.queue):
        weight = POSITION_WEIGHTS[i]
        if card.owner == player:
            value += card.points * weight
        else:
            value -= card.points * weight * 0.8
    return value
```

### Technique 4: Species-Pair Interaction Rules (Priority: MEDIUM)

Mine interaction bonuses from embedding similarities and behavioral data:

| Pair | Embedding Similarity | Observed Interaction |
|------|---------------------|---------------------|
| Hippo-Zebra | -0.292 | Zebra blocks Hippo advancement |
| Crocodile-Zebra | -0.203 | Zebra blocks Crocodile eating |
| Monkey-Monkey | N/A | Pair triggers predator removal |

**Extract as rules:**
```python
if opponent_has("zebra") in queue and card.species == "hippo":
    score -= 0.3  # Blocked advancement
if opponent_has("monkey") in queue and card.species == "monkey":
    score += 0.5  # Combo trigger
```

### Technique 5: Activation Patching for Causal Discovery (Priority: MEDIUM)

Current analysis is correlational. Find what *causes* decisions:

1. Find state pairs where PPO makes different decisions
2. Patch specific features from state A → B
3. Measure if decision changes
4. Features that flip decisions are causal

```python
def find_causal_features(model, state_a, state_b):
    logits_b_original = model(obs_b)

    for feature_group in FEATURE_GROUPS:
        obs_b_patched = obs_b.clone()
        obs_b_patched[feature_group] = obs_a[feature_group]
        logits_b_patched = model(obs_b_patched)

        if decision_changed(logits_b_original, logits_b_patched):
            print(f"Causal feature: {feature_group}")
```

### Technique 6: Multi-Step Minimax (Priority: LOW)

If simpler methods fail, add depth:

```python
def minimax_value(state, player, depth=2):
    if depth == 0 or is_terminal(state):
        return outcome_eval(state, player)

    if state.active_player == player:
        return max(minimax_value(step(state, a), player, depth-1)
                   for a in legal_actions(state))
    else:
        return min(minimax_value(step(state, a), player, depth-1)
                   for a in legal_actions(state))
```

---

## Implementation Plan

### Week 1: Outcome-Based Heuristics

1. **Create `_02_agents/outcome_heuristic.py`**
   - Implement outcome scoring with forward simulation
   - Start with hand-tuned weights
   - Test against random (should hit 90%+ immediately)

2. **Extract weights from PPO behavior**
   - Collect 50K decision samples
   - Run logistic regression on outcome differences
   - Validate weights on held-out set

3. **Benchmark:** Should reach 15-25% vs PPO (6-10x improvement)

### Week 2: Action-Specific Modeling

1. **Create `_05_analysis/action_distillation.py`**
   - Generate action-level training data
   - Train per-action-type classifiers
   - Measure per-action fidelity

2. **Integrate into heuristic**
   - Parrot: Learn target selection
   - Kangaroo: Learn hop distance
   - Chameleon: Learn copy selection

3. **Benchmark:** Target 30-35% vs PPO

### Week 3: Interaction Rules + Attention Weights

1. **Mine species-pair interactions**
   - Compare policy when pairs present vs absent
   - Extract interaction bonuses/penalties

2. **Add attention-weighted position scoring**
   - Quick implementation from existing analysis

3. **Benchmark:** Target 35-40% vs PPO

### Week 4: Causal Probing + Refinement

1. **Activation patching experiments**
   - Identify truly causal features
   - Refine heuristics based on causal insights

2. **Final tuning**
   - Combine all techniques
   - Hyperparameter optimization

3. **Benchmark:** Target 40%+ vs PPO

---

## Success Metrics

| Metric | Current | Week 1 | Week 2 | Week 4 Target |
|--------|---------|--------|--------|---------------|
| Win Rate vs PPO | 2.6% | 20% | 35% | 40%+ |
| Point Margin | -7.55 | -4.0 | -2.0 | -1.0 |
| Action Fidelity | ~10%* | 30% | 45% | 55% |

*Current: species fidelity is 46%, but action fidelity is much lower

---

## Files to Create

| File | Purpose |
|------|---------|
| `_02_agents/outcome_heuristic.py` | Outcome-based scoring with simulation |
| `_05_analysis/action_distillation.py` | Per-action classifier training |
| `_05_analysis/causal_probing.py` | Activation patching experiments |
| `_05_analysis/interaction_mining.py` | Species-pair rule extraction |

---

## Key Insight

**The fundamental error was treating this as a classification problem (which card?) instead of an evaluation problem (which action achieves the best outcome?).**

The PPO model learned to evaluate *consequences*. Our distillation must do the same - simulate actions and score results, not assign static values to cards.
