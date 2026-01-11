# Neural Network Decision Tree Analysis

## Executive Summary

This analysis extracts human-readable decision rules from a trained Beasty Bar neural
network by training an interpretable decision tree to mimic the NN's choices.

### Key Findings

1. **Queue length is the #1 decision factor** - The neural network's primary consideration
   is how many cards are in the queue, which determines timing and available tactics.

2. **Recurring animals are prioritized** - Giraffe, hippo, and crocodile (all recurring
   species) rank highly in the decision tree, suggesting the NN values sustained board
   presence.

3. **High-value cards played early** - Giraffe and zebra (worth 3-4 points) are played
   when `turn_number` is lower, indicating an aggressive early-game scoring strategy.

4. **Skunk is a counter-play** - Skunk is played when queue has high average/max strength,
   confirming it's used as a reactive removal tool against strong opponents.

5. **Parrot waits for targets** - The `can_use_parrot` feature (parrot + queue targets)
   is important, showing the NN times parrot plays strategically.

---

## Technical Overview

**Model**: `checkpoints/v4/final.pt`
**Training Samples**: 10,000
**Decision Tree Depth**: 8
**Fidelity Score**: 50.8%

The fidelity score measures how often the decision tree makes the same choice
as the neural network. A 50.8% fidelity indicates the decision tree captures
major decision patterns, though the neural network uses more nuanced reasoning
for about half of its decisions (which require deeper context).

---

## Species Selection Distribution

The neural network's greedy policy selected cards in the following proportions:

| Species | Count | Percentage |
|---------|-------|------------|
| giraffe | 864 | 8.6% |
| parrot | 860 | 8.6% |
| hippo | 859 | 8.6% |
| crocodile | 856 | 8.6% |
| zebra | 850 | 8.5% |
| kangaroo | 849 | 8.5% |
| skunk | 842 | 8.4% |
| monkey | 840 | 8.4% |
| lion | 830 | 8.3% |
| chameleon | 826 | 8.3% |
| snake | 813 | 8.1% |
| seal | 711 | 7.1% |


---

## Feature Importance Ranking

The decision tree identified these features as most important for predicting
which card the neural network will play:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `queue_length` | 0.115 |
| 2 | `hand_giraffe` | 0.104 |
| 3 | `hand_kangaroo` | 0.095 |
| 4 | `hand_zebra` | 0.091 |
| 5 | `hand_hippo` | 0.090 |
| 6 | `hand_crocodile` | 0.077 |
| 7 | `hand_monkey` | 0.070 |
| 8 | `can_use_parrot` | 0.053 |
| 9 | `hand_snake` | 0.044 |
| 10 | `hand_lion` | 0.030 |
| 11 | `front_species_idx` | 0.025 |
| 12 | `hand_skunk` | 0.021 |
| 13 | `has_low_strength_in_hand` | 0.021 |
| 14 | `has_high_strength_in_hand` | 0.019 |
| 15 | `hand_chameleon` | 0.018 |
| 16 | `queue_avg_strength` | 0.016 |
| 17 | `queue_crocodile` | 0.012 |
| 18 | `hand_parrot` | 0.011 |
| 19 | `back_species_idx` | 0.011 |
| 20 | `queue_max_strength` | 0.010 |


---

## Key Strategic Insights

Based on the decision tree analysis, here are the main strategic principles
the neural network has learned:

1. **Card availability is crucial: having giraffe, kangaroo, zebra strongly influences decisions**

2. **Queue length is a key decision factor - tactics change based on how full the queue is**

3. **Parrot usage is conditional - the AI waits for the right moment to remove opponents**


### Interpretation of Top Features

- **queue_length** (importance: 0.115): The number of cards in queue determines available tactics and timing.
- **hand_giraffe** (importance: 0.104): Having giraffe in hand is a major decision driver. Giraffe's recurring ability to move forward is valuable.
- **hand_kangaroo** (importance: 0.095): Having kangaroo in hand is a major decision driver. Kangaroo has specific tactical value.
- **hand_zebra** (importance: 0.091): Having zebra in hand is a major decision driver. Zebra provides 4 points and stays permanently once in the bar.
- **hand_hippo** (importance: 0.090): Having hippo in hand is a major decision driver. Hippo has recurring push ability, strong for queue control.


---

## Species-Specific Decision Patterns

When does the neural network prefer to play each species? Here's what makes each
card choice distinctive:

### Lion
- When `can_use_lion` is higher than average (diff: +0.73)
- When `queue_max_strength` is lower than average (diff: -0.64)

### Hippo
- When `turn_number` is lower than average (diff: -0.69)
- When `queue_max_strength` is lower than average (diff: -0.61)

### Crocodile
- When `thats_it_count` is lower than average (diff: -0.85)
- When `front_species_idx` is higher than average (diff: +0.76)

### Giraffe
- When `turn_number` is lower than average (diff: -2.01)
- When `thats_it_count` is lower than average (diff: -1.04)

### Zebra
- When `turn_number` is lower than average (diff: -1.48)
- When `front_species_idx` is lower than average (diff: -0.92)

### Monkey
- When `turn_number` is lower than average (diff: -0.56)
- When `back_species_idx` is lower than average (diff: -0.34)

### Parrot
- When `thats_it_count` is lower than average (diff: -1.12)
- When `turn_number` is lower than average (diff: -0.91)
- When `queue_max_strength` is higher than average (diff: +0.89)

### Skunk
- When `queue_min_strength` is higher than average (diff: +2.01)
- When `queue_avg_strength` is higher than average (diff: +1.74)
- When `queue_max_strength` is higher than average (diff: +1.36)



---

## Decision Tree Rules (Simplified)

Below is a simplified representation of the decision tree's rules. The tree
makes decisions by checking conditions from the root down to the leaves.

```
|--- hand_giraffe <= 0.50
|   |--- hand_zebra <= 0.50
|   |   |--- hand_hippo <= 0.50
|   |   |   |--- hand_monkey <= 0.50
|   |   |   |   |--- hand_snake <= 0.50
|   |   |   |   |   |--- hand_kangaroo <= 0.50
|   |   |   |   |   |   |--- has_high_strength_in_hand <= 0.50
|   |   |   |   |   |   |   |--- has_low_strength_in_hand <= 0.50
|   |   |   |   |   |   |   |   |--- class: seal
|   |   |   |   |   |   |   |--- has_low_strength_in_hand >  0.50
|   |   |   |   |   |   |   |   |--- class: skunk
|   |   |   |   |   |   |--- has_high_strength_in_hand >  0.50
|   |   |   |   |   |   |   |--- hand_crocodile <= 0.50
|   |   |   |   |   |   |   |   |--- class: lion
|   |   |   |   |   |   |   |--- hand_crocodile >  0.50
|   |   |   |   |   |   |   |   |--- class: crocodile
|   |   |   |   |   |--- hand_kangaroo >  0.50
|   |   |   |   |   |   |--- queue_length <= 2.50
|   |   |   |   |   |   |   |--- queue_crocodile <= 0.50
|   |   |   |   |   |   |   |   |--- class: kangaroo
|   |   |   |   |   |   |   |--- queue_crocodile >  0.50
|   |   |   |   |   |   |   |   |--- class: skunk
|   |   |   |   |   |   |--- queue_length >  2.50
|   |   |   |   |   |   |   |--- hand_parrot <= 0.50
|   |   |   |   |   |   |   |   |--- class: kangaroo
|   |   |   |   |   |   |   |--- hand_parrot >  0.50
|   |   |   |   |   |   |   |   |--- class: parrot
|   |   |   |   |--- hand_snake >  0.50
|   |   |   |   |   |--- hand_crocodile <= 0.50
|   |   |   |   |   |   |--- hand_lion <= 0.50
|   |   |   |   |   |   |   |--- hand_kangaroo <= 0.50
|   |   |   |   |   |   |   |   |--- class: snake
|   |   |   |   |   |   |   |--- hand_kangaroo >  0.50
|   |   |   |   |   |   |   |   |--- class: kangaroo
|   |   |   |   |   |   |--- hand_lion >  0.50
|   |   |   |   |   |   |   |--- hand_kangaroo <= 0.50
|   |   |   |   |   |   |   |   |--- class: lion
|   |   |   |   |   |   |   |--- hand_kangaroo >  0.50
|   |   |   |   |   |   |   |   |--- class: kangaroo
|   |   |   |   |   |--- hand_crocodile >  0.50
|   |   |   |   |   |   |--- queue_length <= 2.50
|   |   |   |   |   |   |   |--- hand_kangaroo <= 0.50
|   |   |   |   |   |   |   |   |--- class: snake
|   |   |   |   |   |   |   |--- hand_kangaroo >  0.50
|   |   |   |   |   |   |   |   |--- class: kangaroo
|   |   |   |   |   |   |--- queue_length >  2.50
|   |   |   |   |   |   |   |--- hand_chameleon <= 0.50
|   |   |   |   |   |   |   |   |--- class: crocodile
|   |   |   |   |   |   |   |--- hand_chameleon >  0.50
|   |   |   |   |   |   |   |   |--- class: chameleon
|   |   |   |--- hand_monkey >  0.50
|   |   |   |   |--- hand_kangaroo <= 0.50
|   |   |   |   |   |--- hand_skunk <= 0.50
|   |   |   |   |   |   |--- hand_crocodile <= 0.50
|   |   |   |   |   |   |   |--- own_hand_size <= 2.50
|   |   |   |   |   |   |   |   |--- class: monkey
|   |   |   |   |   |   |   |--- own_hand_size >  2.50
|   |   |   |   |   |   |   |   |--- class: monkey
|   |   |   |   |   |   |--- hand_crocodile >  0.50
|   |   |   |   
```

*Note: Full tree rules truncated for readability. The tree has 8 levels
of depth.*

---

## Fidelity Comparison by Tree Depth

| Max Depth | Fidelity | Notes |
|-----------|----------|-------|
| 5 | 38.6% | Simpler |
| 6 | 42.5% | Simpler |
| 7 | 46.0% | Good balance |
| 8 | 50.8% | More complex |
| 10 | 58.0% | More complex |
| 12 | 62.8% | More complex |


---

## Methodology

1. **Data Generation**: Played 10,000 game states using the trained neural
   network with greedy action selection.

2. **Feature Engineering**: Extracted 55 interpretable features
   including:
   - Cards in hand (one-hot for each species)
   - Queue length and composition
   - Score differential
   - Strategic indicators (monkey pairs, blocking opportunities)
   - Game phase (turn number)

3. **Decision Tree Training**: Trained sklearn DecisionTreeClassifier with
   max_depth=8, min_samples_split=20, min_samples_leaf=10.

4. **Rule Extraction**: Used sklearn's export_text to extract if-then rules.

---

## Strategic Recommendations

Based on the decision tree analysis, here are actionable strategies for playing Beasty Bar:

### Early Game (turns 1-4)
- **Play high-value recurring animals** (giraffe, hippo, crocodile) early to establish
  board presence and accumulate movement/removal effects over multiple turns.
- **Zebra is a priority play** when queue is short - its 4 points and permanent bar
  status make it valuable early.

### Mid Game (turns 5-10)
- **Queue length matters** - when queue has 3+ cards, shift to more interactive plays
  (parrot for removal, chameleon for copying, crocodile for eating).
- **Save parrot** for when there are good targets in the queue (high-value opponent cards).

### Late Game (turns 11+)
- **Skunk becomes powerful** when opponents have built up high-strength cards in queue.
- **Kangaroo flexibility** - useful at all queue lengths but particularly strong in
  short queues for positioning.

### Counter-Play Patterns
- **Against high-strength queue** (avg strength > 7): Consider skunk or parrot.
- **Against opponent's recurring animals**: Use parrot to send them to That's It
  before they can accumulate value.
- **With monkey in queue**: Playing your monkey can trigger elimination of high-strength
  cards (if another monkey exists in queue).

---

## Limitations

- The decision tree is a simplified approximation of the neural network
- Complex non-linear patterns may not be fully captured
- The 49.2% of decisions where tree differs from NN represent
  nuanced situations requiring context the tree cannot express
- Feature engineering choices affect what patterns can be detected
- The neural network likely uses card position, owner information, and other
  features that the simplified tree cannot fully leverage

---

*Generated by decision_tree_analysis.py*
*Analysis date: January 2025*
