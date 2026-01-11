# Beasty Bar AI Strategy Guide

**Extracted from trained neural network (v4/final.pt)**
**Training: 600 iterations, 5,027,840 games**
**Win rate: 79% against heuristic opponents**

This guide distills the strategic knowledge learned by the AI through millions of self-play games. These insights were extracted using six complementary analysis methods: species embedding analysis, behavioral pattern mining, decision tree distillation, value function probing, attention visualization, and counterfactual experiments.

---

## Executive Summary: The AI's Core Philosophy

The trained model has learned three fundamental principles:

1. **Abilities beat strength** - Special abilities (jumping, removing, copying) are more valuable than raw power
2. **Timing is everything** - Save high-impact cards (Seal, Snake, Skunk) for decisive moments
3. **Position matters most** - Front queue positions are worth fighting for; back positions are death

---

## Card Tier List (What the AI Values Most)

Based on combined analysis of value head probing, counterfactual experiments, and behavioral patterns:

### S-Tier: Always Valuable
| Card | Why the AI Loves It |
|------|---------------------|
| **Kangaroo** | +0.107 value. Jump ability bypasses queue politics. Most universally valuable card. |
| **Parrot** | +0.026 value. Targeted removal is always useful. |
| **Giraffe** | 36% play rate. Reliable recurring advancement. Works from any position. |

### A-Tier: Strong in Context
| Card | Why the AI Values It |
|------|---------------------|
| **Skunk** | +0.023 value. Usage TRIPLES (+233%) when threats present. The "emergency button." |
| **Crocodile** | +0.025 value. Best card for late game (+96% preference increase). Recurring eating devastates opponents. |
| **Zebra** | 30% consistent play rate. Permanent and cannot be removed. Guaranteed points. |

### B-Tier: Situational
| Card | When the AI Uses It |
|------|---------------------|
| **Monkey** | Only valued with combo (99.8% play rate when opponent Monkey in queue, 0.07% otherwise). |
| **Lion** | Strength-12 seems impressive but AI ranks it -0.038 value. Predictable and easily countered. |
| **Hippo** | Lowest valued card (-0.056). Recurring push is too slow. |

### C-Tier: Specialists
| Card | AI's Approach |
|------|---------------|
| **Seal** | Lowest play rate (10.8%). Saved for late game where 65% of its usage occurs. Queue reversal is situational but devastating. |
| **Snake** | 18% play rate. Sorting only valuable when queue is complex. |
| **Chameleon** | 16% play rate. Flexibility valued, but depends entirely on what's available to copy. |

---

## Timing Strategy: When to Play Each Card

### Early Game (Turns 1-3): Establish Position

**Play these cards:**
- **Kangaroo** (42.7% preference) - Hop ahead immediately
- **Monkey** (41.6%) - Set up for future pair combo
- **Giraffe** (38.2%) - Start advancing
- **Zebra** (33%) - Protected early entry

**Avoid these cards:**
- Seal (6.1%) - Too early for reversal value
- Snake (15.7%) - Queue not complex enough
- Crocodile (17.4%) - Eating is more valuable later

### Mid Game (Turns 4-7): Control the Board

**Play these cards:**
- **Giraffe** (41%) - Still advancing
- **Zebra** (37.7%) - Lock in points
- **Skunk** emerges (26.1%) - Start defending against threats

**Key shift:** Begin responding to opponent threats rather than pure positioning.

### Late Game (Turns 8+): Execute the Endgame

**Play these cards:**
- **Crocodile** (34%) - Eat opponents for swing turns
- **Seal** now (138% increase from early game) - Reversal can steal wins
- **Parrot** (18% increase) - Targeted removal of threats

**The AI's endgame philosophy:** High-impact cards are saved for moments when they're decisive.

---

## Threat Assessment: What to Fear

The AI's value function reveals which opponent cards are most dangerous:

| Threat | Value Drop | Why It's Dangerous |
|--------|------------|-------------------|
| **Zebra** | -0.335 | **Most feared.** Permanent, high points (4), blocks advancement. Cannot be removed. |
| **Giraffe** | -0.210 | Recurring advancement is hard to stop |
| **Snake** | -0.146 | Reorders queue unpredictably |
| **Crocodile** | -0.120 | Recurring eating each turn |
| **Hippo** | -0.119 | Recurring push |
| **Lion** | -0.086 | Despite highest strength, least threatening. Predictable. |

**Counter-intuitive insight:** The AI fears ability cards (Zebra, Giraffe) far more than raw strength cards (Lion). Zebra at strength 7 is 4x more threatening than Lion at strength 12.

---

## Defensive Play: Responding to Threats

When the AI detects threats (Lion, Hippo, or Crocodile in queue), its behavior changes dramatically:

| Your Response | Probability Change | Why |
|---------------|-------------------|-----|
| **Play Skunk** | +233% | Primary defensive weapon. Expels high-strength cards. |
| Play Crocodile | +22% | Fight fire with fire. Counter-eat. |
| Play Lion | +19% | Out-strength the threat |
| Avoid Kangaroo | -41% | Don't jump into danger |
| Avoid Giraffe | -22% | Less aggressive when threatened |

**Specific threat responses:**

- **Against Crocodile:** Skunk dominates (41.6% preference). Zebra (33.9%) is safe since it can't be eaten.
- **Against Hippo:** Zebra (36.4%) blocks its push. Giraffe (40.2%) advances past it.
- **Against Lion:** Giraffe (40.1%) leans past. Crocodile (34.1%) can eat it.

---

## Position Strategy: Where Cards Should Be

The AI pays 1.71x more attention to front queue positions than back positions.

### Queue Position Values

| Position | AI's Attention | Strategy |
|----------|---------------|----------|
| **Position 0 (Front)** | 53% attention | +0.107 advantage when your card. Next to enter Beasty Bar. |
| Position 1 | 37% attention | Also scores on queue overflow. Second priority. |
| Position 2 | 25% attention | Middle is neutral. |
| Position 3 | 27% attention | Getting risky. |
| **Position 4 (Back)** | Least attention | Card bounces to THAT'S IT. Avoid having cards here. |

### Card-Specific Positioning

| Card | Best Position | Why |
|------|--------------|-----|
| **Crocodile** | Front (+0.465 value) | Eats the card behind it |
| **Lion** | Back (+0.367 value) | Protected from counters |
| **Giraffe** | Middle (+0.78 value) | Flexibility, recurring works anywhere |
| **Hippo** | Any (position-insensitive) | Recurring push works from anywhere |

---

## Combo Recognition: Exploiting Synergies

### The Monkey Pair

The AI shows **perfect combo recognition** for Monkey pairs:

- Without opponent Monkey in queue: 0.07% chance to play Monkey
- With opponent Monkey in queue: 99.87% chance to play Monkey
- **1,427x probability increase**

**Lesson:** When opponent has Monkey in queue, ALWAYS play your Monkey if you have one. This is near-mandatory.

### Queue Manipulation Combos

The AI saves queue manipulation cards (Seal, Snake, Parrot) for late game:

| Card | Late Game Usage | Combo Potential |
|------|-----------------|-----------------|
| Seal | 65% of usage | Reverse queue when your cards are at back |
| Snake | 51% of usage | Sort when queue has mixed strengths |
| Parrot | 48% of usage | Snipe opponent's front card before it scores |

---

## Species Relationships: What the AI Learned

The AI's 32-dimensional species embeddings reveal how it conceptualizes card relationships:

### Natural Enemies (Most Different)
- **Hippo vs Zebra** (-0.292): Zebra specifically blocks Hippo. Direct counter.
- **Crocodile vs Zebra** (-0.203): Zebra blocks Crocodile too.
- **Giraffe vs Chameleon** (-0.357): Fixed ability vs adaptive ability - opposite strategies.

### Strategic Allies (Most Similar)
- **Zebra + Chameleon** (0.148): Both are reactive, read the board
- **Chameleon + Parrot** (0.131): Both require board analysis to use well
- **Hippo + Kangaroo** (0.123): Both move forward aggressively

### Unique Identities
The AI clusters these cards as strategically distinct:
- **Lion**: Queue dominance, but vulnerable
- **Seal**: Unique queue reversal (strongest embedding)
- **Monkey**: Gang mechanic for pair combos
- **Crocodile**: Recurring predation

---

## Decision Framework: What Drives the AI's Choices

From decision tree distillation (46.5% fidelity to neural network):

### Most Important Factors
1. **Having Giraffe in hand** (14% importance) - Reliable advancement
2. **Having Zebra in hand** (11% importance) - Protected entry
3. **Queue length** (10.5% importance) - Determines available tactics
4. **Having Hippo in hand** (10% importance) - Recurring push
5. **Can use Parrot** (9% importance) - Removal opportunity

### Simplified Decision Rules

```
IF have Giraffe → strong preference to play it
IF have Zebra AND queue not full → consider playing for safe points
IF queue > 2 cards AND can use Parrot → remove threats
IF opponent Monkey in queue AND have Monkey → PLAY MONKEY (99.8%)
IF threats present → use Skunk (+233% preference)
```

---

## Key Strategic Principles

### 1. Ability Cards > Raw Strength
The AI values Kangaroo (strength 3) higher than Lion (strength 12). Special abilities that manipulate position or remove cards are more powerful than high numbers.

### 2. Save Impact Cards for Late Game
Seal, Snake, and Parrot have low early-game play rates but spike in late game. These cards are "finishers" - their value comes from decisive timing.

### 3. Skunk is Your Panic Button
When the board looks threatening, Skunk usage triples. It's the universal defensive answer.

### 4. Zebra is the Real King
Despite being strength 7, Zebra is:
- The most threatening opponent card (-0.335 value drop)
- One of the most consistently played cards (30%+ across all phases)
- Permanent and immune to removal

### 5. Monkey Combos are Mandatory
99.8% probability to play Monkey when combo exists. The AI treats this as near-compulsory.

### 6. Front Position is Worth Fighting For
1.71x more attention paid to front positions. Get your cards to the front; prevent opponent cards from scoring.

---

## Summary: The AI's Winning Formula

1. **Early game**: Kangaroo, Monkey (for setup), Giraffe, Zebra
2. **Mid game**: Continue Giraffe, add Skunk for defense, watch for Monkey combos
3. **Late game**: Crocodile for eating, Seal/Snake for manipulation, Parrot for sniping
4. **Always**: Fear Zebra, exploit Monkey pairs, value abilities over strength

---

*Generated from 6 parallel analyses: species embeddings, behavioral patterns, decision tree distillation, value function probing, attention visualization, and counterfactual experiments.*

*Model: BeastyBarNetwork v4 | 600 iterations | 5,027,840 games | 79% win rate*
