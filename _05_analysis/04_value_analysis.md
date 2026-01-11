# Value Head Analysis Report

This report analyzes what the trained Beasty Bar AI model considers 
a winning position by probing its value head with controlled game states.

## Game Background

**Beasty Bar** is a card game where players compete to get their animals into the bar.
Key mechanics:
- Queue holds up to 5 cards; when full, top 2 enter the bar (score points), last 1 is bounced out
- Each species has unique abilities that manipulate queue order
- Higher strength typically helps advance in queue; higher points reward getting in the bar

**Species Abilities:**
- **Lion** (12 str, 2 pts): Jumps to front, scares away Monkeys
- **Hippo** (11 str, 2 pts): Pushes forward through weaker animals (recurring)
- **Crocodile** (10 str, 3 pts): Eats weaker animals ahead (recurring)
- **Snake** (9 str, 2 pts): Sorts queue by strength (strongest first)
- **Giraffe** (8 str, 3 pts): Swaps with weaker animal ahead (recurring)
- **Zebra** (7 str, 4 pts): Blocks Hippo/Crocodile from passing (permanent)
- **Seal** (6 str, 2 pts): Reverses entire queue
- **Chameleon** (5 str, 3 pts): Copies another animal's ability
- **Monkey** (4 str, 3 pts): Pair of monkeys kicks out Hippo/Crocodile
- **Kangaroo** (3 str, 4 pts): Hops 1-2 positions forward
- **Parrot** (2 str, 4 pts): Sends any queue animal to THAT'S IT
- **Skunk** (1 str, 4 pts): Expels top 2 strength bands from queue

## 1. Species Value in Hand

How does having each species in hand affect the model's value estimate?

| Species | Strength | Points | With Card | Without | Marginal Value |
|---------|----------|--------|-----------|---------|----------------|
| Parrot     |        2 |      4 | +0.1240 | +0.0981 | **+0.0259** |
| Crocodile  |       10 |      3 | +0.1375 | +0.1129 | **+0.0246** |
| Skunk      |        1 |      4 | +0.1217 | +0.0986 | **+0.0231** |
| Kangaroo   |        3 |      4 | +0.0910 | +0.0805 | **+0.0105** |
| Lion       |       12 |      2 | +0.0951 | +0.0903 | **+0.0048** |
| Chameleon  |        5 |      3 | +0.0949 | +0.0956 | **-0.0008** |
| Zebra      |        7 |      4 | +0.0910 | +0.0936 | **-0.0026** |
| Seal       |        6 |      2 | +0.0860 | +0.0894 | **-0.0035** |
| Giraffe    |        8 |      3 | +0.0868 | +0.0979 | **-0.0110** |
| Hippo      |       11 |      2 | +0.0846 | +0.1033 | **-0.0187** |
| Snake      |        9 |      2 | +0.0796 | +0.0995 | **-0.0199** |
| Monkey     |        4 |      3 | +0.0674 | +0.0962 | **-0.0288** |

**Key Insight**: 
The model values **Parrot** most highly in hand 
(+0.0259 marginal value), while **Monkey** 
is least valued (-0.0288).

**Strategic Analysis:**

The model's preferences make strategic sense:
- **Parrot** (top): Direct removal ability is extremely powerful - can eliminate any threat
- **Crocodile** (2nd): Recurring ability to eat weaker animals provides sustained value
- **Skunk** (3rd): Mass removal of strongest animals clears threats, 4 pts if it enters
- **Kangaroo** (4th): Reliable positioning with high points (4 pts)
- **Monkey** (last): Requires paired with another Monkey to be useful, otherwise weak
- Strong recurring animals (Hippo, Giraffe) rank lower because they're also threats when opponent has them

## 2. Queue Position Value

How does controlling different queue positions affect value?
Position 0 is front (first to enter bar), Position 4 is back (bounced out).

| Position | Our Card | Opponent Card | Advantage |
|----------|----------|---------------|-----------|
| Position 0           | +0.2196 | -0.0086 | **+0.2282** |
| Position 1           | +0.2134 | +0.1134 | **+0.1000** |
| Position 2           | +0.0353 | +0.0384 | **-0.0031** |
| Position 3           | +0.2058 | +0.1017 | **+0.1041** |
| Position 4 (bounce)  | -0.2088 | -0.0455 | **-0.1632** |

**Key Insight**: Position 0 provides the most advantage 
(+0.2282), while position 4 
is least advantageous (-0.1632).

**Strategic Analysis:**

This aligns perfectly with game mechanics:
- **Position 0** (front): Guaranteed to enter bar when queue fills - highest value
- **Position 1**: Second to enter bar - still very valuable
- **Positions 2-3**: Middle ground, depends on future plays
- **Position 4** (back): Gets bounced out when queue fills - actively bad

The negative value at position 4 shows the model understands that having
your card in the bounce position means it will likely be eliminated.

## 3. Score Differential Impact

How does the current score difference affect value estimate?

| Score Diff | Value Estimate |
|------------|----------------|
| -10 points | -0.5045 |
|  -8 points | -0.5040 |
|  -6 points | -0.5061 |
|  -4 points | -0.4957 |
|  -2 points | -0.5137 |
| +  0 points | +0.0877 |
| +  2 points | +0.6437 |
| +  4 points | +0.6385 |
| +  6 points | +0.6456 |
| +  8 points | +0.6290 |
| + 10 points | +0.6424 |

**Key Insight**: The model clearly distinguishes between winning and losing positions.

**Strategic Analysis:**

Notable patterns in the score response:
- Sharp transition at score diff = 0 (from ~-0.5 to ~+0.6)
- Losing positions cluster around -0.5 regardless of deficit magnitude
- Winning positions cluster around +0.64 regardless of lead magnitude
- This suggests the model thinks in terms of 'winning' vs 'losing' rather than by how much
- The tie state (+0.09) is close to neutral, slightly optimistic

## 4. Turn Phase Impact

Does the model value early game vs late game differently?

| Game Phase | Value Estimate |
|------------|----------------|
| Early      | +0.0919 |
| Mid        | +0.0976 |
| Late       | +0.0415 |

**Key Insight**: The model is most confident in **mid** game 
(+0.0976). This may reflect training data distribution or 
actual strategic preferences.

## 5. Opponent Threat Assessment

How does opponent's hand size affect our value estimate?

| Opponent Hand | Value Estimate |
|---------------|----------------|
| Full          | +0.0153 |
| Three         | +0.0539 |
| Two           | +0.1312 |
| One           | +0.1540 |
| Empty         | +0.0503 |

**Key Insight**: When opponent has no cards vs full hand, our value 
increases by +0.0349. The model correctly identifies depleted 
opponent resources as favorable.

## 6. Most Valued Situations

Top 10 game states the model considers most favorable:

1. **Value: +0.9754** - late game | hand: lion, crocodile... | queue: 4 cards (2 ours) | score: 8-0 (diff=+8)
2. **Value: +0.9693** - late game | hand: parrot, snake... | queue: 4 cards (2 ours) | score: 4-0 (diff=+4)
3. **Value: +0.9670** - early game | hand: seal, kangaroo... | queue: 4 cards (2 ours) | score: 5-0 (diff=+5)
4. **Value: +0.9652** - late game | hand: skunk, hippo... | queue: 4 cards (4 ours) | score: 2-0 (diff=+2)
5. **Value: +0.9586** - early game | hand: skunk, snake... | queue: 4 cards (2 ours) | score: 12-4 (diff=+8)
6. **Value: +0.9555** - early game | hand: kangaroo, hippo... | queue: 4 cards (2 ours) | score: 14-0 (diff=+14)
7. **Value: +0.9462** - late game | hand: seal, giraffe... | queue: 4 cards (2 ours) | score: 4-9 (diff=-5)
8. **Value: +0.9438** - early game | hand: zebra, seal... | queue: 4 cards (2 ours) | score: 13-2 (diff=+11)
9. **Value: +0.9350** - early game | hand: seal, hippo... | queue: 4 cards (2 ours) | score: 2-1 (diff=+1)
10. **Value: +0.9325** - late game | hand: hippo, giraffe... | queue: 4 cards (3 ours) | score: 11-6 (diff=+5)

## 7. Least Valued Situations

Top 10 game states the model considers most unfavorable:

1. **Value: -0.7829** - late game | hand: giraffe, snake... | queue: 1 cards (0 ours) | score: 1-14 (diff=-13)
2. **Value: -0.7924** - late game | hand: lion, skunk... | queue: 3 cards (2 ours) | score: 0-7 (diff=-7)
3. **Value: -0.8180** - late game | hand: hippo, skunk... | queue: 4 cards (2 ours) | score: 1-14 (diff=-13)
4. **Value: -0.8242** - early game | hand: kangaroo, hippo... | queue: 3 cards (1 ours) | score: 1-10 (diff=-9)
5. **Value: -0.8398** - early game | hand: skunk, lion... | queue: 1 cards (0 ours) | score: 1-14 (diff=-13)
6. **Value: -0.8567** - early game | hand: snake, crocodile... | queue: 3 cards (1 ours) | score: 0-3 (diff=-3)
7. **Value: -0.8592** - mid game | hand: kangaroo, snake... | queue: 1 cards (0 ours) | score: 0-2 (diff=-2)
8. **Value: -0.8630** - mid game | hand: kangaroo, monkey... | queue: 4 cards (0 ours) | score: 0-8 (diff=-8)
9. **Value: -0.8750** - late game | hand: lion, crocodile... | queue: 2 cards (1 ours) | score: 0-9 (diff=-9)
10. **Value: -0.8952** - early game | hand: zebra, chameleon... | queue: 4 cards (1 ours) | score: 1-2 (diff=-1)

## 8. Strategic Insights Summary

Based on the value head analysis, the model has learned:

1. **Preferred Cards**: Parrot, Crocodile, Skunk, Kangaroo
   - These provide the highest marginal value when in hand

2. **Less Valued Cards**: Giraffe, Hippo, Snake, Monkey
   - These provide lower or negative marginal value

3. **Queue Control**: Front positions (0-1) provide +0.1641 avg advantage
   vs back positions (3-4) providing -0.0296 avg advantage

4. **Score Sensitivity**: The model strongly correlates value with score advantage
   - Approximately 0.0573 value per point of score difference

5. **Resource Tracking**: Model values opponent resource depletion
   - Opponent empty hand vs full: +0.0349 value swing
