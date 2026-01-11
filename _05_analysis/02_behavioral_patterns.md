# Behavioral Pattern Analysis: Trained Beasty Bar AI

**Model**: `checkpoints/v4/final.pt`
**Analysis Date**: 2024
**States Analyzed**: 500 randomly generated game positions across all turn phases

---

## Executive Summary

This analysis extracts strategic patterns from a trained neural network model by examining its policy distributions across 500+ game states. The model demonstrates sophisticated understanding of game phases, threat response, and card timing.

### Key Findings

1. **Early game focus on positioning cards** (Kangaroo, Monkey, Giraffe) that can advance in the queue
2. **Late game preference for high-value predators** (Crocodile, Giraffe) that eat points or dominate
3. **Strong defensive response** with Skunk when threats are present (+233% preference increase)
4. **Conservative use of Seal** (queue reversal) - lowest play rate at 10.8%
5. **Opportunistic Monkey pairs** - plays increase 50% when another Monkey is in queue

---

## 1. Card Play Frequency by Turn Phase

### Overall Play Rates (plays / availability)

| Rank | Species    | Play Rate | Interpretation                               |
|------|------------|-----------|----------------------------------------------|
| 1    | Giraffe    | 36.2%     | Most favored overall - reliable advancement |
| 2    | Kangaroo   | 35.1%     | High value - consistent queue jumping        |
| 3    | Zebra      | 30.2%     | Protected entry - cannot be removed          |
| 4    | Lion       | 29.1%     | Alpha predator - pushes to front             |
| 5    | Monkey     | 28.8%     | Pairs drive out predators                    |
| 6    | Crocodile  | 26.6%     | Recurring eater - late game value            |
| 7    | Hippo      | 26.5%     | Recurring pusher                             |
| 8    | Parrot     | 20.9%     | Targeted removal                             |
| 9    | Skunk      | 20.5%     | Situational defense                          |
| 10   | Snake      | 18.0%     | Sorting utility                              |
| 11   | Chameleon  | 16.5%     | Copy ability - context dependent             |
| 12   | Seal       | 10.8%     | Queue reversal - saved for specific moments  |

### Early Game Preferences (Turns 1-3)

The model prioritizes **fast positioning cards** early to establish queue presence:

| Rank | Species    | Mean Prob | Strategy                                     |
|------|------------|-----------|----------------------------------------------|
| 1    | Kangaroo   | 42.7%     | Hop ahead immediately                        |
| 2    | Monkey     | 41.6%     | Setup for pair combo                         |
| 3    | Giraffe    | 38.2%     | Lean forward through weaker cards            |
| 4    | Zebra      | 33.0%     | Protected entry - cannot be bounced          |
| 5    | Lion       | 27.9%     | Establish dominance                          |
| 6    | Hippo      | 27.7%     | Recurring push potential                     |

**Early game avoids**: Seal (6.1%), Snake (15.7%), Crocodile (17.4%)

### Mid Game Preferences (Turns 4-7)

Shift toward **board control and point accumulation**:

| Rank | Species    | Mean Prob | Strategy                                     |
|------|------------|-----------|----------------------------------------------|
| 1    | Giraffe    | 41.0%     | Still advancing through queue                |
| 2    | Zebra      | 37.7%     | Protected high points                        |
| 3    | Kangaroo   | 34.1%     | Continued positioning                        |
| 4    | Lion       | 32.8%     | Push through accumulated cards               |
| 5    | Hippo      | 27.8%     | Recurring disruption                         |
| 6    | Skunk      | 26.1%     | Start using defensively                      |

**Mid game emergence**: Skunk usage increases as threats accumulate

### Late Game Preferences (Turns 8+)

Focus shifts to **high-value removal and endgame positioning**:

| Rank | Species    | Mean Prob | Strategy                                     |
|------|------------|-----------|----------------------------------------------|
| 1    | Crocodile  | 34.0%     | Eat opponents' cards for swing turns         |
| 2    | Giraffe    | 32.8%     | Still valuable for advancement               |
| 3    | Zebra      | 28.7%     | Guaranteed points                            |
| 4    | Kangaroo   | 28.3%     | Position for final pushes                    |
| 5    | Monkey     | 26.9%     | Drive out remaining threats                  |
| 6    | Hippo      | 26.1%     | Recurring pushes still valuable              |

**Late game increases**: Seal (+138%), Snake (+23%), Parrot (+18%) all increase relative to early game

---

## 2. Situational Preferences: Threat Response

### When Any Threat (Lion/Hippo/Crocodile) is Present in Queue

The model shows **dramatically different preferences** when facing threats:

| Rank | Species    | Mean Prob | Change vs No Threats | Interpretation           |
|------|------------|-----------|----------------------|--------------------------|
| 1    | **Skunk**  | **34.3%** | **+233%**            | Primary defensive weapon |
| 2    | Giraffe    | 31.1%     | -22%                 | Less aggressive          |
| 3    | Crocodile  | 30.0%     | +22%                 | Fight fire with fire     |
| 4    | Lion       | 29.8%     | +19%                 | Counter with strength    |
| 5    | Zebra      | 28.3%     | -17%                 | Still protected          |
| 6    | Kangaroo   | 24.9%     | -41%                 | Avoid jumping into danger|

**Key insight**: The model's Skunk preference **triples** when threats are present, confirming it learned Skunk's ability to expel high-strength cards.

### Specific Threat Responses

#### When Lion is in Queue (Strength 12)
- **Giraffe** leads at 40.1% - can lean past weaker cards
- **Crocodile** at 34.1% - recurring eating can remove Lion
- **Skunk** at 25.5% - expels top strength band

#### When Crocodile is in Queue (Strength 10, Recurring)
- **Skunk** dominates at **41.6%** - best counter
- **Zebra** at 33.9% - cannot be eaten (permanent)
- **Lion** at 30.8% - strength dominance
- **Giraffe** at 30.5% - advancement

#### When Hippo is in Queue (Strength 11, Recurring)
- **Giraffe** at 40.2% - can advance past
- **Zebra** at 36.4% - immune to bouncing
- **Hippo** at 35.2% - fight with own Hippo
- **Lion** at 30.0% - out-strength it

---

## 3. Queue Manipulation Card Usage

### Seal (Reverses queue order)
- **Total high-preference plays**: 20
- **Phase distribution**: Early 15% | Mid 20% | Late 65%
- **Strategy**: Saved for late game when queue reversal is decisive
- **Overall play rate**: 10.8% (lowest of all cards)

### Snake (Sorts queue by strength)
- **Total high-preference plays**: 35
- **Phase distribution**: Early 20% | Mid 29% | Late 51%
- **Strategy**: Used increasingly as game progresses and queue complexity increases
- **Overall play rate**: 18.0%

### Parrot (Removes target card to THAT'S IT)
- **Total high-preference plays**: 29
- **Phase distribution**: Early 31% | Mid 21% | Late 48%
- **Strategy**: More evenly distributed but peaks in late game for targeted removal
- **Overall play rate**: 20.9%

**Pattern**: All three manipulation cards are saved preferentially for late game (48-65% of usage), suggesting the model learned their highest value comes when the board state is more developed.

---

## 4. Combo Play Analysis

### Monkey Pair Strategy

The Monkey's special ability triggers when two Monkeys are in the queue together, driving out the heaviest predators (Lion, Hippo, Crocodile).

| Situation                     | Monkey Plays | Interpretation                    |
|-------------------------------|--------------|-----------------------------------|
| Monkey already in queue       | 20           | Actively seeks pair combo         |
| No Monkey in queue            | 157          | Still plays for setup             |
| **Ratio**                     | **1:7.85**   | Queue Monkey -> 50% more likely   |

**Insight**: When a Monkey is already in queue, the model is approximately **50% more likely** to play its own Monkey, showing learned understanding of the pair combo. However, it doesn't exclusively wait for setups - it plays Monkeys proactively to create future combo opportunities.

---

## 5. Identified Strategic Patterns

### Pattern 1: Positioning-First Early Game
The model prioritizes cards that can advance in the queue (Kangaroo 42.7%, Giraffe 38.2%) over raw strength (Crocodile 17.4%) in early turns. This suggests learned understanding that early positioning beats early aggression.

### Pattern 2: Skunk as Defensive Anchor
Skunk usage shows the strongest situational variance:
- **10.3%** mean probability when no threats present
- **34.3%** mean probability when threats present (3.3x increase)

This is the clearest evidence of threat-aware defensive play.

### Pattern 3: Crocodile for Late Game Swings
Crocodile preference increases 96% from early (17.4%) to late game (34.0%), indicating learned value of its recurring eating ability for final-turn point swings.

### Pattern 4: Seal Conservation
With the lowest play rate (10.8%) and 65% of high-preference plays in late game, the model treats Seal as a "finisher" - a card whose queue-reversal ability is most valuable at decisive moments.

### Pattern 5: Zebra Consistency
Zebra maintains 30-38% preference across all phases, reflecting its unique permanent status (cannot be removed) making it a reliable point source at any time.

### Pattern 6: Chameleon Opportunism
Chameleon's low overall play rate (16.5%) but high max probability (100%) indicates the model waits for high-value copy targets rather than playing it speculatively.

---

## 6. Statistical Summary

### Policy Distribution by Species (All States)

| Species    | Mean   | Std Dev | Max    | Min       |
|------------|--------|---------|--------|-----------|
| Giraffe    | 0.359  | 0.406   | 0.9999 | 9.0e-09   |
| Kangaroo   | 0.339  | 0.418   | 1.0000 | 1.7e-08   |
| Zebra      | 0.316  | 0.401   | 0.9999 | 6.1e-17   |
| Monkey     | 0.281  | 0.373   | 0.9998 | 4.7e-18   |
| Hippo      | 0.269  | 0.367   | 0.9998 | 6.2e-19   |
| Lion       | 0.268  | 0.387   | 0.9999 | 4.3e-19   |
| Crocodile  | 0.265  | 0.380   | 0.9999 | 2.5e-09   |
| Parrot     | 0.221  | 0.367   | 0.9999 | 5.8e-09   |
| Skunk      | 0.215  | 0.360   | 0.9999 | 1.1e-18   |
| Chameleon  | 0.187  | 0.327   | 1.0000 | 2.2e-10   |
| Snake      | 0.183  | 0.309   | 0.9992 | 1.7e-09   |
| Seal       | 0.113  | 0.269   | 0.9999 | 7.8e-09   |

**Note**: High standard deviations indicate strong context-dependence - the model doesn't play cards uniformly but responds to game state.

---

## Methodology

1. **State Generation**: 500 random game states generated by playing random moves from initial positions (0-15 turns per game)
2. **Policy Extraction**: For each state, ran model forward pass to get policy probability distribution over legal actions
3. **Species Aggregation**: Summed action probabilities by target species (accounting for multi-action cards)
4. **Phase Classification**: Early (turns 1-3), Mid (turns 4-7), Late (turns 8+)
5. **Threat Detection**: Checked queue for Lion, Hippo, or Crocodile presence

---

## Files Generated

- `behavioral_patterns_raw.json` - Complete numerical analysis data
- `02_behavioral_patterns.md` - This report
