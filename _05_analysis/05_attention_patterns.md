# Attention Pattern Analysis for Beasty Bar AI

This analysis examines what the trained neural network "looks at" when processing
the queue zone. The queue is processed by a TransformerEncoder which uses
self-attention to understand relationships between cards.

## Summary

### Attention by Queue Position

Position 0 is the **front** of the queue (next to enter the bar when queue is full).
Position 4 is the **back** (most recently played, will "bounce" if queue fills).

| Position | Avg Attention | Std Dev | Interpretation |
|----------|--------------|---------|----------------|
| 0 | 0.5314 | 0.2861 | Front - about to score |
| 1 | 0.3710 | 0.1246 | Second - also scores on overflow |
| 2 | 0.2450 | 0.1210 | Middle positions |
| 3 | 0.2710 | 0.0525 | Middle positions |

**Front positions (0-1) receive 1.71x more attention than back positions (3-4)**

The model prioritizes cards near the front of the queue, which makes strategic sense:
- Cards at positions 0-1 will score (enter Beasty Bar) when the queue fills
- The card at position 4 will be bounced to THAT'S IT

### Attention: Own Cards vs Opponent Cards

| Card Owner | Avg Attention | Std Dev | Sample Count |
|------------|--------------|---------|--------------|
| Own cards | 0.4045 | 0.2307 | 400 |
| Opponent cards | 0.3955 | 0.2350 | 400 |

The model pays **roughly equal attention** to own and opponent cards.

### Attention by Species

How much attention does each species receive when in the queue?

| Species | Avg Attention | Std Dev | Count | Notes |
|---------|--------------|---------|-------|-------|
| Crocodile | 0.6142 | 0.3017 | 68 | Str 10, recurring ability |
| Monkey | 0.4743 | 0.2712 | 20 | Str 4, removes high strength |
| Skunk | 0.4531 | 0.2769 | 86 | Str 1, mass removal |
| Zebra | 0.4167 | 0.1967 | 34 | Str 7, permanent (stays in bar) |
| Giraffe | 0.4130 | 0.1968 | 62 | Str 8, moves forward |
| Hippo | 0.4017 | 0.2200 | 84 | Str 11, recurring ability |
| Seal | 0.3957 | 0.2685 | 72 | Str 6, flips queue order |
| Parrot | 0.3857 | 0.2146 | 100 | Str 2, removes cards |
| Snake | 0.3496 | 0.1893 | 30 | Str 9, reorders queue |
| Lion | 0.3474 | 0.1450 | 66 | Highest strength (12), jumps to front |
| Kangaroo | 0.3176 | 0.1307 | 98 | Str 3, hops forward |
| Chameleon | 0.3083 | 0.1699 | 80 | Str 5, copies abilities |

**Highest attention species**: Crocodile, Monkey, Skunk

### Position Attention When Specific Species Present

How does the model's attention distribution change based on which species are in the queue?

#### When Lion is in queue:

| Position | Avg Attention |
|----------|--------------|
| 0 | 0.3463 |
| 1 | 0.3227 |
| 2 | 0.2679 |
| 3 | 0.2673 |

#### When Monkey is in queue:

| Position | Avg Attention |
|----------|--------------|
| 0 | 0.5005 |
| 1 | 0.3128 |
| 2 | 0.2204 |
| 3 | 0.2926 |

#### When Parrot is in queue:

| Position | Avg Attention |
|----------|--------------|
| 0 | 0.4034 |
| 1 | 0.3443 |
| 2 | 0.2451 |
| 3 | 0.2859 |

#### When Snake is in queue:

| Position | Avg Attention |
|----------|--------------|
| 0 | 0.3789 |
| 1 | 0.3167 |
| 2 | 0.2097 |
| 3 | 0.2629 |

#### When Seal is in queue:

| Position | Avg Attention |
|----------|--------------|
| 0 | 0.4471 |
| 1 | 0.3705 |
| 2 | 0.2002 |
| 3 | 0.2727 |

### Attention Patterns by Queue Length

**Queue length 1**:
  - Position 0: 1.0000

**Queue length 2**:
  - Position 0: 0.4912
  - Position 1: 0.5088

**Queue length 3**:
  - Position 0: 0.3482
  - Position 1: 0.3100
  - Position 2: 0.3418

**Queue length 4**:
  - Position 0: 0.2864
  - Position 1: 0.2942
  - Position 2: 0.1483
  - Position 3: 0.2710

## Strategic Interpretation

Based on the attention patterns, we can infer the model's strategic priorities:


1. **Scoring-Focused Attention**: The model pays significantly more attention to cards near the front of the queue. This aligns with the game's scoring mechanism - when the queue fills to 5 cards, positions 0-1 enter the Beasty Bar (scoring zone) while position 4 bounces to THAT'S IT (penalty zone). The model has learned that front positions are strategically more important.

2. **Balanced Awareness**: The model distributes attention roughly equally between own and opponent cards. This indicates a balanced strategy that considers both offensive (optimizing own scores) and defensive (tracking opponent positions) factors.

3. **Species-Specific Attention Patterns**:
   - **Crocodile** (recurring eater) gets high attention, likely because its recurring ability can eliminate cards each turn
   - **Skunk** (mass removal) draws attention due to its ability to eliminate multiple high-strength cards
   - **Monkey** (pair-based removal) gets attention for its ability to remove high-strength cards

### Key Strategic Insights

The attention patterns reveal several aspects of the model's learned strategy:

1. **Position Awareness**: The model has learned the importance of queue position, 
   paying more attention to cards that will score soon (front) versus those that 
   might be bounced (back).

2. **Recurring Species Importance**: Cards with recurring abilities (Crocodile, Hippo, 
   Giraffe) receive notable attention because they act every turn and can repeatedly 
   affect the queue state.

3. **Threat Assessment**: High-impact cards like Skunk and Crocodile receive extra 
   attention, suggesting the model has learned to track potential threats that could 
   dramatically change the game state.

## Methodology

- Generated 200+ diverse game states by playing random moves from initial states
- Captured attention weights from the TransformerEncoder in the queue_encoder
- Averaged attention across all 4 transformer layers and 8 attention heads
- Analyzed attention received by each queue position (column-wise sums)
- Normalized attention to create comparable distributions

## Model Architecture

- Queue encoder: TransformerEncoder with 4 layers, 8 heads, 256-dim hidden
- Uses positional encoding to capture card order in queue
- Pre-LN (layer normalization before attention) for training stability
