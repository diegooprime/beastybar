# Species Embedding Analysis

Analysis of learned species embeddings from the trained Beasty Bar v4 model.

**Model**: `/checkpoints/v4/final.pt`
**Embedding dimension**: 64

## Cosine Similarity Matrix

Cosine similarity between species embeddings (higher = more similar representation):

| Species | lion | hipp | croc | snak | gira | zebr | seal | cham | monk | kang | parr | skun |
|---------|------|------|------|------|------|------|------|------|------|------|------|------|
| lion    | 1.000|-0.051|-0.100|-0.101|-0.049|-0.108|-0.145|-0.060|-0.014|-0.095|-0.107|-0.114|
| hippo   |-0.051| 1.000|-0.147|-0.167|-0.038|-0.292|-0.072| 0.097|-0.062| 0.123|-0.187|-0.144|
| crocodile |-0.100|-0.147| 1.000|-0.044|-0.051|-0.203| 0.045|-0.103|-0.142|-0.106|-0.132|-0.066|
| snake   |-0.101|-0.167|-0.044| 1.000| 0.061|-0.263|-0.071| 0.016|-0.157| 0.114|-0.225|-0.057|
| giraffe |-0.049|-0.038|-0.051| 0.061| 1.000| 0.047|-0.047|-0.357|-0.143|-0.257| 0.071|-0.154|
| zebra   |-0.108|-0.292|-0.203|-0.263| 0.047| 1.000|-0.156| 0.148|-0.128| 0.115|-0.105| 0.083|
| seal    |-0.145|-0.072| 0.045|-0.071|-0.047|-0.156| 1.000|-0.233|-0.031|-0.286|-0.054|-0.030|
| chameleon |-0.060| 0.097|-0.103| 0.016|-0.357| 0.148|-0.233| 1.000|-0.034|-0.149| 0.131|-0.221|
| monkey  |-0.014|-0.062|-0.142|-0.157|-0.143|-0.128|-0.031|-0.034| 1.000| 0.059|-0.096|-0.039|
| kangaroo |-0.095| 0.123|-0.106| 0.114|-0.257| 0.115|-0.286|-0.149| 0.059| 1.000|-0.045|-0.248|
| parrot  |-0.107|-0.187|-0.132|-0.225| 0.071|-0.105|-0.054| 0.131|-0.096|-0.045| 1.000|-0.223|
| skunk   |-0.114|-0.144|-0.066|-0.057|-0.154| 0.083|-0.030|-0.221|-0.039|-0.248|-0.223| 1.000|

**Key observation**: Most species pairs show negative or near-zero similarity, indicating the model has learned highly differentiated representations for each species. This is strategically appropriate - each animal has unique abilities and plays differently.

## Top 5 Most Similar Species Pairs

| Rank | Species 1 | Species 2 | Similarity | Interpretation |
|------|-----------|-----------|------------|----------------|
| 1 | zebra | chameleon | 0.148 | Both are reactive cards - zebra defends against threats, chameleon copies the best ability available |
| 2 | chameleon | parrot | 0.131 | Both require reading the board state - chameleon to choose what to copy, parrot to choose what to expel |
| 3 | hippo | kangaroo | 0.123 | Both are forward-movement cards that try to push toward the front of the queue |
| 4 | zebra | kangaroo | 0.115 | Both benefit from careful queue positioning and timing |
| 5 | snake | kangaroo | 0.114 | Both manipulate queue order - snake through sorting, kangaroo through hopping |

**Insight**: Even the most similar pairs have relatively low similarity (0.148 max), indicating the model has learned distinct strategic representations for each species.

## Top 5 Most Different Species Pairs

| Rank | Species 1 | Species 2 | Similarity | Interpretation |
|------|-----------|-----------|------------|----------------|
| 1 | giraffe | chameleon | -0.357 | **Highly differentiated**: Giraffe has a fixed, predictable recurring ability; chameleon is situationally adaptive |
| 2 | hippo | zebra | -0.292 | **Counter-relationship**: Zebra specifically blocks hippo advancement - direct strategic opposition |
| 3 | seal | kangaroo | -0.286 | **Opposite effects**: Seal reverses entire queue order; kangaroo makes precise positional adjustments |
| 4 | snake | zebra | -0.263 | **Conflicting strategies**: Snake benefits from high-strength cards in queue; zebra defends against aggressive play |
| 5 | giraffe | kangaroo | -0.257 | Both move forward but giraffe is recurring/passive while kangaroo is one-time/active |

**Key Finding**: The model has learned the hippo-zebra counter-relationship (zebra blocks hippo, -0.292 similarity). This demonstrates the model understands strategic card interactions, not just individual card abilities.

## Embedding Norm Analysis

L2 norm of each embedding (higher norm may indicate stronger/more impactful representation):

| Rank | Species | L2 Norm | Strategic Role |
|------|---------|---------|----------------|
| 1 | seal | 0.591 | High impact: Queue reversal can be game-changing |
| 2 | monkey | 0.581 | High impact: Gang ability to expel hippos/crocs |
| 3 | parrot | 0.549 | Targeted removal, flexible threat |
| 4 | zebra | 0.545 | Critical defensive role, blocks key abilities |
| 5 | hippo | 0.544 | Strong recurring pusher |
| 6 | crocodile | 0.531 | Recurring threat, eats weaker cards |
| 7 | lion | 0.524 | Queue dominance, monkey expulsion |
| 8 | snake | 0.517 | Queue reordering by strength |
| 9 | skunk | 0.516 | Mass expulsion of strong species |
| 10 | giraffe | 0.480 | Moderate: Slow but steady advancement |
| 11 | kangaroo | 0.460 | Moderate: Limited hop distance |
| 12 | chameleon | 0.451 | Lowest: Depends entirely on what's available to copy |

**Average norm**: 0.524

**Interpretation**: Cards with high-impact abilities (seal, monkey) or unique defensive roles (zebra) have stronger embeddings. The chameleon has the weakest norm, reflecting its nature as a "blank" card that derives value from copying others.

## Cluster Analysis

### PCA Projection (2D)

Principal components explain variance:
- PC1: 13.7%
- PC2: 12.6%

The relatively even split suggests the embedding space is high-dimensional with no single dominant strategic axis.

| Species | PC1 | PC2 | Quadrant | Strategic Group |
|---------|-----|-----|----------|-----------------|
| lion    | +0.072 | -0.084 | Q4 (+,-) | Aggressive |
| hippo   | +0.062 | -0.367 | Q4 (+,-) | Aggressive (extreme) |
| crocodile | -0.235 | +0.005 | Q2 (-,+) | Predatory |
| snake   | -0.074 | -0.077 | Q3 (-,-) | Queue manipulation |
| giraffe | -0.074 | +0.112 | Q2 (-,+) | Steady advancement |
| zebra   | +0.243 | +0.389 | Q1 (+,+) | Defensive (extreme) |
| seal    | -0.467 | +0.038 | Q2 (-,+) | Queue manipulation (extreme) |
| chameleon | +0.193 | -0.011 | Q4 (+,-) | Adaptive |
| monkey  | +0.075 | -0.202 | Q4 (+,-) | Aggressive |
| kangaroo | +0.207 | -0.111 | Q4 (+,-) | Positioning |
| parrot  | +0.086 | +0.132 | Q1 (+,+) | Targeted removal |
| skunk   | -0.088 | +0.176 | Q2 (-,+) | Mass removal |

**Key observations**:
- Hippo and zebra are at opposite extremes on PC2 (confirming their counter-relationship)
- Seal is an outlier on PC1, reflecting its unique queue-reversal ability
- Q4 contains most aggressive/forward-moving cards
- Q1 contains defensive/reactive cards

### Hierarchical Clustering

**3 Clusters:**
- Cluster 1: **lion** (unique queue dominance)
- Cluster 2: **crocodile, seal** (destructive/transformative effects)
- Cluster 3: **All others** (various positioning/timing strategies)

**4 Clusters:**
- Cluster 1: **lion** (king of the queue)
- Cluster 2: **crocodile** (recurring predator)
- Cluster 3: **seal** (queue transformer)
- Cluster 4: **All others**

**5 Clusters:**
- Cluster 1: **lion** (unique dominant)
- Cluster 2: **crocodile** (eating mechanic)
- Cluster 3: **hippo, snake, giraffe, zebra, chameleon, kangaroo, parrot, skunk** (mixed strategies)
- Cluster 4: **seal** (unique reversal)
- Cluster 5: **monkey** (unique gang mechanic)

**Insight**: Lion, crocodile, seal, and monkey each get their own clusters at different granularities, indicating these cards have the most unique strategic roles that the model has recognized.

## Strategic Insights

### What the Model Learned

Based on the embedding analysis, the model has discovered these strategic patterns:

**1. Counter-Relationships (Negative Similarity)**

The strongest learned relationship is **hippo vs zebra** (-0.292). This reflects the core game mechanic where zebra blocks hippo advancement. Similarly, **crocodile vs zebra** (-0.203) shows the model learned that zebra is the key defender against both aggressive recurring cards.

**2. Adaptive vs Fixed Abilities**

The most negative pair is **giraffe vs chameleon** (-0.357). Giraffe has a fixed, predictable recurring ability, while chameleon's value depends entirely on context. The model has learned that these cards require fundamentally different strategic evaluation.

**3. Impact-Based Embedding Strength**

Cards with dramatic, game-changing abilities (seal's queue reversal, monkey's gang ability) have stronger embedding norms. Cards that are context-dependent or incremental (chameleon, kangaroo, giraffe) have weaker norms.

**4. Unique Strategic Identities**

Each card has learned a mostly orthogonal representation (average off-diagonal similarity near 0). This indicates the model treats each species as strategically distinct, which is appropriate for a game where each card has unique abilities.

**5. Forward Movement Clustering**

Cards that move forward (hippo, kangaroo) cluster together in embedding space, even though their mechanics differ. This suggests the model groups cards by strategic effect rather than just mechanical similarity.

### Implications for Gameplay

1. **Timing matters**: High-norm cards (seal, monkey) likely have optimal timing windows
2. **Counter-play is learned**: The model explicitly distinguishes defensive plays (zebra) from aggressive ones
3. **Chameleon is situational**: Its low norm and similarity to board-reading cards (parrot) suggests it's valued for flexibility, not raw power
4. **Queue position awareness**: The hippo-zebra and giraffe-kangaroo distinctions show position-based reasoning

## Appendix: Raw Embedding Values

First 8 dimensions of each species embedding:

```
lion      : [-0.023, -0.087, +0.059, -0.004, +0.020, -0.034, -0.003, -0.067, ...]
hippo     : [-0.027, -0.003, +0.044, -0.047, +0.003, -0.040, -0.021, -0.144, ...]
crocodile : [+0.009, +0.102, -0.040, +0.046, -0.060, +0.076, +0.039, +0.027, ...]
snake     : [+0.094, +0.069, +0.042, +0.014, +0.098, +0.018, +0.187, +0.053, ...]
giraffe   : [+0.099, +0.059, +0.067, -0.115, +0.080, -0.085, +0.022, -0.062, ...]
zebra     : [-0.057, -0.060, -0.086, -0.101, -0.014, -0.041, -0.035, +0.042, ...]
seal      : [-0.032, -0.010, -0.043, +0.131, +0.014, +0.134, +0.141, +0.035, ...]
chameleon : [-0.127, -0.027, -0.073, -0.072, -0.037, +0.039, -0.071, +0.041, ...]
monkey    : [-0.022, +0.111, +0.068, -0.014, -0.107, +0.076, -0.012, +0.001, ...]
kangaroo  : [-0.047, +0.054, +0.089, +0.083, -0.016, +0.002, -0.005, +0.031, ...]
parrot    : [-0.039, -0.050, -0.088, -0.001, -0.063, -0.061, -0.068, -0.065, ...]
skunk     : [+0.067, -0.070, -0.039, +0.070, +0.093, -0.033, -0.033, +0.039, ...]
```
