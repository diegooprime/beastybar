# Neural Network Architecture

This is a **Policy-Value Network** (Actor-Critic style) built with PyTorch for the Beasty Bar card game. It takes a game state and outputs:
1. **Policy**: 124 action logits (which card to play)
2. **Value**: A single number in [-1, 1] estimating how good the position is

## The Big Picture

```
Input (988 floats) → Parse into zones → Encode each zone → Fuse together → Policy + Value heads
```

## Architecture Components

### 1. Input Parsing

The 988-dimensional input is split into game zones:

| Zone | Shape | Description |
|------|-------|-------------|
| Queue | 5×17 | Cards waiting to enter bar (order matters) |
| Beasty Bar | 24×17 | Cards in the bar (order doesn't matter) |
| That's It | 24×17 | Eliminated cards |
| Own Hand | 4×17 | Your cards |
| Opponent Hand | 4×3 | Opponent cards (masked - only presence info) |
| Scalars | 7 | Game state numbers (scores, turns, etc.) |

### 2. Card Encoder

Converts raw card features → learned embeddings:

```
Card (17 features) → Species Embedding + Feature Projection → Combined (128 dim)
```

**Key neurons:**
- **Species Embedding**: Lookup table mapping 12 species → 32-dim vectors (learned)
- **Feature Projection**: Linear layer projecting 5 numeric features → 32-dim
- **Combine Layer**: Linear(64 → 128) to merge species + features

**Activations**: GELU + LayerNorm + Dropout

### 3. Zone Encoders - The Transformers

Two types handle different semantics:

#### a) TransformerZoneEncoder - For Queue
- Uses **positional encoding** (sinusoidal) because order matters in the queue
- Standard Transformer encoder with self-attention
- 4 attention heads, 1 layer
- Outputs single 128-dim vector via masked mean pooling

#### b) SetTransformerEncoder - For Bar/Hand
- **NO positional encoding** - treats cards as an unordered set
- Same Transformer architecture but permutation-invariant
- Cards can be in any order and get the same representation

**Transformer internals:**
- `d_model`: 128 (hidden dim)
- `nhead`: 4 attention heads (each sees 32-dim slice)
- `dim_feedforward`: 512 (4× hidden)
- Activation: GELU
- Pre-LayerNorm (more stable training)

### 4. Positional Encoding

Classic sinusoidal encoding from "Attention is All You Need":

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This lets the model know "this card is 1st in queue" vs "3rd in queue".

### 5. Fusion Blocks

After encoding all zones, we have 6 vectors of 128-dim each (768 total).

Three **FusionBlock** layers compress and mix this:

```
[768] → FusionBlock → [128] → FusionBlock → [128] → FusionBlock → [128]
```

Each block is:
```
Input → Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm → GELU → Dropout → + Residual
```

### 6. Output Heads

#### Policy Head
```
128 → Linear(128→64) → GELU → Dropout → Linear(64→124) → raw logits
```
Outputs 124 action scores (masked externally for legal moves)

#### Value Head
```
128 → Linear(128→64) → GELU → Dropout → Linear(64→1) → Tanh
```
Outputs single value in **[-1, 1]** (Tanh squashes it)

## Activation Functions Used

| Function | Where | Why |
|----------|-------|-----|
| **GELU** | Everywhere (hidden layers) | Smoother than ReLU, works well with Transformers |
| **Tanh** | Value head output only | Constrains value to [-1, 1] range |
| **Softmax** | Applied externally to policy | Converts logits to probabilities |

**GELU** (Gaussian Error Linear Unit):
```
GELU(x) = x * Φ(x)  where Φ is the cumulative normal distribution
```
It's like ReLU but smoother - small negative values aren't completely zeroed.

## Weight Initialization

- **Linear layers**: Xavier Uniform (good for symmetric activations)
- **Embeddings**: Normal(0, 0.02)
- **LayerNorm**: weights=1, bias=0

## Parameter Count

~1.3 million parameters total, broken down:
- Card encoders: ~50K
- Zone transformers (5): ~700K
- Fusion blocks (3): ~200K
- Policy/Value heads: ~20K

## Data Flow Summary

```
Observation (988)
    │
    ▼
Parse into 6 zones
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Queue(5×17) → CardEncoder → TransformerZoneEncoder → 128│
│ Bar(24×17)  → CardEncoder → SetTransformerEncoder  → 128│
│ ThatsIt     → CardEncoder → SetTransformerEncoder  → 128│
│ Hand        → CardEncoder → SetTransformerEncoder  → 128│
│ OppHand(4×3)→ MaskedEncoder→ SetTransformerEncoder→ 128│
│ Scalars(7)  → Linear+GELU                          → 128│
└─────────────────────────────────────────────────────────┘
    │
    ▼ Concatenate (768)
    │
    ▼
Fusion1 → Fusion2 → Fusion3 (128)
    │
    ▼
┌──────────────────┬──────────────────┐
│ Policy Head      │ Value Head       │
│ → 124 logits     │ → 1 value [-1,1] │
└──────────────────┴──────────────────┘
```

## Design Rationale

The clever design uses **order-aware** Transformers for the queue (position matters) and **order-invariant** Set Transformers for hands/bar (cards are interchangeable). This inductive bias helps the network learn faster by encoding domain knowledge about the game structure directly into the architecture.
