# Neural Network

Transformer-based policy-value network for Beasty Bar.

## Architecture

> **Note:** The values below reflect the **trained model** configuration
> (see `configs/iter600_to_1000.yaml`). Code defaults in `NetworkConfig`
> may differ (e.g., `species_embedding_dim` defaults to 32, not 64).

```
Input (988) → Zone Encoders → Fusion → Policy (124) + Value (1)
```

**Input parsing (988 dims):**
- Queue: 5 slots × 17 features (order matters → TransformerEncoder)
- Beasty Bar: 24 slots × 17 features (unordered → SetTransformer)
- That's It: 24 slots × 17 features
- Own hand: 4 cards × 17 features
- Opponent hand: 4 slots × 3 features (masked)
- Scalars: 7 dims

**Card encoding:**
- Species embedding: 12 species → 64-dim learned vectors
- Feature projection: 5 numeric features → 64-dim
- Combined: 128-dim per card

**Zone encoders:**
- TransformerEncoder: For queue (position matters)
- SetTransformer: For bar/hand (permutation invariant)
- 4 attention heads, 128-dim hidden, GELU activation

**Fusion:** 3 blocks combining all zone encodings (768 → 128)

**Output heads:**
- Policy: Linear(128→64) → GELU → Linear(64→124) → logits
- Value: Linear(128→64) → GELU → Linear(64→1) → Tanh → [-1, 1]

## Usage

```python
from _02_agents.neural.network import BeastyBarNetwork
from _02_agents.neural.agent import NeuralAgent

network = BeastyBarNetwork()
agent = NeuralAgent(network)

action = agent.select_action(state, legal_actions)
```

## Parameters

~1.3M total:
- Card encoders: ~50K
- Zone transformers: ~700K
- Fusion blocks: ~200K
- Heads: ~20K
