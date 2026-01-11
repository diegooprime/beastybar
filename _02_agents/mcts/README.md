# MCTS for Beasty Bar

AlphaZero-style Monte Carlo Tree Search with neural network guidance.

## Usage

```python
from _02_agents.mcts import MCTSAgent, BatchMCTS
from _02_agents.neural.network import BeastyBarNetwork
from _01_simulator import state, engine

# Create agent
network = BeastyBarNetwork()
agent = MCTSAgent(network, num_simulations=200)

# Play
game_state = state.initial_state(seed=42)
legal = list(engine.legal_actions(game_state, 0))
action = agent.select_action(game_state, legal)
```

## BatchMCTS (5-10x faster)

Batched leaf evaluation with virtual loss for parallel exploration.

```python
batch_mcts = BatchMCTS(
    network,
    num_simulations=200,
    batch_size=16,
    virtual_loss=3.0,
)

states = [state.initial_state(seed=i) for i in range(32)]
distributions = batch_mcts.search_batch(states, perspective=0)
```

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_simulations` | 200 | Search depth (more = stronger, slower) |
| `c_puct` | 2.0 | Exploration constant |
| `temperature` | 1.0 | Action sampling (0 = greedy) |
| `dirichlet_alpha` | 0.5 | Root noise concentration |
| `dirichlet_epsilon` | 0.4 | Root noise mixing |
| `virtual_loss` | 3.0 | Parallel exploration diversity |
| `batch_size` | 8 | Leaves per batch evaluation |

## PUCT Formula

```
Score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
```

- `Q(a)` — mean value (exploitation)
- `P(a)` — policy prior from network
- `N(a)` — visit count

## Files

| File | Purpose |
|------|---------|
| `search.py` | Core MCTS with PUCT selection |
| `batch_mcts.py` | Batched evaluation with virtual loss |
| `agent.py` | MCTSAgent wrapper |

## References

- [AlphaZero (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero (Silver et al., 2017)](https://www.nature.com/articles/nature24270)
