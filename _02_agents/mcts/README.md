# MCTS

Experimental. Not used in training—PPO was used instead.

AlphaZero-style tree search with neural network guidance. Works but slower than PPO for this game.

## Usage

```python
from _02_agents.mcts import MCTSAgent
from _02_agents.neural.network import BeastyBarNetwork

network = BeastyBarNetwork()
agent = MCTSAgent(network, num_simulations=200)
action = agent.select_action(state, legal_actions)
```

## Files

| File | Purpose |
|------|---------|
| `search.py` | MCTS with PUCT |
| `batch_mcts.py` | Batched evaluation |
| `agent.py` | Agent wrapper |
