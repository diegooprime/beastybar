# Neural MCTS for Beasty Bar

Production-ready Monte Carlo Tree Search with neural network integration following the AlphaZero approach.

## Architecture

```
_02_agents/mcts/
├── search.py       # Core MCTS with PUCT selection
├── batch_mcts.py   # Batched evaluation with virtual loss
├── agent.py        # MCTSAgent wrapper
└── simple_node.py  # Standalone node (non-neural)
```

## Key Features

### AlphaZero-Style Search
- **Policy priors**: Neural network guides action selection
- **Value estimates**: Network evaluates positions (no rollouts)
- **PUCT selection**: UCB with policy priors for exploration-exploitation
- **Dirichlet noise**: Root exploration for training diversity

### BatchMCTS (Implemented)
- Batched neural network evaluation across parallel search trees
- Virtual loss mechanism prevents path convergence in parallel selection
- 5-10x speedup over sequential MCTS
- See `BATCH_MCTS.md` for details

### Production-Ready
- Automatic legal action masking
- Temperature scheduling for exploration control
- Device auto-detection (CUDA > MPS > CPU)
- Efficient batch processing

## Usage

### Basic Usage

```python
from _02_agents.mcts import MCTSAgent
from _02_agents.neural.network import BeastyBarNetwork
from _01_simulator import state, engine

# Create network and agent
network = BeastyBarNetwork()
agent = MCTSAgent(network, num_simulations=200)

# Play
game_state = state.initial_state(seed=42)
legal = list(engine.legal_actions(game_state, 0))
action = agent.select_action(game_state, legal)
```

### BatchMCTS for Training

```python
from _02_agents.mcts import BatchMCTS

batch_mcts = BatchMCTS(
    network,
    num_simulations=200,
    batch_size=16,
    virtual_loss=3.0,
)

# Search multiple states in parallel
states = [state.initial_state(seed=i) for i in range(32)]
distributions = batch_mcts.search_batch(states, perspective=0)
```

### Training Integration

```python
# Get policy distribution for training
policy = agent.get_policy(game_state)

# Temperature scheduling
agent.set_temperature(1.0)  # Exploration phase
agent.set_temperature(0.1)  # Exploitation phase

# Deterministic evaluation
action = agent.select_action_deterministic(game_state, legal)
```

## PUCT Formula

```
Score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
```

Where:
- `Q(a)` = mean value of child node (exploitation)
- `P(a)` = policy prior from network
- `N(a)` = visit count of child
- `N_parent` = visit count of parent
- `c_puct` = exploration constant (default: 2.0)

## Dirichlet Noise

Applied at root for exploration during training:

```
P'(a) = (1 - ε) * P(a) + ε * η_a
```

Where `η ~ Dir(α)` with `α = 0.5` for Beasty Bar

## Hyperparameters

### Recommended Settings

| Phase | Simulations | Temperature | c_puct | Dirichlet ε |
|-------|-------------|-------------|--------|-------------|
| Training | 200-400 | 1.0 | 2.0 | 0.4 |
| Evaluation | 400-800 | 0.0-0.1 | 1.5 | 0.0 |
| Fast Play | 50-100 | 1.0 | 2.0 | 0.25 |

### Parameter Effects

- **num_simulations**: More = stronger but slower
- **temperature**: Higher = more exploration
- **c_puct**: Higher = more exploration of novel actions
- **dirichlet_epsilon**: Higher = more root noise
- **virtual_loss**: Higher = more diverse parallel exploration

## Performance

### Sequential MCTS
- 100 simulations: ~0.5-1 second
- 400 simulations: ~2-4 seconds

### BatchMCTS (batch_size=16)
- 5-10x speedup over sequential
- ~0.17s per game with 200 simulations

## Integration with Training

The MCTS implementation supports AlphaZero-style self-play training:

1. **Generate games**: Use MCTSAgent with high temperature
2. **Extract policies**: Get visit distributions with `get_policy()`
3. **Train network**: Supervised learning on (state, policy, value) tuples
4. **Iterate**: Improved network → stronger MCTS → better training data

See `_03_training/MCTS_TRAINING.md` for training pipeline details.

## Value Range

Network outputs values in [-1, 1]:
- +1 = guaranteed win
- -1 = guaranteed loss
- 0 = even position

Terminal states include margin bonus: ±1.0 + 0.2 * (margin / 36)

## Testing

```bash
pytest _05_other/tests/test_mcts_neural.py -v
pytest _05_other/tests/test_batch_mcts.py -v
```

## References

- [AlphaZero Paper (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero Paper (Silver et al., 2017)](https://www.nature.com/articles/nature24270)
