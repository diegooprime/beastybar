# Neural MCTS Implementation for Beasty Bar

This package provides a production-ready Monte Carlo Tree Search implementation with neural network integration following the AlphaZero approach.

## Architecture

### Core Components

1. **`search.py`** - Main MCTS implementation
   - `MCTS` class: Tree search algorithm with neural network guidance
   - `MCTSNode` dataclass: Tree nodes with UCB scoring

2. **`agent.py`** - Agent wrapper
   - `MCTSAgent` class: Full Agent implementation for game playing
   - Temperature control, policy extraction, and training integration

3. **`simple_node.py`** - Standalone node implementation
   - `SimpleMCTSNode` class: Works with Action-based priors (non-neural)
   - Useful for understanding MCTS mechanics

## Key Features

### AlphaZero-Style Search
- **Policy Priors**: Neural network provides action probabilities to guide search
- **Value Estimates**: Network evaluates positions instead of rollouts
- **PUCT Selection**: UCB with policy priors for exploration-exploitation balance
- **Dirichlet Noise**: Exploration at root for training diversity

### Production-Ready
- **Action Masking**: Automatic legal action enforcement
- **Temperature Scheduling**: Control exploration/exploitation during training
- **Device Management**: Automatic GPU/CPU detection
- **Batch Processing**: Efficient network evaluation

### Training Integration
- **Policy Extraction**: Get visit count distributions for supervised learning
- **Deterministic Play**: Evaluation mode for reproducible results
- **Hyperparameter Tuning**: Configurable simulation count, exploration constants

## Usage

### Basic Usage

```python
from _02_agents.mcts import MCTSAgent
from _02_agents.neural.network import BeastyBarNetwork
from _01_simulator import state, engine

# Create network and agent
network = BeastyBarNetwork()
agent = MCTSAgent(network, num_simulations=400)

# Play a game
game_state = state.initial_state(seed=42)
legal = list(engine.legal_actions(game_state, 0))
action = agent.select_action(game_state, legal)
```

### Training Integration

```python
# Get policy distribution for training
policy = agent.get_policy(game_state)

# Temperature scheduling (high early, low late)
agent.set_temperature(1.5)  # Exploration phase
agent.set_temperature(0.3)  # Exploitation phase

# Deterministic evaluation
action = agent.select_action_deterministic(game_state, legal)
```

### Custom Configuration

```python
from _02_agents.neural.utils import NetworkConfig

# Create smaller network
config = NetworkConfig(hidden_dim=64, num_heads=2, num_layers=1)
network = BeastyBarNetwork(config)

# Configure MCTS
agent = MCTSAgent(
    network=network,
    num_simulations=200,
    c_puct=1.5,              # Exploration constant
    dirichlet_alpha=0.3,     # Noise concentration
    dirichlet_epsilon=0.25,  # Noise weight
    temperature=1.0,         # Action sampling temperature
)
```

## Algorithm Details

### MCTS Phases

1. **SELECT**: Walk tree using PUCT until unexpanded node or terminal state
2. **EXPAND**: Evaluate position with neural network (policy + value)
3. **BACKUP**: Propagate value up tree, alternating sign for opponent

### PUCT Formula

```
Score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
```

Where:
- `Q(a)` = mean value of child node (exploitation)
- `P(a)` = policy prior from network
- `N(a)` = visit count of child
- `N_parent` = visit count of parent
- `c_puct` = exploration constant (typically 1.5)

### Dirichlet Noise

Applied at root for exploration during training:

```
P'(a) = (1 - ε) * P(a) + ε * η_a
```

Where `η ~ Dir(α)` with `α = 0.3` for Beasty Bar

## Hyperparameters

### Recommended Settings

| Phase | Simulations | Temperature | c_puct | Dirichlet ε |
|-------|-------------|-------------|--------|-------------|
| Training | 400-800 | 1.0 | 1.5 | 0.25 |
| Evaluation | 800-1600 | 0.0-0.1 | 1.0 | 0.0 |
| Fast Play | 100-200 | 1.0 | 2.0 | 0.25 |

### Parameter Effects

- **num_simulations**: More simulations = stronger play but slower
- **temperature**: Higher = more exploration, lower = greedy
- **c_puct**: Higher = more exploration, lower = more exploitation
- **dirichlet_epsilon**: Higher = more noise, more exploration

## Performance

### Network Inference
- Default network: ~1.3M parameters
- Inference time: ~5-10ms per position (CPU)
- Batch processing supported for efficiency

### MCTS Speed
- 100 simulations: ~0.5-1 second
- 400 simulations: ~2-4 seconds
- 1600 simulations: ~8-15 seconds

Times depend on hardware and network size.

## Testing

Comprehensive test suite in `_05_other/tests/test_mcts_neural.py`:

```bash
pytest _05_other/tests/test_mcts_neural.py -v
```

Tests cover:
- Basic search functionality
- Agent action selection
- Deterministic play
- Temperature control
- Policy extraction
- Terminal state handling
- Device management
- Dirichlet noise

## Examples

See `_05_other/examples/mcts_neural_example.py` for:
- Basic usage
- Deterministic play
- Policy extraction
- Hyperparameter tuning
- Custom network configuration
- Temperature scheduling
- Direct MCTS usage

Run examples:

```bash
python _05_other/examples/mcts_neural_example.py
```

## Integration with Training

This MCTS implementation is designed for self-play training:

1. **Generate Games**: Use MCTSAgent with high temperature for diverse play
2. **Extract Policies**: Get visit distributions with `get_policy()`
3. **Train Network**: Supervised learning on (state, policy, value) tuples
4. **Iterate**: Improved network → stronger MCTS → better training data

See `_03_training/` for training pipeline integration.

## Comparison with Simple MCTS

| Feature | Neural MCTS | Simple MCTS |
|---------|-------------|-------------|
| Policy | Neural network | Uniform/manual |
| Evaluation | Network value | Rollouts |
| Speed | Fast (no rollouts) | Slower |
| Strength | Learns from data | Fixed heuristics |
| Use Case | Training/competition | Baseline/testing |

## References

- Silver et al., "Mastering the game of Go with deep neural networks and tree search" (2016)
- Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017)

## Implementation Notes

### Action Encoding
Actions are converted to integer indices (0-123) for network compatibility:
- Observation → 988-dim tensor
- Policy logits → 124-dim tensor (one per action)
- Action mask → 124-dim binary tensor

### Value Range
Network outputs values in [-1, 1]:
- +1 = guaranteed win
- -1 = guaranteed loss
- 0 = even position

Terminal states include margin bonus: ±1.0 + 0.2 * (margin / 36)

### Device Handling
Network automatically placed on best available device:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon)
3. CPU (fallback)

## Future Enhancements

Potential improvements:
- [ ] Virtual loss for parallel MCTS
- [ ] UCB-Tuned for better exploration
- [ ] Progressive widening for large action spaces
- [ ] MCTS with prior network uncertainty
- [ ] Transposition table for repeated states
