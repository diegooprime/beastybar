"""Neural network agents for Beasty Bar.

This package provides neural network-based agents for playing Beasty Bar,
including model architecture, utilities, and trained agent implementations.

Modules:
    utils: Network utilities (checkpointing, action sampling, configuration)
    network: Neural network architecture (V1)
    network_v2: Enhanced architecture with asymmetric encoders (V2)
    agent: NeuralAgent class for policy-based action selection
    compile: Torch compile utilities for inference speedup
"""

from _02_agents.neural.agent import (
    InferenceMode,
    NeuralAgent,
    create_neural_agent,
    load_neural_agent,
)
from _02_agents.neural.compile import (
    compile_for_inference,
    compile_for_training,
    is_torch_compile_available,
    maybe_compile_network,
)
from _02_agents.neural.network import BeastyBarNetwork, create_network
from _02_agents.neural.network_v2 import (
    BeastyBarNetworkV2,
    NetworkConfigV2,
    create_network_v2,
)
from _02_agents.neural.utils import (
    ACTION_DIM,
    OBSERVATION_DIM,
    CheckpointData,
    NetworkConfig,
    batch_greedy_actions,
    batch_sample_actions,
    compute_action_probs,
    count_parameters,
    default_config,
    get_device,
    get_model_summary,
    greedy_action,
    load_checkpoint,
    load_network_from_checkpoint,
    move_to_device,
    sample_action,
    save_checkpoint,
    seed_all,
)

__all__ = [
    # Network V1
    "ACTION_DIM",
    "OBSERVATION_DIM",
    "BeastyBarNetwork",
    "CheckpointData",
    # Network V2
    "BeastyBarNetworkV2",
    "NetworkConfigV2",
    "create_network_v2",
    # Agent
    "InferenceMode",
    "NetworkConfig",
    "NeuralAgent",
    "batch_greedy_actions",
    "batch_sample_actions",
    "compile_for_inference",
    "compile_for_training",
    "compute_action_probs",
    "count_parameters",
    "create_network",
    "create_neural_agent",
    "default_config",
    "get_device",
    "get_model_summary",
    "greedy_action",
    "is_torch_compile_available",
    "load_checkpoint",
    "load_network_from_checkpoint",
    "load_neural_agent",
    "maybe_compile_network",
    "move_to_device",
    "sample_action",
    "save_checkpoint",
    "seed_all",
]
