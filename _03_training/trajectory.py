"""Trajectory step dataclasses for self-play training.

This module consolidates the various step dataclasses used across training modules
into a clean hierarchy:

- PPOStep: Finalized PPO training step with reward and done flag
- PPOPendingStep: In-progress PPO step before reward assignment
- MCTSStep: Finalized MCTS training step with value target
- MCTSPendingStep: In-progress MCTS step before value assignment

Backward compatibility aliases:
- TrajectoryStep = PPOStep
- PendingStep = PPOPendingStep
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# PPO Training Steps
# ============================================================================


@dataclass(frozen=True, slots=True)
class PPOStep:
    """Single finalized step in a player's PPO training trajectory.

    This is the final training data format for PPO, containing all information
    needed for policy gradient updates including the reward signal.

    Attributes:
        observation: State observation tensor of shape (OBSERVATION_DIM,).
        action_mask: Legal action mask of shape (ACTION_DIM,).
        action: Selected action index.
        action_prob: Probability of selected action under policy.
        value: Value estimate from critic at this state.
        reward: Reward received (set after game ends).
        done: Whether this is the last step of the trajectory.
    """

    observation: NDArray[np.float32]  # (OBSERVATION_DIM,)
    action_mask: NDArray[np.float32]  # (ACTION_DIM,)
    action: int
    action_prob: float
    value: float
    reward: float = 0.0
    done: bool = False


@dataclass
class PPOPendingStep:
    """Temporary storage for PPO trajectory step before reward assignment.

    Used during game play to collect step data. Once the game concludes,
    pending steps are converted to PPOStep with appropriate reward values.

    Note: This class does NOT use slots=True to allow dynamic attribute
    assignment (e.g., _reward) which is used by vectorized_env.py.

    Attributes:
        observation: State observation tensor of shape (OBSERVATION_DIM,).
        action_mask: Legal action mask of shape (ACTION_DIM,).
        action: Selected action index.
        action_prob: Probability of selected action under policy.
        value: Value estimate from critic at this state.
        player: Player who took this action (0 or 1).
    """

    observation: NDArray[np.float32]  # (OBSERVATION_DIM,)
    action_mask: NDArray[np.float32]  # (ACTION_DIM,)
    action: int
    action_prob: float
    value: float
    player: int = 0  # Default for backward compat with internal usage


# ============================================================================
# MCTS Training Steps
# ============================================================================


@dataclass(slots=True)
class MCTSStep:
    """Single finalized step in MCTS self-play trajectory.

    Unlike PPO steps which only store the action taken, MCTS steps
    store the complete improved policy (MCTS visit distribution) as a training
    target for the policy network.

    Attributes:
        observation: State observation tensor of shape (OBSERVATION_DIM,).
        action_mask: Legal action mask of shape (ACTION_DIM,).
        mcts_policy: Improved policy from MCTS search as dense array (ACTION_DIM,).
                    This is the training target for the policy network.
        action_taken: Action actually selected (sampled from mcts_policy).
        value: Game outcome from this player's perspective (+1/-1/0).
               Filled after game completes.
        player: Player who took this action (0 or 1).
    """

    observation: NDArray[np.float32]  # (OBSERVATION_DIM,)
    action_mask: NDArray[np.float32]  # (ACTION_DIM,)
    mcts_policy: NDArray[np.float32]  # (ACTION_DIM,) - dense policy target
    action_taken: int
    value: float  # Filled after game ends
    player: int


@dataclass(slots=True)
class MCTSPendingStep:
    """Temporary storage for MCTS step before value assignment.

    Used during game play to collect step data. Once the game concludes,
    pending steps are converted to MCTSStep with appropriate value targets.

    Attributes:
        observation: State observation tensor of shape (OBSERVATION_DIM,).
        action_mask: Legal action mask of shape (ACTION_DIM,).
        mcts_policy: Improved policy from MCTS search as dense array (ACTION_DIM,).
        action_taken: Action actually selected.
        player: Player who took this action (0 or 1).
    """

    observation: NDArray[np.float32]  # (OBSERVATION_DIM,)
    action_mask: NDArray[np.float32]  # (ACTION_DIM,)
    mcts_policy: NDArray[np.float32]  # (ACTION_DIM,)
    action_taken: int
    player: int


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Alias for backward compatibility with existing code
TrajectoryStep = PPOStep
"""Alias for PPOStep - backward compatibility with self_play.py exports."""

PendingStep = PPOPendingStep
"""Alias for PPOPendingStep - backward compatibility with vectorized_env.py."""


__all__ = [
    # Primary classes
    "PPOStep",
    "PPOPendingStep",
    "MCTSStep",
    "MCTSPendingStep",
    # Backward compatibility aliases
    "TrajectoryStep",
    "PendingStep",
]
