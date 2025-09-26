"""Batch simulation entry point."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence, Tuple

from . import actions, engine, state

AgentFn = Callable[[state.State, Sequence[actions.Action]], actions.Action]


@dataclass
class SimulationConfig:
    seed: int
    games: int = 1
    agent_a: Optional[AgentFn] = None
    agent_b: Optional[AgentFn] = None


def new_game(seed: int, *, starting_player: int = 0) -> state.State:
    """Create a fresh game using the deterministic state initializer."""

    return state.initial_state(seed=seed, starting_player=starting_player)


def legal_actions(game_state: state.State, player: int) -> Tuple[actions.Action, ...]:
    """Expose the engine legal actions as a tuple for agent consumption."""

    return tuple(engine.legal_actions(game_state, player))


def apply(game_state: state.State, action: actions.Action) -> state.State:
    """Apply an action using the engine step function."""

    return engine.step(game_state, action)


def is_terminal(game_state: state.State) -> bool:
    """Proxy to the engine terminal check."""

    return engine.is_terminal(game_state)


def score(game_state: state.State) -> Tuple[int, ...]:
    """Return immutable score data for the finished game."""

    return tuple(engine.score(game_state))


def run(config: SimulationConfig) -> Iterator[state.State]:
    """Run one or more games according to the provided configuration.

    Agents default to a simple deterministic policy (first legal action) when
    not supplied. Seeds advance deterministically so re-running the same config
    yields identical sequences of games.
    """

    if config.games < 1:
        raise ValueError("Number of games must be at least 1")

    agents: Tuple[AgentFn, AgentFn] = (
        config.agent_a or _default_agent,
        config.agent_b or _default_agent,
    )

    for offset in range(config.games):
        game_seed = config.seed + offset
        game = new_game(game_seed)
        yield _play_game(game, agents)


def _play_game(game: state.State, agents: Tuple[AgentFn, AgentFn]) -> state.State:
    current = game
    while not engine.is_terminal(current):
        player = current.active_player
        legal = legal_actions(current, player)
        if not legal:
            raise RuntimeError("No legal actions available for player")
        action = agents[player](current, legal)
        current = apply(current, action)
    return current


def _default_agent(game_state: state.State, legal: Sequence[actions.Action]) -> actions.Action:
    return legal[0]


__all__ = [
    "AgentFn",
    "SimulationConfig",
    "new_game",
    "legal_actions",
    "apply",
    "is_terminal",
    "score",
    "run",
]
