"""Rollout collection logic for self-play training."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import torch
from torch.distributions import Categorical

from _01_simulator import action_space, observations, rewards, simulate, state
from _02_agents.base import Agent

from . import encoders

AgentFactory = Callable[[], Agent]


@dataclass
class RolloutBatch:
    """Aggregated rollout tensors for PPO updates."""

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    action_masks: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    entropies: torch.Tensor
    episodes: int
    steps: int
    episode_rewards: List[float]


@dataclass
class RolloutConfig:
    """Configuration parameters for rollout collection."""

    min_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    margin_weight: float = 0.25
    jitter_scale: float = 0.01


def collect_rollouts(
    *,
    model,
    opponent_factories: Sequence[AgentFactory],
    config: RolloutConfig,
    base_seed: int,
    device: torch.device,
) -> RolloutBatch:
    """Run self-play games until ``min_steps`` decisions are recorded."""

    if not opponent_factories:
        raise ValueError("At least one opponent factory is required")

    rng = random.Random(base_seed)
    transitions: list[dict[str, torch.Tensor | float]] = []
    entropies: list[float] = []
    episode_rewards: list[float] = []

    steps = 0
    episodes = 0

    catalog = action_space.canonical_actions()

    while steps < config.min_steps:
        opponent_factory = rng.choice(opponent_factories)
        opponent = opponent_factory()

        game_seed = rng.randrange(0, 10_000_000)
        current = simulate.new_game(game_seed, starting_player=0)

        opponent_view = state.mask_state_for_player(current, 1)
        opponent.start_game(opponent_view)

        episode_transitions: list[int] = []

        while not simulate.is_terminal(current):
            player = current.active_player
            legal = simulate.legal_actions(current, player)

            if not legal:
                current = state.set_active_player(
                    current,
                    current.next_player(),
                    advance_turn=True,
                )
                continue

            if player == 0:
                obs = observations.build_observation(current, 0)
                obs_tensor = encoders.encode_observation(obs).to(device)
                with torch.no_grad():
                    logits, value = model(obs_tensor)

                action_space_view = action_space.legal_action_space(current, 0)
                legal_indices = torch.tensor(
                    action_space_view.legal_indices,
                    dtype=torch.long,
                    device=device,
                )
                if legal_indices.numel() == 0:
                    current = state.set_active_player(
                        current,
                        current.next_player(),
                        advance_turn=True,
                    )
                    continue

                legal_logits = logits.squeeze(0)[legal_indices]
                dist = Categorical(logits=legal_logits)
                sampled = dist.sample()
                action_index = legal_indices[sampled]
                action = catalog[int(action_index.item())]

                log_prob = dist.log_prob(sampled)
                entropy = dist.entropy()

                transitions.append(
                    {
                        "observation": obs_tensor.cpu(),
                        "action": torch.tensor(int(action_index.item()), dtype=torch.long),
                        "log_prob": log_prob.detach().cpu(),
                        "value": value.squeeze(0).detach().cpu(),
                        "reward": torch.tensor(0.0, dtype=torch.float32),
                        "done": torch.tensor(0.0, dtype=torch.float32),
                        "mask": torch.tensor(action_space_view.mask, dtype=torch.bool),
                    }
                )
                entropies.append(float(entropy.item()))
                episode_transitions.append(len(transitions) - 1)
                steps += 1

                next_state = simulate.apply(current, action)
                current = next_state
            else:
                view = state.mask_state_for_player(current, player)
                action = opponent.select_action(view, legal)
                current = simulate.apply(current, action)

        opponent_view = state.mask_state_for_player(current, 1)
        opponent.end_game(opponent_view)

        if episode_transitions:
            reward_tuple = rewards.shaped_reward(
                current,
                margin_weight=config.margin_weight,
                jitter_scale=config.jitter_scale,
            )
            rl_reward = float(reward_tuple[0])
            episode_rewards.append(rl_reward)
            last_index = episode_transitions[-1]
            transitions[last_index]["reward"] = torch.tensor(rl_reward, dtype=torch.float32)
            transitions[last_index]["done"] = torch.tensor(1.0, dtype=torch.float32)
            episodes += 1
        else:
            episode_rewards.append(0.0)
            episodes += 1

    batch = _build_batch(transitions, entropies, config)
    batch.steps = steps
    batch.episodes = episodes
    batch.episode_rewards = episode_rewards
    return batch


def _build_batch(
    transitions: list[dict[str, torch.Tensor]],
    entropies: list[float],
    config: RolloutConfig,
) -> RolloutBatch:
    if not transitions:
        raise ValueError("No transitions captured during rollout")

    observations = torch.stack([item["observation"] for item in transitions])
    actions = torch.stack([item["action"] for item in transitions])
    log_probs = torch.stack([item["log_prob"] for item in transitions])
    values = torch.stack([item["value"] for item in transitions])
    rewards = torch.stack([item["reward"] for item in transitions])
    dones = torch.stack([item["done"] for item in transitions])
    masks = torch.stack([item["mask"] for item in transitions])
    entropies_tensor = torch.tensor(entropies, dtype=torch.float32)

    advantages, returns = _gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    return RolloutBatch(
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        action_masks=masks,
        advantages=advantages,
        returns=returns,
        entropies=entropies_tensor,
        episodes=0,
        steps=0,
        episode_rewards=[],
    )


def _gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    next_advantage = 0.0
    next_value = 0.0

    for index in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[index].item()
        delta = rewards[index].item() + gamma * next_value * mask - values[index].item()
        next_advantage = delta + gamma * gae_lambda * mask * next_advantage
        advantages[index] = next_advantage
        next_value = values[index].item()

    returns = advantages + values
    return advantages, returns


__all__ = ["AgentFactory", "RolloutBatch", "RolloutConfig", "collect_rollouts"]
