"""Common game utilities for training modules."""

from __future__ import annotations


def compute_winner(scores: tuple[int, int]) -> int | None:
    """Determine winner from scores.

    Args:
        scores: Tuple of (player0_score, player1_score).

    Returns:
        0 if player 0 wins, 1 if player 1 wins, None for draw.
    """
    if scores[0] > scores[1]:
        return 0
    elif scores[1] > scores[0]:
        return 1
    return None


def compute_rewards(
    winner: int | None,
    scores: tuple[int, int] | None = None,
    shaped: bool = False,
) -> tuple[float, float]:
    """Compute rewards for both players based on winner.

    Args:
        winner: 0 if player 0 wins, 1 if player 1 wins, None for draw.
        scores: Optional tuple of (player0_score, player1_score) for shaped rewards.
        shaped: If True, use margin-based shaped rewards (requires scores).

    Returns:
        Tuple of (reward_p0, reward_p1).
    """
    if shaped and scores is not None:
        # Shaped rewards based on score margin
        margin = abs(scores[0] - scores[1])
        base_reward = min(1.0 + margin * 0.1, 2.0)  # Cap at 2.0

        if winner == 0:
            return (base_reward, -base_reward)
        elif winner == 1:
            return (-base_reward, base_reward)
        return (0.0, 0.0)

    # Standard rewards: +1 win, -1 loss, 0 draw
    if winner == 0:
        return (1.0, -1.0)
    elif winner == 1:
        return (-1.0, 1.0)
    return (0.0, 0.0)
