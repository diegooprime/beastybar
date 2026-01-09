#!/usr/bin/env python
"""Play against the neural agent interactively.

Usage:
    python scripts/play.py --model checkpoint.pt
    python scripts/play.py --model checkpoint.pt --you-start
    python scripts/play.py --model checkpoint.pt --opponent heuristic

This script allows you to play Beasty Bar against the trained neural agent
in an interactive terminal session. The game state is displayed after each
turn, and you can choose your actions from the list of legal moves.

Examples:
    # Play against neural agent
    python scripts/play.py --model checkpoints/final.pt

    # You go first
    python scripts/play.py --model checkpoints/final.pt --you-start

    # Watch neural agent vs heuristic
    python scripts/play.py --model checkpoints/final.pt --opponent heuristic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import TYPE_CHECKING

from _01_simulator import engine, simulate
from _01_simulator.formatting import card_list
from _02_agents.base import Agent
from _02_agents.neural.agent import load_neural_agent
from _03_training.evaluation import create_opponent

if TYPE_CHECKING:
    from _01_simulator.state import State

logger = logging.getLogger(__name__)


def format_state(state: State) -> str:
    """Format game state for display.

    Args:
        state: Game state to format.

    Returns:
        Formatted string representation of the state.
    """
    lines = []
    lines.append(f"Turn {state.turn} | Active Player: {state.active_player}")
    lines.append("")

    # Queue
    if state.queue:
        lines.append(f"Queue ({len(state.queue)} cards):")
        lines.append(f"  {card_list(state.queue)}")
    else:
        lines.append("Queue: (empty)")
    lines.append("")

    # Beasty Bar
    if state.beasty_bar:
        lines.append(f"Beasty Bar ({len(state.beasty_bar)} cards):")
        lines.append(f"  {card_list(state.beasty_bar)}")
    else:
        lines.append("Beasty Bar: (empty)")
    lines.append("")

    # That's It
    if state.thats_it:
        lines.append(f"That's It ({len(state.thats_it)} cards):")
        lines.append(f"  {card_list(state.thats_it)}")
    else:
        lines.append("That's It: (empty)")
    lines.append("")

    # Hands
    lines.append(f"Player 0 Hand ({len(state.hands[0])} cards):")
    if state.hands[0]:
        lines.append(f"  {card_list(state.hands[0])}")
    else:
        lines.append("  (empty)")

    lines.append(f"Player 1 Hand ({len(state.hands[1])} cards):")
    if state.hands[1]:
        lines.append(f"  {card_list(state.hands[1])}")
    else:
        lines.append("  (empty)")
    lines.append("")

    # Decks
    lines.append(f"Player 0 Deck: {len(state.decks[0])} cards remaining")
    lines.append(f"Player 1 Deck: {len(state.decks[1])} cards remaining")

    return "\n".join(lines)


class HumanAgent(Agent):
    """Interactive human player agent."""

    @property
    def name(self) -> str:
        return "Human"

    def select_action(self, game_state: State, legal_actions):
        """Prompt human player to select an action.

        Args:
            game_state: Current game state.
            legal_actions: List of legal actions.

        Returns:
            Selected action.
        """
        # Display current state
        print("\n" + "=" * 80)
        print(format_state(game_state))
        print("=" * 80)

        # Display legal actions
        print(f"\nPlayer {game_state.active_player}'s turn - Legal actions:")
        for i, action in enumerate(legal_actions):
            print(f"  [{i}] {action}")

        # Get player input
        while True:
            try:
                choice = input(f"\nSelect action [0-{len(legal_actions)-1}]: ").strip()
                idx = int(choice)
                if 0 <= idx < len(legal_actions):
                    selected = legal_actions[idx]
                    print(f"You selected: {selected}")
                    return selected
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(legal_actions)-1}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\nGame interrupted by user.")
                sys.exit(0)


def play_interactive_game(
    player0: Agent,
    player1: Agent,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[int, int]:
    """Play a single interactive game.

    Args:
        player0: Agent for player 0.
        player1: Agent for player 1.
        seed: Random seed for game.
        verbose: Whether to print detailed state information.

    Returns:
        Tuple of (player0_score, player1_score).
    """
    config = simulate.SimulationConfig(
        seed=seed,
        games=1,
        agent_a=player0,
        agent_b=player1,
    )

    print("\n" + "=" * 80)
    print(f"Starting game: {player0.name} vs {player1.name}")
    print(f"Random seed: {seed}")
    print("=" * 80)

    for final_state in simulate.run(config):
        scores = engine.score(final_state)

        # Display final state
        print("\n" + "=" * 80)
        print("GAME OVER!")
        print(format_state(final_state))
        print("=" * 80)

        # Display scores
        print("\nFinal Scores:")
        print(f"  {player0.name} (Player 0): {scores[0]} points")
        print(f"  {player1.name} (Player 1): {scores[1]} points")

        if scores[0] > scores[1]:
            print(f"\n{player0.name} wins!")
        elif scores[1] > scores[0]:
            print(f"\n{player1.name} wins!")
        else:
            print("\nIt's a draw!")

        return scores[0], scores[1]

    return 0, 0  # Should not reach here


def main() -> int:
    """Main interactive play CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Play Beasty Bar against the neural agent interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device for model inference (default: auto)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["greedy", "stochastic", "temperature"],
        default="greedy",
        help="Inference mode for neural agent (default: greedy)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for 'temperature' mode (default: 1.0)",
    )

    # Game settings
    parser.add_argument(
        "--you-start",
        action="store_true",
        help="You play as player 0 (go first)",
    )

    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        help="Watch neural agent play against another agent instead (e.g., 'heuristic', 'random', 'mcts-500')",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the game (default: 42)",
    )

    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to play (default: 1)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game information",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load neural agent
        logger.info(f"Loading model from {args.model}")
        neural_agent = load_neural_agent(
            checkpoint_path=args.model,
            mode=args.mode,
            temperature=args.temperature,
            device=args.device,
        )
        logger.info(f"Model loaded: {neural_agent.name}, device={neural_agent.device}")

        # Set up opponent
        if args.opponent:
            # Watch neural agent vs another agent
            opponent_agent = create_opponent(args.opponent)
            logger.info(f"Neural agent will play against {opponent_agent.name}")

            if args.you_start:
                player0, player1 = neural_agent, opponent_agent
            else:
                player0, player1 = opponent_agent, neural_agent

        else:
            # Interactive play
            human_agent = HumanAgent()
            logger.info("Starting interactive play mode")

            if args.you_start:
                player0, player1 = human_agent, neural_agent
                print("\nYou are Player 0 (you go first)")
            else:
                player0, player1 = neural_agent, human_agent
                print("\nYou are Player 1 (neural agent goes first)")

        # Play games
        wins_0 = 0
        wins_1 = 0
        draws = 0

        for game_num in range(args.games):
            if args.games > 1:
                print(f"\n{'=' * 80}")
                print(f"Game {game_num + 1} of {args.games}")
                print(f"{'=' * 80}")

            game_seed = args.seed + game_num
            score_0, score_1 = play_interactive_game(
                player0,
                player1,
                seed=game_seed,
                verbose=args.verbose,
            )

            if score_0 > score_1:
                wins_0 += 1
            elif score_1 > score_0:
                wins_1 += 1
            else:
                draws += 1

        # Summary for multiple games
        if args.games > 1:
            print("\n" + "=" * 80)
            print("MATCH SUMMARY")
            print("=" * 80)
            print(f"Games played: {args.games}")
            print(f"{player0.name}: {wins_0} wins")
            print(f"{player1.name}: {wins_1} wins")
            print(f"Draws: {draws}")

            if wins_0 > wins_1:
                print(f"\n{player0.name} wins the match!")
            elif wins_1 > wins_0:
                print(f"\n{player1.name} wins the match!")
            else:
                print("\nThe match is tied!")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1

    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1

    except KeyboardInterrupt:
        logger.info("\nGame interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Game failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
