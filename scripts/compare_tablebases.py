#!/usr/bin/env python3
"""Compare retrograde tablebase with minimax solver.

Generates positions using the retrograde method, then verifies
each result using the forward minimax solver from endgame.py.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from _02_agents.tablebase.enumerate import PositionEnumerator, EnumerationConfig
from _02_agents.tablebase.retrograde import RetrogradeTablebase, RetrogradeConfig
from _02_agents.tablebase.endgame import EndgameTablebase, TablebaseConfig, GameTheoreticValue
from _01_simulator import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("compare_tablebases")


def compare_tablebases(max_cards: int = 2, sample_size: int | None = None) -> None:
    """Compare retrograde and minimax tablebase results.

    Args:
        max_cards: Maximum cards per player to enumerate
        sample_size: If set, only compare this many positions (for speed)
    """
    logger.info(f"=== Comparing tablebases for {max_cards} cards ===")

    # Step 1: Generate retrograde tablebase
    logger.info("Generating retrograde tablebase...")
    retro_start = time.time()

    # max_cards is total cards (both players combined)
    retro_config = RetrogradeConfig(max_cards=max_cards * 2)
    retro_tb = RetrogradeTablebase(retro_config)
    retro_tb.generate()

    retro_time = time.time() - retro_start
    logger.info(f"Retrograde: {len(retro_tb._positions)} positions in {retro_time:.2f}s")

    # Count retrograde results
    retro_wins = sum(1 for v in retro_tb._values.values() if v == GameTheoreticValue.WIN)
    retro_losses = sum(1 for v in retro_tb._values.values() if v == GameTheoreticValue.LOSS)
    retro_draws = sum(1 for v in retro_tb._values.values() if v == GameTheoreticValue.DRAW)
    logger.info(f"Retrograde stats: WIN={retro_wins}, LOSS={retro_losses}, DRAW={retro_draws}")

    # Step 2: Create minimax solver
    logger.info("Creating minimax solver...")
    minimax_config = TablebaseConfig(
        max_cards_per_player=max_cards,
        max_total_cards=max_cards * 2,
    )
    minimax_tb = EndgameTablebase(config=minimax_config)

    # Step 3: Compare each position
    logger.info("Comparing positions...")
    minimax_start = time.time()

    # Get positions from retrograde tablebase
    positions = list(retro_tb._positions.items())
    if sample_size and sample_size < len(positions):
        import random
        random.seed(42)
        positions = random.sample(positions, sample_size)
        logger.info(f"Sampling {sample_size} positions for comparison")

    matches = 0
    mismatches = []
    minimax_wins = 0
    minimax_losses = 0
    minimax_draws = 0

    for i, (key, game_state) in enumerate(positions):
        # Get retrograde result
        retro_value = GameTheoreticValue(retro_tb._values[key])

        # Solve with minimax
        perspective = game_state.active_player
        minimax_entry = minimax_tb.solve(game_state, perspective)

        # Count minimax results
        if minimax_entry.value == GameTheoreticValue.WIN:
            minimax_wins += 1
        elif minimax_entry.value == GameTheoreticValue.LOSS:
            minimax_losses += 1
        elif minimax_entry.value == GameTheoreticValue.DRAW:
            minimax_draws += 1

        # Compare
        if retro_value == minimax_entry.value:
            matches += 1
        else:
            mismatches.append({
                'key': key,
                'state': game_state,
                'retro': retro_value.name,
                'minimax': minimax_entry.value.name,
            })

        if (i + 1) % 1000 == 0:
            logger.info(f"Compared {i + 1}/{len(positions)} positions...")

    minimax_time = time.time() - minimax_start

    # Report results
    logger.info("")
    logger.info("=== COMPARISON RESULTS ===")
    logger.info(f"Total positions compared: {len(positions)}")
    logger.info(f"Matches: {matches} ({100*matches/len(positions):.1f}%)")
    logger.info(f"Mismatches: {len(mismatches)}")
    logger.info("")
    logger.info(f"Retrograde: WIN={retro_wins}, LOSS={retro_losses}, DRAW={retro_draws}")
    logger.info(f"Minimax:    WIN={minimax_wins}, LOSS={minimax_losses}, DRAW={minimax_draws}")
    logger.info("")
    logger.info(f"Retrograde time: {retro_time:.2f}s")
    logger.info(f"Minimax time: {minimax_time:.2f}s ({minimax_time/len(positions)*1000:.2f}ms/pos)")

    if mismatches:
        logger.warning("")
        logger.warning("=== MISMATCHES (first 10) ===")
        for m in mismatches[:10]:
            logger.warning(f"  Key {m['key'][:20]}...: retro={m['retro']}, minimax={m['minimax']}")
    else:
        logger.info("")
        logger.info("SUCCESS: All positions match between retrograde and minimax!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare tablebase implementations")
    parser.add_argument("--max-cards", type=int, default=2, help="Max cards per player")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (default: all)")

    args = parser.parse_args()
    compare_tablebases(max_cards=args.max_cards, sample_size=args.sample)
