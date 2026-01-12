"""Claude API agent for Beasty Bar."""

from __future__ import annotations

import os
import re

import anthropic

from _01_simulator import actions, rules, state

MODEL = "claude-opus-4-5-20251101"

# All species each player starts with
ALL_SPECIES = set(rules.BASE_DECK)


def get_claude_move(
    game_state: state.State,
    player: int,
    legal_actions: tuple[actions.Action, ...],
) -> actions.Action:
    """Get Claude's move via Anthropic API with extended thinking."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = _format_game_state(game_state, player, legal_actions)

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 2000,
        },
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse the action index from response (check text blocks, skip thinking)
    for block in response.content:
        if block.type == "text":
            text = block.text.strip()
            action_index = _parse_action_index(text, len(legal_actions))
            return legal_actions[action_index - 1]

    # Fallback if no text block found
    return legal_actions[0]


def _format_game_state(
    game_state: state.State,
    player: int,
    legal_actions: tuple[actions.Action, ...],
) -> str:
    """Format game state as a prompt for Claude."""
    opponent = 1 - player

    # Format queue
    queue_parts = []
    for i, card in enumerate(game_state.zones.queue):
        owner = "You" if card.owner == player else "Opp"
        queue_parts.append(f"{i}: {card.species}({card.strength}) [{owner}]")
    queue_str = " -> ".join(queue_parts) if queue_parts else "[empty]"

    # Format hand
    hand = game_state.players[player].hand
    hand_parts = []
    for i, card in enumerate(hand):
        hand_parts.append(f"[{i}] {card.species}({card.strength}, {card.points}pts)")
    hand_str = "\n".join(hand_parts)

    # Format legal actions
    action_parts = []
    for i, action in enumerate(legal_actions, 1):
        card = hand[action.hand_index]
        label = _action_label(card, action)
        action_parts.append(f"{i}. {label}")
    actions_str = "\n".join(action_parts)

    # Count scores
    my_score = sum(c.points for c in game_state.zones.beasty_bar if c.owner == player)
    opp_score = sum(c.points for c in game_state.zones.beasty_bar if c.owner == opponent)

    # Game history - cards that scored and cards eliminated
    history_parts = []

    scored_cards = game_state.zones.beasty_bar
    if scored_cards:
        scored_str = ", ".join(
            f"{c.species}({'You' if c.owner == player else 'Opp'})"
            for c in scored_cards
        )
        history_parts.append(f"Scored (Heaven's Gate): {scored_str}")

    eliminated_cards = game_state.zones.thats_it
    if eliminated_cards:
        elim_str = ", ".join(
            f"{c.species}({'You' if c.owner == player else 'Opp'})"
            for c in eliminated_cards
        )
        history_parts.append(f"Eliminated (That's It): {elim_str}")

    history_str = "\n".join(history_parts) if history_parts else "No cards scored or eliminated yet"

    # Opponent info - what cards they've used and what might remain
    opp_played = _get_played_species(game_state, opponent)
    opp_in_queue = {c.species for c in game_state.zones.queue if c.owner == opponent}
    opp_known_used = opp_played | opp_in_queue
    opp_possible_remaining = ALL_SPECIES - opp_known_used

    opp_hand_size = len(game_state.players[opponent].hand)
    opp_deck_size = len(game_state.players[opponent].deck)

    opp_info_parts = [
        f"Cards in hand: {opp_hand_size}",
        f"Cards in deck: {opp_deck_size}",
    ]
    if opp_known_used:
        opp_info_parts.append(f"Cards already played: {', '.join(sorted(opp_known_used))}")
    if opp_possible_remaining:
        opp_info_parts.append(f"Possible remaining: {', '.join(sorted(opp_possible_remaining))}")

    opp_info_str = "\n".join(opp_info_parts)

    # Your remaining cards info
    _get_played_species(game_state, player)
    {c.species for c in game_state.zones.queue if c.owner == player}
    {c.species for c in hand}
    my_in_deck = {c.species for c in game_state.players[player].deck}

    my_info_parts = [
        f"Cards in deck: {len(game_state.players[player].deck)}",
    ]
    if my_in_deck:
        my_info_parts.append(f"Deck contents: {', '.join(sorted(my_in_deck))}")

    my_info_str = "\n".join(my_info_parts)

    return f"""You are playing Beasty Bar, a strategic card game where animals queue to enter Heaven's Gate.

RULES:
- Cards enter from the right of the queue and move left toward Heaven's Gate
- When 5 cards are in queue: leftmost 2 enter Heaven's Gate (score points), rightmost 1 goes to That's It (eliminated, no points)
- Each animal has a special ability that triggers when played
- Goal: Maximize your points by getting your high-value animals into Heaven's Gate

ANIMAL ABILITIES (strength, points):
- Lion (12, 2pts): Pushes to front of queue immediately
- Hippo (11, 2pts): Eats ALL cards with strength ≤5 (recurring - triggers again when new cards enter)
- Crocodile (10, 3pts): Eats the card directly in front of it (recurring)
- Snake (9, 2pts): Swaps positions with any one card in queue
- Giraffe (8, 3pts): Moves forward past all cards with strength <8 (recurring)
- Zebra (7, 4pts): Permanent - stays in queue, cannot be moved or eaten
- Seal (6, 2pts): Copies the ability of an adjacent card
- Chameleon (5, 3pts): Takes the strength of an adjacent card (for positioning)
- Monkey (4, 3pts): Moves to be next to another monkey if one is present
- Kangaroo (3, 4pts): Hops forward over 1-2 cards (you choose distance)
- Parrot (2, 4pts): Copies strength of card directly behind it
- Skunk (1, 4pts): Kicks out the STRONGEST card in queue to That's It

STRATEGIC TIPS:
- Low-strength cards (skunk, parrot, kangaroo) have high points - protect them!
- Hippo is devastating when opponent has low-strength cards in queue
- Skunk can eliminate opponent's strong cards before they score
- Lion guarantees front position but only 2 points
- Watch the queue length - at 5 cards, scoring happens!

=== CURRENT GAME STATE ===

Turn: {game_state.turn}
Scores: You {my_score} - Opponent {opp_score}

QUEUE (position 0 = front/Heaven's Gate, higher = back/That's It):
{queue_str}

YOUR HAND:
{hand_str}

YOUR INFO:
{my_info_str}

OPPONENT INFO:
{opp_info_str}

GAME HISTORY:
{history_str}

=== LEGAL ACTIONS ===
{actions_str}

Think carefully about:
1. What will happen when you play each card (ability effects)
2. Queue positions after abilities resolve
3. When will scoring trigger (queue reaches 5)?
4. What might opponent play next turn?
5. Point maximization vs denying opponent points

Choose the best action. Reply with ONLY the action number (e.g., "1" or "3")."""


def _get_played_species(game_state: state.State, player: int) -> set[str]:
    """Get species that a player has already played (scored or eliminated)."""
    played = set()
    for card in game_state.zones.beasty_bar:
        if card.owner == player:
            played.add(card.species)
    for card in game_state.zones.thats_it:
        if card.owner == player:
            played.add(card.species)
    return played


def _action_label(card: state.Card, action: actions.Action) -> str:
    """Format action as human-readable label."""
    species = card.species
    if species == "kangaroo" and action.params:
        return f"Play {species} (hop {action.params[0]})"
    if action.params:
        params = ",".join(str(p) for p in action.params)
        return f"Play {species} ({params})"
    return f"Play {species}"


def _parse_action_index(text: str, num_actions: int) -> int:
    """Parse action index from Claude's response."""
    # Try to find a number in the response
    match = re.search(r"\d+", text)
    if match:
        index = int(match.group())
        if 1 <= index <= num_actions:
            return index
    # Default to first action if parsing fails
    return 1
