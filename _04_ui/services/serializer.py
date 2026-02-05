"""Serialization functions for the Beasty Bar web UI."""

from __future__ import annotations

from collections import defaultdict
from datetime import timezone

from _01_simulator import actions, engine, simulate, state
from _04_ui.core.session import GameSession, TurnLogEntry  # noqa: TC001


def serialize_battle_state(game_state: state.State) -> dict:
    """Serialize game state for AI battle replay (both hands visible)."""
    return {
        "turn": game_state.turn,
        "activePlayer": game_state.active_player,
        "isTerminal": simulate.is_terminal(game_state),
        "score": simulate.score(game_state) if simulate.is_terminal(game_state) else None,
        "queue": [card_view(card) for card in game_state.zones.queue],
        "zones": {
            "beastyBar": [card_view(card) for card in game_state.zones.beasty_bar],
            "thatsIt": [card_view(card) for card in game_state.zones.thats_it],
        },
        "hands": [[card_view(card) for card in player_state.hand] for player_state in game_state.players],
    }


def serialize(game_state: state.State, seed: int | None, store: GameSession) -> dict:
    """Serialize game state for the main game view."""
    active_player = game_state.active_player
    legal = simulate.legal_actions(game_state, active_player) if not simulate.is_terminal(game_state) else ()
    log_entries = list(store.log)
    visible_state = game_state
    is_ai_turn = active_player != store.human_player and not simulate.is_terminal(game_state)
    return {
        "seed": seed,
        "turn": game_state.turn,
        "activePlayer": active_player,
        "isTerminal": simulate.is_terminal(game_state),
        "score": simulate.score(game_state) if simulate.is_terminal(game_state) else None,
        "humanPlayer": store.human_player,
        "aiOpponent": store.ai_opponent,
        "isAiTurn": is_ai_turn,
        "queue": [card_view(card) for card in visible_state.zones.queue],
        "zones": {
            "beastyBar": [card_view(card) for card in visible_state.zones.beasty_bar],
            "thatsIt": [card_view(card) for card in visible_state.zones.thats_it],
        },
        "hands": [[card_view(card) for card in player_state.hand] for player_state in visible_state.players],
        "legalActions": [serialize_action(game_state, action) for action in legal],
        "log": [serialize_log_entry(entry) for entry in log_entries],
        "logText": format_log_text(store, log_entries),
        "turnFlow": serialize_turn_steps(log_entries[-1].steps) if log_entries else [],
    }


def serialize_action(game_state: state.State, action: actions.Action) -> dict:
    """Serialize an action for the API response."""
    player = game_state.active_player
    card = game_state.players[player].hand[action.hand_index]
    return {
        "handIndex": action.hand_index,
        "params": list(action.params),
        "card": card_view(card),
        "label": action_label(card, action),
    }


def card_view(card: state.Card) -> dict:
    """Convert a Card to a JSON-serializable dict."""
    return {
        "owner": card.owner,
        "species": card.species,
        "strength": card.strength,
        "points": card.points,
    }


def action_label(card: state.Card, action: actions.Action) -> str:
    """Generate a human-readable label for an action."""
    species = card.species
    if species == "kangaroo":
        if action.params:
            hop = action.params[0]
            return f"Play {species} (hop {hop})"
        return f"Play {species}"
    if action.params:
        params = ",".join(str(p) for p in action.params)
        return f"Play {species} ({params})"
    return f"Play {species}"


def serialize_log_entry(entry: TurnLogEntry) -> dict:
    """Serialize a log entry for the API response."""
    timestamp = entry.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    iso_ts = timestamp.isoformat(timespec="milliseconds")
    epoch_ms = int(timestamp.timestamp() * 1000)
    return {
        "id": entry.id,
        "turn": entry.turn,
        "player": entry.player,
        "action": entry.action,
        "effects": list(entry.effects),
        "timestamp": iso_ts,
        "timestampMs": epoch_ms,
        "steps": serialize_turn_steps(entry.steps),
    }


def serialize_turn_steps(steps: tuple[engine.TurnStep, ...]) -> list[dict]:
    """Serialize turn steps for the API response."""
    return [
        {
            "name": step.name,
            "events": list(step.events),
        }
        for step in steps
    ]


def format_log_text(store: GameSession, entries: list[TurnLogEntry]) -> str:
    """Format log entries as human-readable text."""
    if not entries:
        return ""

    lines: list[str] = []
    for entry in entries:
        if entry.player is None:
            header = f"Turn {entry.turn} - Setup: {entry.action}"
        else:
            label = log_player_label(store, entry.player)
            header = f"Turn {entry.turn} - {label}: {entry.action}"
        lines.append(header)
        for effect in entry.effects:
            lines.append(f"  - {effect}")
    return "\n".join(lines)


def log_player_label(store: GameSession, player: int) -> str:
    """Get a human-readable label for a player."""
    if player == store.human_player:
        return f"P{player} (You)"
    return f"P{player}"


def format_queue_for_claude(queue: tuple[state.Card, ...]) -> str:
    """Format the queue for Claude Code to read."""
    if not queue:
        return "[empty]"
    cards = []
    for i, card in enumerate(queue):
        owner = "You" if card.owner == 1 else "Opp"
        cards.append(f"{i}: {card.species}({card.strength}) [{owner}]")
    return " -> ".join(cards)


def format_hand_for_claude(hand: tuple[state.Card, ...], legal: tuple[actions.Action, ...]) -> str:
    """Format a hand for Claude Code to read."""
    if not hand:
        return "[empty]"
    legal_indices = {a.hand_index for a in legal}
    cards = []
    for i, card in enumerate(hand):
        playable = "*" if i in legal_indices else " "
        cards.append(f"[{i}]{playable} {card.species}({card.strength}, {card.points}pts)")
    return "\n".join(cards)


def format_legal_actions_for_claude(game_state: state.State, legal: tuple[actions.Action, ...]) -> str:
    """Format legal actions for Claude Code to read."""
    if not legal:
        return "[none]"
    player = game_state.active_player
    lines = []
    for i, action in enumerate(legal, 1):
        card = game_state.players[player].hand[action.hand_index]
        label = action_label(card, action)
        lines.append(f"{i}. {label}")
    return "\n".join(lines)


def count_score(beasty_bar: tuple[state.Card, ...], player: int) -> int:
    """Count the score for a player from the beasty bar."""
    return sum(c.points for c in beasty_bar if c.owner == player)


# Helper functions for game effects description

def card_key(card: state.Card) -> tuple[int, str, int, int]:
    """Create a value-based key for card comparison."""
    return (card.owner, card.species, card.strength, card.points)


def drawn_cards(before: state.State, after: state.State, player: int) -> list[state.Card]:
    """Find cards in after hand that weren't in before hand."""
    before_keys = [card_key(card) for card in before.players[player].hand]
    new_cards = []
    for card in after.players[player].hand:
        key = card_key(card)
        if key in before_keys:
            before_keys.remove(key)  # Handle duplicates
        else:
            new_cards.append(card)
    return new_cards


def zone_new_cards(before_cards: tuple[state.Card, ...], after_cards: tuple[state.Card, ...]) -> list[state.Card]:
    """Find cards in after zone that weren't in before zone."""
    before_keys = [card_key(card) for card in before_cards]
    new_cards = []
    for card in after_cards:
        key = card_key(card)
        if key in before_keys:
            before_keys.remove(key)  # Handle duplicates
        else:
            new_cards.append(card)
    return new_cards


def format_card_list(cards: list[state.Card]) -> str:
    """Format a list of cards as a string."""
    return ", ".join(f"{card.species} (P{card.owner})" for card in cards)


def turn_flow_summary(steps: tuple[engine.TurnStep, ...]) -> list[str]:
    """Generate a summary of the turn flow."""
    lines: list[str] = []
    for idx, step in enumerate(steps, start=1):
        title = step.name.title()
        if step.events:
            lines.append(f"Step {idx} - {title}: {step.events[0]}")
            for extra in step.events[1:]:
                lines.append(f"  -> {extra}")
        else:
            lines.append(f"Step {idx} - {title}: No effect.")
    return lines


def describe_action_effects(
    store: GameSession,
    before: state.State,
    after: state.State,
    player: int,
    *,
    include_draw: bool = True,
) -> list[str]:
    """Describe the effects of an action."""
    effects: list[str] = []

    scored = zone_new_cards(before.zones.beasty_bar, after.zones.beasty_bar)
    if scored:
        owners: dict[int, int] = defaultdict(int)
        for card in scored:
            owners[card.owner] += card.points
        gains = ", ".join(f"P{owner} +{points}" for owner, points in sorted(owners.items()))
        effects.append(f"Heaven's Gate: {format_card_list(scored)} ({gains})")

    bounced = zone_new_cards(before.zones.thats_it, after.zones.thats_it)
    if bounced:
        effects.append(f"Sent to THAT'S IT: {format_card_list(bounced)}")

    if include_draw:
        draw = drawn_cards(before, after, player)
        if draw:
            if player == store.human_player:
                effects.append(f"Drew {format_card_list(draw)}")
            else:
                count = len(draw)
                label = "card" if count == 1 else "cards"
                effects.append(f"Drew {count} {label}")

    return effects
