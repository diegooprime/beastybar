# Beasty Bar Simulator – Technical Reference

## Environment
- Python ≥3.10 via `pyproject.toml`; build uses setuptools so the package remains editable-friendly.
- Runtime deps: `fastapi` and `uvicorn` for serving the UI, `httpx` for HTTP-based tests.
- Pytest configured to search `tests/`, add project root to `PYTHONPATH`, and use terse `-ra` reporting.

## Core Rules Engine

### `beastybar/rules.py`
- Centralises constants like zone names, queue limit, hand/deck sizes, and player count so the rest of the code never hard-codes values.
- `Species` dataclass expresses per-animal metadata (strength, points, recurring/permanent flags); keeping it frozen prevents accidental edits.
- `SPECIES` map provides the canonical lookup, and `BASE_DECK` enforces the 12-card composition at import time to surface bad edits early.

### `beastybar/state.py`
- Defines frozen dataclasses (`Card`, `PlayerState`, `Zones`, `State`) to model the game; immutability guarantees snapshots can be shared across agents.
- `Card.__post_init__` validates species ownership against rule constants so invalid cards fail fast.
- `initial_state` takes a seed and builds per-player decks with deterministic shuffles, making replays reproducible and testable.
- Mutation helpers (`draw_card`, `remove_hand_card`, `append_queue`, `insert_queue`, `remove_queue_card`, `push_to_zone`, `replace_queue`, `set_active_player`) return new state instances while preserving invariants; private `_replace_player` and `_replace_zones` keep tuple packing centralised.
- `_ZONE_NAMES` whitelist protects against typos when pushing to shared zones.

### `beastybar/actions.py`
- `Action` dataclass is the uniform payload agents and the engine use; optional `params` slot keeps card-specific choices out of core logic.

### `beastybar/cards.py`
- `_HANDLERS` dispatch table wires each species to a resolver so adding new animals only touches one mapping.
- `resolve_play` runs the on-play effect immediately after a card enters the queue; the default handler is a no-op for species without abilities.
- Species handlers enforce official rules:
  - `lion` removes extra lions, scares monkeys, and jumps to the front to preserve dominance.
  - `snake` sorts the queue by strength to mirror its rulebook effect.
  - `giraffe` single-step swaps against weaker neighbours, matching the “peek over one” behaviour.
  - `kangaroo` honours hop parameters (bounded to two) and defaults to maximum reach when no params are given.
  - `monkey` bounces hippos/crocodiles when pairs form and rebuilds the queue so monkeys cluster in play order.
  - `parrot` and chameleon-as-parrot enforce explicit targets, sending victims to `thats_it`.
  - `seal` flips queue order using tuple reversal before recurrences.
  - `chameleon` clones a temporary card, reuses the target handler, and then swaps references back to preserve identity and owners.
  - `skunk` isolates top-strength tiers and expels them, leaving the rest intact.
- `process_recurring` scans the queue from gate to bounce applying recurring effects (hippo pushing forward, crocodile eating, giraffe hopping) while respecting zebra blocks; returning both new state and next index avoids infinite loops.
- `_swap_card_reference` keeps real card identity consistent after chameleon copies to avoid leaking placeholder instances.

### `beastybar/engine.py`
- `legal_actions` enumerates playable `Action`s for the active player, expanding kangaroo hops, parrot targets, and chameleon copy parameters so agents see every valid move.
- `_validate_action` mirrors `legal_actions` to catch stale or user-crafted actions before `step` mutates the state; species-specific branches ensure parameters stay in range.
- `step` orchestrates one turn: remove the chosen card from hand, append to queue, execute on-play logic, apply recurrences, run the five-card check, draw a replacement, and advance the active player/turn counter.
- `_apply_five_card_check` implements the Heaven’s Gate mechanic—front two enter the bar and the rear card bounces—so both engine and specialised agents share identical logic.
- `is_terminal` and `score` expose end-of-game detection and scoring totals to UI/tests.
- Helper `_chameleon_params` lets chameleon inherit parameterised moves (currently parrot) while keeping other species parameter-free.

### `beastybar/simulate.py`
- Thin wrapper around the engine for batch usage; `SimulationConfig` captures seed, game count, and optional agent callables.
- `run` yields finished states while auto-seeding successive games and defaulting to a deterministic “first legal” agent for unattended simulations.
- Convenience proxies (`new_game`, `legal_actions`, `apply`, `is_terminal`, `score`) keep agent code decoupled from engine internals.
- `_default_agent` illustrates the minimal policy used when no agent is provided.

### `tournament.py`
- Provides data classes for series configuration, telemetry (`ActionRecord`, `GameRecord`), and summary statistics.
- Implements Wilson-score–based early stopping, alternated starting positions, and optional action logging for reproducibility.
- CLI (`main`) supports head-to-head or round-robin tournaments, CSV/JSON exports, and Elo leaderboard generation with configurable base rating and K-factor.
- Helper routines snapshot hands/zones, serialise results, and compute leaderboard rankings in rating order.
