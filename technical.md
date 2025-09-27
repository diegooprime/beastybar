# Beasty Bar Simulator – Technical Reference

## Architecture
- Layered layout: rules engine (`beastybar`), human UI (`beastybar/ui`), agent tooling (`beastybar/agents`), automation scripts, and regression tests.
- Deterministic, immutable state updates keep replays reproducible and make it safe to branch simulations.
- Minimal dependencies (FastAPI stack + httpx) keep install light and let pytest run without network.
- Seeding and telemetry directories (`logs/`) capture tournament outputs for offline analysis.

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

### `beastybar/__init__.py`
- Re-exports key submodules to offer a concise public API for consumers/tests.

## Agent Toolkit (`beastybar/agents`)

### `base.py`
- Abstract `Agent` class defines lifecycle hooks (`start_game`, `end_game`) and enforces implementers provide `select_action`; the callable wrapper returned by `bind` guards against illegal outputs.
- `ensure_legal` is a reusable guard when agents sample from candidate lists.

### `baselines.py`
- `FirstLegalAgent` provides a deterministic policy ideal for reproducible tests.
- `RandomAgent` keeps its own RNG (seedable) to allow stochastic baselines without global randomness.

### `evaluation.py`
- Defines `HeuristicFn` signature between state and float, plus helpers to apply heuristics after simulating an action.
- `material_advantage` scores queue/bar state cheaply, weighting bar presence most heavily, queue position by distance, and penalising losses to `thats_it`.
- `best_action` iterates legal moves once, deriving both the best action and its heuristic score for reuse by greedy agents.

### `greedy.py`
- `GreedyAgent` plugs any heuristic into `best_action`; defaults to `material_advantage` so users can swap heuristics without subclassing.

### `diego.py`
- Implements a rule-based agent tuned to “Diego” community heuristics: pre-groups legal actions per hand card, samples randomly for stochastic species (`chameleon`, `giraffe`, `snake`), and applies species-specific filters (e.g., crocodile thresholds, seal value gates, monkey pairing).
- `ActionOutcome` captures rich telemetry (zone changes, point swings, queue placement) enabling nuanced scoring.
- `_score` composes multiple priorities (net bar gain, opponent losses, queue position, hand order) into a tuple for deterministic comparisons.
- Falls back to `FirstLegalAgent` when filters reject all options, preserving progress.

### `frontrunner.py`
- Focuses on controlling the front of the queue; simulates one-step outcomes manually to inspect entrant counts and recurring removal.
- `_reject_lion` and `_reject_lone_front` enforce heuristics that avoid stranded lions or exposing seals/crocs behind solo leaders.
- `_SimulationTrace` records intermediate queues to reason about recurring effects and five-card entrants without mutating the canonical state.

### `killer.py`
- Prioritises opponent point loss by tracing both bounce and five-card check results; weights bounce losses more heavily than gate expulsions.
- Uses `_trace_losses` to run card logic without mutating the original state, mirroring engine sequencing.
- Falls back to `FirstLegalAgent` when no option exceeds thresholds to avoid deadlocks.

### `heuristic50k.py`
- Encodes aggregated heuristics derived from 50k logged games (`_HEURISTIC_WEIGHTS`), scaling them to bias toward high-impact contexts.
- `_lookahead_bonus` actually simulates the action using the engine to factor immediate bar deltas and queue front control into the score.
- Uses `GreedyAgent` as a fallback to retain strong baseline behaviour when no heuristic matches.

### `tournament.py`
- Provides data classes for series configuration, telemetry (`ActionRecord`, `GameRecord`), and summary statistics.
- Implements Wilson-score–based early stopping, alternated starting positions, and optional action logging for reproducibility.
- CLI (`main`) supports head-to-head or round-robin tournaments, CSV/JSON exports, and Elo leaderboard generation with configurable base rating and K-factor.
- Helper routines snapshot hands/zones, serialise results, and compute leaderboard rankings in rating order.

### `__init__.py`
- Re-exports the primary agent classes and helpers to simplify imports for consumers.

## Web UI (`beastybar/ui`)
- `app.py` exposes `create_app`, assembling a FastAPI instance with in-memory `SessionStore` (holding state, seed, human/agent roles).
- Endpoints:
  - `GET /` serves the static index for humans.
  - `GET /api/agents` lists built-in agent keys.
  - `POST /api/new-game` seeds/reset games, optionally instantiating an opponent agent and auto-playing non-human turns.
  - `GET /api/state`/`/api/legal-actions` expose current state and legal moves.
  - `POST /api/action` validates payloads against the legal set before applying them; on agent games it chains `_auto_play` until it is the human’s turn or the game ends.
- Static assets (`static/index.html`) deliver the entire UI: CSS for a responsive layout, DOM scaffolding for queue, hands, piles, log, and modal, plus inline JavaScript that:
  - Boots the session, fetches state, and re-renders after each API call.
  - Tracks legal actions per hand card, supports parameter selection (kangaroo hops, parrot targets), and provides accessible tooltips.
  - Maintains an event log, modals for zone contents, and keyboard shortcuts (Escape to cancel).
  - Ensures pile previews stay in sync via lightweight state derived from API responses.

## Scripts
- `scripts/analyze_logs.py` aggregates tournament telemetry (`*_vs_*.json`) into weighted heuristics: parses per-action records, buckets context by queue length/front control, computes win-rate and score-diff metrics, normalises rankings, and optionally exports JSON tables for new heuristic agents.

## Tests (`tests/`)
- `test_state.py` validates initial setup invariants (hand sizes, empty zones).
- `test_actions.py` focuses on `engine.legal_actions` and parameter validation, ensuring chameleon/parrot/kangaroo rules are enforced.
- `test_cards.py` exercises individual card behaviours and recurring effects against expected queue/zones.
- `test_simulate.py` confirms the batch interface yields deterministic, reproducible runs.
- `test_properties.py` runs property-like checks: total card conservation, queue limit, deterministic seeds.
- `test_replay.py` codifies golden turn-by-turn snapshots for complex interactions (monkeys, chameleon-parrot combos).
- `test_agents.py` guards agent heuristics, enforcing species-specific decisions and heuristic helper behaviour.
- `test_ui.py` hits FastAPI endpoints to verify game lifecycle and static routing.
- `tests/__init__.py` keeps the package discoverable for pytest.

## Documentation & Project Assets
- `readme.md` outlines the high-level roadmap and layer definitions for contributors.
- `rules.md` mirrors the 2019 rulebook, anchoring the implementation against official wording.
- `notes.md` captures developer assumptions (e.g., order of recurring effects, chameleon parameter forwarding).
- `todo.md` tracks agent/evaluation milestones; unchecked boxes mark planned work even if partially implemented, preserving historical intent.
- `logs/` stores tournament outputs (per-match directories plus aggregated JSON summaries) used by the analysis script.
- `technical.md` (this file) centralises technical rationale for future maintainers.
