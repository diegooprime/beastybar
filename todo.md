# Project TODO

## Phase 0 – Foundations
- [x] Stub package structure (`beastybar/`, `tests/`)
- [x] Populate rules metadata constants
- [x] Choose baseline test harness (`pytest`)

## Phase 1 – State Model
- [x] Define immutable `State` representation
- [x] Implement deck/hand/queue helper utilities

## Phase 2 – Rule Engine
- [x] Implement turn step pipeline and recurring resolution
- [x] Flesh out `legal_actions` parameter handling and validation
- [x] Validate action parameter errors
- [x] Add unit tests per animal ability
- [x] Implement scoring and terminal checks

## Phase 3 – Interfaces
- [x] Expose batch API functions
- [ ] Create CLI debug harness
- [x] Scaffold web UI shell

## Phase 4 – QA & Tooling
- [ ] Golden replay fixtures
- [ ] CI/test automation scripts
- [ ] Developer workflow notes

## Phase 5 – Agents & Evaluation
- [ ] Scaffold agent base abstractions and adapters (`beastybar/agents`)
- [ ] Implement deterministic random agent and first-legal helper
- [ ] Add greedy heuristic agent with configurable scoring hook
- [ ] Build tournament harness for bulk simulations (CSV/JSON export)
- [ ] Document agent usage and data collection workflow

### Active Focus
- [x] Capture golden two-turn replay (regression fixture)
- [x] Draft batch API scaffolding (`simulate.py`, agent hooks)
- [x] Document card behaviours and assumptions (seal swap, chameleon params)
