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
- [ ] Scaffold web UI shell

## Phase 4 – QA & Tooling
- [ ] Golden replay fixtures
- [ ] CI/test automation scripts
- [ ] Developer workflow notes

### Active Focus
- [x] Capture golden two-turn replay (regression fixture)
- [x] Draft batch API scaffolding (`simulate.py`, agent hooks)
- [x] Document card behaviours and assumptions (seal swap, chameleon params)
