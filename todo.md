# Project TODO

## Phase 0 – Foundations
- [x] Stub package structure (`beastybar/`, `tests/`)
- [x] Populate rules metadata constants
- [x] Choose baseline test harness (`pytest`)

## Phase 1 – State Model
- [x] Define immutable `State` representation
- [x] Implement deck/hand/queue helper utilities

## Phase 2 – Rule Engine
- [ ] Implement `step`, `legal_actions`, recurring resolution
- [x] Add unit tests per animal ability
- [x] Implement scoring and terminal checks

## Phase 3 – Interfaces
- [ ] Expose batch API functions
- [ ] Create CLI debug harness
- [ ] Scaffold web UI shell

## Phase 4 – QA & Tooling
- [ ] Golden replay fixtures
- [ ] CI/test automation scripts
- [ ] Developer workflow notes

### Active Focus
- [ ] Flesh out `legal_actions` to surface parameterised choices
- [ ] Build negative tests for invalid action parameters
- [ ] Document card behaviours and assumptions (seal swap, chameleon params)
