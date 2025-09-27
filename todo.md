# Project TODO

## Phase 5 – Agents & Evaluation
- [x] Scaffold agent base abstractions and adapters (`beastybar/agents`)
- [x] Implement deterministic random agent and first-legal helper
- [x] Add greedy heuristic agent with configurable scoring hook
- [x] Build tournament harness for bulk simulations (CSV/JSON export)
- [ ] Document agent usage and data collection workflow
- [ ] Implement Diego heuristic agent (rule thresholds + fallback)
- [ ] Implement FrontRunner agent (queue control heuristic)
- [ ] Implement Killer agent (opponent point maximized removal)
- [ ] Extend tournament runner with opt-in per-action telemetry logging
- [ ] Add CLI/flag to run round robin of all agents with telemetry
- [ ] Run ~50k logged games covering all agent matchups and store raw event logs
- [ ] Build analysis script to aggregate logged events into candidate heuristics
- [ ] Generate Markdown report highlighting top 10 data-backed heuristics
- [ ] Implement DataHeuristic agent using top 10 rules and validate via tournaments
- [ ] Re-run bulk (millions) simulations with telemetry disabled and summarize results
