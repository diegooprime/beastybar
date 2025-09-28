# Core training loop
https://chatgpt.com/c/68d98568-51f0-8332-b348-0bd86223f633

- Optimize probability of winning. That equals maximizing win rate over our evaluation distribution.
- Evaluation = fixed opponent pool: first, random, greedy, diego + champion in a round robin of X games. 

- New agent plays against the pool (first, random, diego and greedy) of existing agents in a round robin of X games. 
- Should the elo be computed everytime we run a new training round? or how should we manage this when adding a new opponent...? how does elo play out here....

# Logs
- Metrics: Win Rate, Point Margin, 
- Row per game: 


Things to add:
- Stochasticity to greedy
- CI, what parameters should i choose? 
what should be the promotion rule? elo? 


# Value dataset policy/value network
## 1. Inputs (state representation)
You don’t log everything; log what lets you reconstruct the position cleanly.
* **Queue state**

  * Up to 5 slots. For each: species, strength, points, owner.
* **Heaven’s Gate / Bar**

  * For each player: total points in bar, number of animals already in.
* **That’s It**

  * For each player: total points discarded.
* **Hands**

  * Active player’s hand cards (species, points, strength).
  * (Optional for opponent: remaining-card counts per species — derived from deck + public info).
* **Turn context**

  * Turn index (0–23).
  * Player to act (0 or 1).
  * Seed ID (for reproducibility).

That’s enough to reproduce the game situation without replaying from the beginning.

---
## 2. Targets (what the value model should predict)
* **Final reward for the acting player**

  * E.g. `+1` if win, `-1` if loss, `0` if tie.
* Or finer: **point margin normalized** (e.g. `(my_points - opp_points)/total_points`).
* Store at least both: `win_flag` and `point_margin`.

---
## 3. Metadata (to improve training quality)
* **Action actually taken** (card + params). Useful if you also want to train a policy later.
* **Game ID / Episode ID** (so you can group turns).
* **Seat role** (were you player 0 or player 1).
---
## 4. Minimal log row example
```
{
  game_id: 88217,
  turn: 12,
  player: 1,
  queue: [ {species:"Lion",owner:0,strength:12,points:2}, ... ],
  bar_points: {0:6,1:4},
  thatsit_points: {0:2,1:4},
  hand: [ {species:"Parrot",strength:2,points:4}, ... ],
  turn_index: 12,
  seed: 3308908521,
  action_played: {species:"Parrot",params:{target:"Lion"}},
  reward_final: 1,             # win/loss/tie
  point_margin: +4,            # my_points - opp_points
}
```
## 5. What not to track
* Full history of every move (too bulky; you can reconstruct from seed if needed).
* Redundant text like “Player 1, choose a card” (no signal).
* Raw board screenshots or redundant logs.



> When a training run calls play_series, it builds a SeriesResult containing both per-game telemetry and an aggregate roll-up, so you always have raw data plus summary stats.

  - Per-game record GameRecord (_03_training/tournament.py:80) captures the matchup index, RNG seed, which player started, total turns, final two-player score tuple, winner (or None for ties),
  and optionally the full action trace.
  - Aggregate summary SeriesSummary (_03_training/tournament.py:90) stores the number of games, win counts for each seat, tie count, average score per seat, and average game length in turns;
  this is what the CLI prints after summarize runs (_03_training/tournament.py:259).

  Action-level telemetry is opt-in via collect_actions=True. When enabled, each step produces an ActionRecord (_03_training/tournament.py:164 and _03_training/tournament.py:333):
py:291).
  - export_json serializes the richer structure. Each game is a dict with the core fields plus an actions array when present (_03_training/tournament.py:391). Each action entry becomes a dict
  with explicit keys (e.g., handBefore, legalOptions, queueBefore) where nested tuples are expanded into lists/dicts to stay JSON-friendly (_03_training/tournament.py:405).

  Round-robin mode applies the same recording pipeline; it just drives multiple play_series runs and writes the per-match JSON into the requested directory.