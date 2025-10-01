
I need to learn and get up to speed with ALL of this!
* **PPO core** with masking, entropy, KL approx, grad-clip. Clean and correct.  
* **Policy/value net** config + orthogonal init + save/load checkpoints.   
* **Encoders** flatten canonical observation, expose obs size.  
* **Tournament harness** with Wilson early-stop, JSON/CSV export, and self-play manifest plumbing.    
* **Self-play CLI**: artifacts, rolling metrics, promotion gates, reservoir of checkpoints.   
* **Policy loader** for champion manifest → tournament/UI. 

# Gaps / risks

* **First-player bias in rollouts.** Learner is always player 0 and always starts. Randomize or alternate starting player in `collect_rollouts`.  
* **Sparse reward only at terminal step.** You write the shaped reward to the final transition only; all earlier steps get 0. Consider per-turn shaping or advantage bootstrap across episodes; current GAE will propagate but learning is slower.  
* **No exploration control.** Sampling is from raw logits with no temperature/epsilon schedules. Add τ or ε hooks during rollout. 
* **Opponent mix.** Active opponents = baselines + reservoir. Good, but no weighting or curriculum. Add Elo-weighted sampling or staged curriculum. 
* **Eval parity with tournaments.** Self-play promotion thresholds differ from earlier guidance; defaults are WR≥0.55 and ΔElo≥25. Verify alignment with your gate policy.  
* **Metrics/tests.** No unit tests for PPO/GAE/rollout masking. Add deterministic smoke tests and a tiny toy env test.

