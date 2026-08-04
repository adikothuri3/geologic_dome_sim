---
title: Experiment log
updated: 2026-08-04
status: current
---

# Experiment log

Every training or reconstruction run gets a row — automatically, with the commit hash, at the time of the run. **Rows are never deleted**: failed runs with honest takeaways are the point. Keep it a table; anything longer than a takeaway sentence belongs in the weekly lab notebook (`lab-notebook/`, outside this vault).

Columns: `run_id` (date + short slug), `commit` (short hash of the code that ran), `config` (env/model + the settings that mattered), `n_envs` (parallel envs, or `—` for recon runs), `metrics` (the numbers — see the evaluation harness in [[pipeline]]), `takeaway` (one sentence).

| run_id | commit | config | n_envs | metrics | takeaway |
| --- | --- | --- | --- | --- | --- |
| 2026-08-03-g1-joystick-smoke | 078625b-dirty | G1JoystickFlatTerrain, PPO, bs=8, nmb=32, steps=2,000,000, seed=0 | 256 | final eval reward nan, best nan, 0 min — FAILED | AttributeError: jax.device_put_replicated is deprecated; use jax.device_put instead. See https://docs.jax.dev/en/latest/migrate_pmap.html#drop-in-replacements for a drop-in replacement. |
| 2026-08-03-g1-joystick-smoke | 078625b-dirty | G1JoystickFlatTerrain, PPO, bs=8, nmb=32, steps=2,000,000, seed=0 | 256 | final eval reward -3.07, best -3.07, 3 min | smoke test only, not a usable policy |
| 2026-08-04-g1-joystick | 1aa7006 | G1JoystickFlatTerrain, PPO, bs=64, nmb=32, steps=100,000,000, seed=0 | 2048 | final eval reward 2.96, best 4.53, 27 min | baseline at 2048 envs; reward 3.0 |
| 2026-08-04-g1-joystick-full-collision-smoke | 1aa7006-dirty | G1JoystickFlatTerrain, **full-collision**, PPO, bs=8, nmb=32, njmax=384, steps=2,000,000, seed=0 | 256 | final eval reward -4.02, best -4.02, 3 min | smoke test only, not a usable policy |

