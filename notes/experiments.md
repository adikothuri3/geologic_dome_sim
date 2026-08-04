---
title: Experiment log
updated: 2026-08-03
status: current
---

# Experiment log

Every training or reconstruction run gets a row — automatically, with the commit hash, at the time of the run. **Rows are never deleted**: failed runs with honest takeaways are the point. Keep it a table; anything longer than a takeaway sentence belongs in the weekly lab notebook (`lab-notebook/`, outside this vault).

Columns: `run_id` (date + short slug), `commit` (short hash of the code that ran), `config` (env/model + the settings that mattered), `n_envs` (parallel envs, or `—` for recon runs), `metrics` (the numbers — see the evaluation harness in [[pipeline]]), `takeaway` (one sentence).

| run_id | commit | config | n_envs | metrics | takeaway |
| --- | --- | --- | --- | --- | --- |
| 2026-08-03-g1-joystick-smoke | 078625b-dirty | G1JoystickFlatTerrain, PPO, bs=8, nmb=32, steps=2,000,000, seed=0 | 256 | final eval reward nan, best nan, 0 min — FAILED | AttributeError: jax.device_put_replicated is deprecated; use jax.device_put instead. See https://docs.jax.dev/en/latest/migrate_pmap.html#drop-in-replacements for a drop-in replacement. |
| 2026-08-03-g1-joystick-smoke | 078625b-dirty | G1JoystickFlatTerrain, PPO, bs=8, nmb=32, steps=2,000,000, seed=0 | 256 | final eval reward -3.07, best -3.07, 3 min | smoke test only, not a usable policy |

