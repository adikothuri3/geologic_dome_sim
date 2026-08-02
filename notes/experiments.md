---
title: Experiment log
updated: 2026-08-01
status: current
---

# Experiment log

Every training or reconstruction run gets a row — automatically, with the commit hash, at the time of the run. **Rows are never deleted**: failed runs with honest takeaways are the point. Keep it a table; anything longer than a takeaway sentence belongs in the weekly lab notebook (`lab-notebook/`, outside this vault).

Columns: `run_id` (date + short slug), `commit` (short hash of the code that ran), `config` (env/model + the settings that mattered), `n_envs` (parallel envs, or `—` for recon runs), `metrics` (the numbers — see the evaluation harness in [[pipeline]]), `takeaway` (one sentence).

| run_id | commit | config | n_envs | metrics | takeaway |
| --- | --- | --- | --- | --- | --- |

*No runs yet — first rows expected in Phase 2 (policy training) and Phase 3 (reconstruction).*
