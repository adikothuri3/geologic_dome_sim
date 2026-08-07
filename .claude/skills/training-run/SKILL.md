---
name: training-run
description: Run discipline for every RL training run — Isaac Lab (primary) or legacy MJX/Playground — within this machine's limits; capture config + commit hash and append the mandatory results row to notes/experiments.md. Use for every RL training or fine-tuning run.
---

# training-run

Wraps any training run in either sim. Non-negotiables, both tracks: hardware limits respected, run logged.

## Non-negotiables (any sim)

1. **Commit first**: the code that runs must be committed — record `git rev-parse --short HEAD`. If the tree is dirty, commit or stash before launching.
2. **Capture config**: task/env name, terrain asset, num_envs, batch size, learning rate, timesteps/iterations, randomization settings, seed. Save alongside checkpoints.
3. Watch `nvidia-smi` during the first minutes; kill and reduce if VRAM is pinned at the ceiling.
4. **Log the run** (success **or** failure — failed runs are logged too): append one row to the table in `notes/experiments.md`, never delete or edit existing rows, bump `updated`. Longer narrative goes in `lab-notebook/`, not the vault.

| column | content |
| --- | --- |
| `run_id` | `YYYY-MM-DD-<short-slug>` |
| `commit` | short hash of the code that ran |
| `config` | task/model + the settings that mattered |
| `n_envs` | parallel env count |
| `metrics` | evaluation-harness numbers: success rate over N rollouts, mean distance before fall, velocity-tracking error (terrain vs. flat, before vs. after) |
| `takeaway` | one honest sentence |

## Isaac Lab (primary track)

- **This box is below Isaac's minimum spec** (8 GB VRAM vs 16; 16 GB RAM vs 32). Local runs are
  smoke tests only: always `--headless`, `num_envs` **64–256**, short `--max_iterations`.
  Anything meant to produce a policy goes to a **cloud GPU** (≥24 GB).
- Logs and checkpoints under `runs/isaac/<YYYY-MM-DD-slug>/`.
- Tasks: `Isaac-Velocity-Flat-G1-v0` / `Isaac-Velocity-Rough-G1-v0`, RSL-RL runner; configs and
  overrides live in `sims/isaac/tasks/`. See `sims/isaac/README.md` for the smoke ladder.
- Phase 5: curriculum flat → gentle terrain → full terrain (TerrainGenerator curriculum);
  DR via EventManager terms. Validate via export → sim2sim in vanilla MuJoCo at 50 Hz
  (`sims/mujoco/`) before claiming a result.

## MJX / Playground (legacy track)

- Cap `num_envs` at **1024–2048** (Playground defaults like 8192 will OOM on 8 GB); hold
  `num_minibatches = 32` and let `sims/mujoco/scripts/train_g1.py` derive `batch_size`.
  If it still OOMs: halve envs first.
- `train_g1.py` enforces the clean-tree gate, captures config, writes `runs/mujoco/<run_id>/`,
  and appends the experiments row itself.
