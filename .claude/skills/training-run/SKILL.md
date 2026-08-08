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

- **`num_envs` 4096 is the default, and it fits on this box.** Measured 2026-08-08: 5.05 GB of
  8.00 at ~3.7 s/iteration, because the G1 USD is instanced so what scales is PhysX state, not
  geometry. The previous rule here said "local runs are smoke tests only, 64–256 envs" — that
  was a guess, **wrong by 16×**, and it cost a training run 9.2M samples against upstream's
  147M. See `notes/setup.md`.
- **Always `--headless`.** The RTX renderer is the thing that genuinely does not fit — it alone
  eats ~7 GB, and `play_g1_flat.py --video` crashes this card. Video and multi-run sweeps go to
  a **cloud GPU**: `colab/isaac_g1_flat_colab.ipynb` (L4, not A100) or a rented ≥24 GB box.
- **Read the reward terms, not mean reward.** On the velocity task mean reward rose −30 → +4.11
  across a run in which the robot never took a step. `feet_air_time` and `track_lin_vel_xy_exp`
  are the progress signals; `track_lin_vel_xy_exp` at 0.37 is what standing still scores.
- Logs and checkpoints under `runs/isaac/<YYYY-MM-DD-slug>/`, alongside `progress.jsonl`,
  `best.json` + `model_best.pt`, and `outcome.json` — read `outcome.json` for a run's verdict,
  never the exit code, which Kit forces to 0 on shutdown.
- Tasks: the **`Dome-G1FullCollision-*`** family in `sims/isaac/tasks/dome_g1/` (full-collision
  `G1_CFG`, our DR set), not the stock `Isaac-Velocity-*-G1-v0`, which use `G1_MINIMAL_CFG`
  with collision meshes stripped and randomize nothing but observations. RSL-RL runner. See
  `sims/isaac/README.md` for the smoke ladder — and run `check_isaac.py --gate c` before any
  DR run, because a task that randomizes nothing trains and logs exactly like one that does.
- Phase 5: curriculum flat → gentle terrain → full terrain (TerrainGenerator curriculum);
  DR via EventManager terms. Validate via export → sim2sim in vanilla MuJoCo at 50 Hz
  (`sims/mujoco/`) before claiming a result.

## MJX / Playground (legacy track)

- Cap `num_envs` at **1024–2048** (Playground defaults like 8192 will OOM on 8 GB); hold
  `num_minibatches = 32` and let `sims/mujoco/scripts/train_g1.py` derive `batch_size`.
  If it still OOMs: halve envs first.
- `train_g1.py` enforces the clean-tree gate, captures config, writes `runs/mujoco/<run_id>/`,
  and appends the experiments row itself.
