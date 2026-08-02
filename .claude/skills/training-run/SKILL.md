---
name: training-run
description: Run an MJX/Playground fine-tune within this machine's 8 GB VRAM limits, capture config + commit hash, and append the mandatory results row to notes/experiments.md. Use for every RL training or fine-tuning run.
---

# training-run

Wraps any MJX / MuJoCo Playground training run (Phase 2 baseline, Phase 5 fine-tunes). Non-negotiables: VRAM limits respected, run logged.

## Before launch

1. **Env count**: cap `num_envs` at **1024–2048** on this 8 GB RTX 4060 Ti (Playground defaults like 8192 will OOM). Reduce batch size proportionally. If it still OOMs: halve envs, then reduce render/obs resolution. Big sweeps go to cloud GPU, not this box.
2. **Commit first**: the code that runs must be committed — record `git rev-parse --short HEAD`. If the tree is dirty, commit or stash before launching.
3. **Capture config**: env name, terrain asset, num_envs, batch size, learning rate, num_timesteps, randomization settings (terrain ±20% variants, friction/mass/pushes/latency for domain randomization), seed. Save alongside checkpoints.
4. Watch `nvidia-smi` during the first minutes; kill and reduce if VRAM is pinned at the ceiling.

## After the run (success **or** failure — failed runs are logged too)

Append one row to the table in `notes/experiments.md` (never delete or edit existing rows):

| column | content |
| --- | --- |
| `run_id` | `YYYY-MM-DD-<short-slug>` |
| `commit` | short hash of the code that ran |
| `config` | env/model + the settings that mattered |
| `n_envs` | parallel env count |
| `metrics` | evaluation-harness numbers: success rate over N rollouts, mean distance before fall, velocity-tracking error (recon vs. flat, before vs. after) |
| `takeaway` | one honest sentence |

Bump `updated` in the frontmatter. Longer narrative goes in `lab-notebook/`, not the vault.

## Phase 5 specifics

Curriculum flat → gentle recon → full recon. Fine-tune from the Playground G1 joystick baseline — never from scratch (see `notes/decisions.md`). Validate via ONNX export → sim2sim in vanilla MuJoCo at 50 Hz before claiming a result.
