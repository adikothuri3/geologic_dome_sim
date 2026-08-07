# `sims/mujoco/` — the legacy MuJoCo/MJX track

> **Status: working legacy since 2026-08-07.** The primary simulator is now Isaac Sim /
> Isaac Lab (`sims/isaac/`) to mirror the Robot Everest team's actual stack — see the
> 2026-08-07 entry in `notes/decisions.md`. This track is kept **runnable**, not frozen:
> it is the record of Phases 1–2 and keeps a permanent job as the 50 Hz sim2sim
> validator for Isaac-trained policies.

## What this track proved

- **Phase 1** (done Aug 2): G1 posed through keyframes on a numpy heightfield, rendered
  to video (`scripts/pose_and_render.py` → `reports/phase1_stand.mp4`).
- **Phase 2** (done Aug 4): self-trained PPO joystick policy walking and turning under
  command on a **full-body-collision** G1 (`scripts/train_g1.py`, MJX/Playground/brax).
  Reward terms documented in `notes/locomotion-policy.md`.
- **Phase 4 chain** (Aug 6, upstream footage): video → cloud → scale → cleanup →
  heightfield → G1 settling at 0.3 mm foot penetration (`terrain/cloud_to_hfield.py`,
  `scripts/settle_g1_recon.py`).

## Layout

| Dir | Contents |
| --- | --- |
| `scripts/` | training (`train_g1.py`), model generation (`make_full_collision_xml.py`), gates (`check_*.py`, `settle_g1_recon.py`), analysis (`render_policy.py`, `compare_gaits.py`), installers (`setup_wsl.sh`, `setup_wsl_stage2.ps1`), Phase 1 demo (`pose_and_render.py`) |
| `terrain/` | numpy heightfield tools (`make_hfield.py`), gates (`drop_test.py`), point-cloud → MuJoCo hfield converter (`cloud_to_hfield.py`), built assets (`assets/`) |
| `xmls/` | the four MJCF scenes + the `menagerie` junction |

## Runtime

Everything here runs in **WSL** with the `~/venvs/dome` venv (`source ~/venvs/dome/bin/activate`,
repo at `/mnt/c/Users/Aditya/VSCode/GeologicDome`), except `pose_and_render.py` which also runs
on native Windows. Installer: `bash sims/mujoco/scripts/setup_wsl.sh --all`.

**The `xmls/menagerie` junction/symlink is required** — MJCF `meshdir` resolves relative to the
scene file and cannot read env vars:

```powershell
New-Item -ItemType Junction -Path sims\mujoco\xmls\menagerie -Target C:\Users\Aditya\src\menagerie
```
```bash
ln -s ~/src/menagerie sims/mujoco/xmls/menagerie   # WSL alternative; setup_wsl.sh --base does this
```

(`check_render.py` and `inspect_model.py` are the exception: they load Menagerie's own scene
via the `MENAGERIE_DIR` env var, default `~/src/menagerie`, and never touch repo paths.)

## The gates

```bash
python sims/mujoco/scripts/inspect_model.py             # Phase 1: G1 structure asserts
python sims/mujoco/scripts/check_render.py              # offscreen EGL/OSMesa render probe
python sims/mujoco/terrain/drop_test.py                 # terrain contact gate (box drop)
python sims/mujoco/scripts/settle_g1_recon.py --asset loop_office --render  # robot-on-recon gate
python sims/mujoco/scripts/check_phase2.py              # JAX GPU + G1 env smoke
python sims/mujoco/scripts/check_full_collision.py      # full-collision model gate
python sims/mujoco/scripts/train_g1.py --smoke          # ~2 min end-to-end training proof
```

Training runs write to `runs/mujoco/` and append a row to `notes/experiments.md`
(success or failure) per the `training-run` skill.

## Ongoing job: sim2sim validation

Phase 7 of the pipeline (see `notes/pipeline.md`) validates exported policies in vanilla
MuJoCo at 50 Hz before deployment. That runs here, against these XMLs, regardless of which
simulator trained the policy.
