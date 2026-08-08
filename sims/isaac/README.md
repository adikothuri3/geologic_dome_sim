# `sims/isaac/` — the primary sim track (Isaac Sim + Isaac Lab)

> **Status: installed and smoke-tested 2026-08-07** — all three gates green on the local
> box (SimulationApp ~8 s headless; full-collision G1 builds and steps; 10-iteration
> RSL-RL smoke at ~690 steps/s). This is the primary simulator going forward — it mirrors
> the Robot Everest team's actual stack ("Mapping the full Everest route using Lingbot-Map
> into IsaacSim, with domain randomization over snow, ice friction, and wind gust"). See
> the 2026-08-07 entry in `notes/decisions.md` for the pivot rationale, and
> `notes/setup.md` for install state + the three Windows quirks (Norton TLS, EULA env var,
> Kit stdout hijack). The MuJoCo track lives on as working legacy + sim2sim validator in
> `sims/mujoco/`.

## Pinned versions (and why)

| Component | Version | Why |
| --- | --- | --- |
| Isaac Sim | **5.1.0** | pip-installable, native Windows 11, matches Isaac Lab 2.3.x |
| Isaac Lab | **2.3.x** (`main` branch) | last Windows-supported line. **Do NOT use Isaac Lab 3.0 beta** — its Newton/kit-less `develop` branch is Ubuntu-only as of mid-2026 |
| Python | **3.11** | required by Isaac Sim 5.1 (not 3.12 — the WSL/MJX venvs' 3.12 does not carry over) |
| RL library | RSL-RL (ships with Isaac Lab) | what the built-in G1 velocity tasks and `unitree_rl_lab` use |

No WSL: Isaac Sim 5.x runs **natively on Windows 11**. (The Omniverse Launcher is deprecated —
pip is the supported install path.)

## Install

Run `sims/isaac/setup_isaac.ps1` (idempotent). What it does:

1. Verifies `py -3.11` and `nvidia-smi` exist.
2. Creates `%USERPROFILE%\venvs\isaac` (matches the `~/venvs/*` convention).
3. `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
4. Clones Isaac Lab to `C:\Users\Aditya\src\IsaacLab` (matches the menagerie convention),
   checks out the 2.3.x line, runs `isaaclab.bat --install`.

## Smoke ladder (run in order; each step is a gate)

```powershell
$py = "$env:USERPROFILE\venvs\isaac\Scripts\python.exe"

# (a) SimulationApp opens headless, closes clean, versions printed.
#     First run is SLOW (shader cache) — be patient.
& $py sims\isaac\scripts\check_isaac.py --gate a

# (b) The FULL-COLLISION G1 flat task builds, resets, and survives zero-action
#     steps with finite observations. First run downloads the G1 USD assets.
& $py sims\isaac\scripts\check_isaac.py --gate b --num_envs 8

# (c) Every domain-randomization term in the DR task actually reaches PhysX,
#     measured as per-environment spread. 32 envs, so a spread means something.
& $py sims\isaac\scripts\check_isaac.py --gate c --num_envs 32

# (d) A 10-iteration training smoke completes, writes runs/isaac/<run_id>/,
#     and appends its notes/experiments.md row.
& $py sims\isaac\scripts\train_g1_flat.py --smoke
```

## The tasks (Phase 4a)

The stock `Isaac-Velocity-Flat-G1-v0` uses **`G1_MINIMAL_CFG`** — collision meshes
stripped for speed, the same feet-only shortcut Playground took and the same reason
it's wrong for this project. `sims/isaac/tasks/dome_g1/` registers three tasks, all on
`G1_CFG` (every link's collision geometry live):

| Task | What it is |
| --- | --- |
| `Dome-G1FullCollision-Flat-v0` | upstream's flat velocity config as-shipped. The **no-DR A/B control** |
| `Dome-G1FullCollision-Flat-DR-v0` | **the Phase-4a training task** — velocity tracking under the Phase-2 randomization set |
| `Dome-G1FullCollision-Flat-DR-Play-v0` | 16 envs, clean sensors, no pushes — what `play_g1_flat.py` evaluates in |

> [!warning] Upstream's G1 config randomizes almost nothing
> `G1RoughEnvCfg.__post_init__` *disables* most of Isaac's own randomization:
> `push_robot = None`, `add_base_mass = None`, `base_com = None`, reset velocities
> zeroed, joint-reset scale pinned to `(1.0, 1.0)`, and the physics-material ranges
> collapsed to point values (static `0.8→0.8`, dynamic `0.6→0.6`). What survives in the
> stock G1 flat task is observation noise and nothing else. That is why the DR variant
> exists as a separate task, and why gate (c) measures the randomization out of PhysX
> instead of trusting the config: a task that randomizes nothing trains, logs and plots
> exactly like one that does.

`DomeG1DREventCfg` (in `tasks/dome_g1/flat_env_cfg.py`) is the MuJoCo Phase-2
`domain_randomize` set rebuilt with Isaac EventManager terms — friction, per-link and
torso mass, torso CoM, armature, ±0.05 rad initial pose, and pushes every 5–10 s — plus
the joystick command distribution Playground used (direct `vx ±1.0 / vy ±0.5 / ωz ±1.0`,
no heading controller). Rewards and observations stay upstream's. Full term-by-term
mapping, including the two deliberate departures, is in that file's docstring and in
`notes/locomotion-policy.md`.

Training: `sims/isaac/scripts/train_g1_flat.py` — same run discipline as the legacy
`train_g1.py` (clean-tree gate, config capture, `runs/isaac/<run_id>/` logs, mandatory
`notes/experiments.md` row, success or failure). `--variant dr` is the default;
`--variant baseline` runs the control. Local default 256 envs headless; real runs
(4096 envs, 1500+ iterations) belong on a cloud GPU.

Evaluation: `sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id>` holds one velocity
command for a whole rollout and reports per-channel tracking MAE and survival across a
five-command sweep, writing `play_metrics.json` into the run dir. Those are the same
quantities `sims/mujoco/scripts/compare_gaits.py` produces, so an Isaac policy on flat
ground is comparable against the Phase-2 MuJoCo baseline and not only against itself.
`--video` additionally writes an mp4, at the cost of bringing the renderer up.

## This box vs. Isaac's minimums

Official minimum is **16 GB VRAM / 32 GB RAM**; this machine has **8 GB VRAM / 16 GB RAM** —
below spec on both. The plan decided 2026-08-07 was "local = prototyping only, `num_envs`
64–256; cloud = real training". **Half of that was wrong, and it cost a training run.**

Measured 2026-08-08 with `check_isaac.py --gate b`, full-collision G1, headless:

| envs | VRAM used | throughput |
| --- | --- | --- |
| 256 | 3.40 GB | 3,011 env-steps/s |
| 1024 | 3.77 GB | 10,856 |
| 2048 | 4.21 GB | 19,480 |
| **4096** | **5.05 GB** of 8.00 | **28,975** |

VRAM grows sub-linearly — 16× the envs costs 1.5× the memory — because the G1's USD is
instanced, so what scales with `num_envs` is PhysX state, not geometry. Upstream's own
4096-env config runs here with ~3 GB to spare at ~3.7 s/iteration.

Why it mattered: the first Phase-4a attempt honoured the 256-env guess, and 256 × 1500 is
**9.2M samples against upstream's 147M** — 16× short. It reached 100 % survival with velocity
tracking still at exactly zero (MAE equal to the command magnitude on every nonzero channel,
0.02 on stand-still): a policy that had learned to stand and not to walk, which is the ordering
a `−200` termination penalty buys. The same task at 4096 envs trains. Both rows are in
`notes/experiments.md`.

What survives from the original plan:

- **Always `--headless`.** The RTX renderer is the thing that genuinely does not fit here —
  it alone eats ~7 GB. Use `--video` for short eval clips only, never a viewport.
- **Cloud is still right for Phase 5**, where sweeps mean several runs at once and terrain
  scenes add mesh collision on top of the articulation. It is no longer required to train a
  single flat-plane policy.

### The cloud run

A box with **≥24 GB VRAM** (Lambda / Brev / AWS; an A10, L40S or A100 all work). Isaac's own
minimum is 16 GB, and the headroom matters because Phase 5 adds terrain mesh collision.

```bash
git clone <repo> && cd GeologicDome
bash sims/isaac/setup_isaac_cloud.sh            # ~10 GB download, the slow step
export OMNI_KIT_ACCEPT_EULA=YES
PY=~/venvs/isaac/bin/python
```

**1 — Prove the randomization is live.** Seconds, and it is the difference between paying for
a randomized policy and paying for one trained on a single fixed robot. Every term is measured
out of PhysX as spread *across environments*; it does not read the config back.

```bash
$PY sims/isaac/scripts/check_isaac.py --gate c --num_envs 32
```

**2 — Train.** 4096 envs × 1500 iterations is upstream's own recipe (147M samples).

```bash
$PY sims/isaac/scripts/train_g1_flat.py --num_envs 4096 --max_iterations 1500
```

> [!warning] Watch `feet_air_time`, and kill the run early if it stays flat
> This is the failure the 2026-08-08 local run hit, and it is invisible in mean reward —
> which kept *rising* the whole time, on the yaw term, while the robot pivoted on the spot
> and never took a step. The tell is one reward term:
>
> ```bash
> grep "Episode_Reward/feet_air_time" nohup.out | tail -20
> ```
>
> **Healthy:** climbing past ~0.05 by iteration 300–500 and still rising, with
> `track_lin_vel_xy_exp` above 0.5.
> **Dead:** pinned near 0.01 while `track_ang_vel_z_exp` climbs. In the failed run it sat at
> ~0.01 from iteration 199 to 1599 — flat for 1,400 iterations. If you see that shape by
> iteration ~500, kill it; more compute provably does not fix it (`notes/decisions.md`,
> 2026-08-08).
>
> `feet_air_time_joystick` in `tasks/dome_g1/mdp.py` is the fix for that failure and it is
> **not yet validated** — this run is its first real test. If it *does* stay flat, the next
> thing to try is `action_rate_l2` (it was the dominant term at −0.41), accepting that
> lowering it trades gait smoothness for locomotion.

**3 — Score it, render it, plot it.**

```bash
$PY sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id> --video
$PY sims/isaac/scripts/plot_play.py    runs/isaac/<run_id>
```

`play_metrics.json` + the terminal table give per-command tracking MAE, survival, and the two
smoothness numbers; `--video` writes an mp4 of the sweep; `plot_play.py` writes
`reports/<run_id>-tracking.png`. A policy that works reads as MAE well under the command
magnitude on every channel — the failed run scored MAE **equal** to it, which is the signature
of zero motion.

**Optional A/B.** `--variant baseline` runs upstream's config (no dynamics DR, forward-only
commands, heading control) on the same robot. Worth one run if the DR variant misbehaves,
because it separates "our randomization is too aggressive" from "our task definition is wrong".

Nothing in these scripts is local-only: paths are repo-relative, the app is always headless,
and the `notes/experiments.md` row is written before teardown either way.

If a run is interrupted, `--resume` continues it from the highest checkpoint in a run directory,
restoring weights, optimizer state and iteration count. `--max_iterations` is an **absolute
target**, not a delta, so the same number means the same thing whether or not it took two goes:

```bash
python sims/isaac/scripts/train_g1_flat.py --resume runs/isaac/<run_id> --max_iterations 1500
```

> [!warning] After `AppLauncher`, `sys.exit("message")` is not an exit
> Kit rebinds `sys.exit` to pybind11's `post_quit()`, which takes an int — a string argument
> raises `TypeError`, and in `train_g1_flat.py` that got caught and logged as a FAILED training
> run. Any script here that bails with a message after the app starts must
> `raise SystemExit(msg)`. Before the app starts, plain `sys.exit` is fine.

## Key APIs for the phases ahead

- **G1 asset:** `isaaclab_assets/robots/unitree.py` — `G1_CFG` (**full collision**, our
  requirement) and `G1_MINIMAL_CFG` (simplified). Tasks: `Isaac-Velocity-Flat-G1-v0`,
  `Isaac-Velocity-Rough-G1-v0` (+ `-Play-v0` variants).
- **Unitree's official Isaac Lab repo:** `unitreerobotics/unitree_rl_lab` (Isaac Lab 2.3.x,
  RSL-RL, `Unitree-G1-29dof-Velocity`) — the reference for sim2real-tested G1 configs.
- **Terrain import (Phase 4b):** `TerrainImporterCfg.terrain_type ∈ plane | generator | usd`.
  Mesh route: LingBot cloud → Open3D mesh → **OBJ** (MeshConverter accepts OBJ/STL/FBX,
  **not PLY**) → USD via `MeshConverterCfg(collision_approximation="triangleMesh")`.
  Procedural fallback: `TerrainGeneratorCfg` + `HfRandomUniformTerrainCfg` /
  `HfPyramidSlopedTerrainCfg` (mountain-ish rough terrain without a reconstruction).
- **Domain randomization:** `EventManager` / `EventTermCfg` — `randomize_rigid_body_material`
  (snow/ice friction), `randomize_rigid_body_mass`, `push_by_setting_velocity` (wind-gust
  proxy). Modes: `startup`, `reset`, `interval`.

## Layout

- `scripts/` — `check_isaac.py` (smoke gates a/b/c), `train_g1_flat.py` (Phase 4a trainer),
  `play_g1_flat.py` (rollout + velocity-tracking scorer)
- `tasks/dome_g1/` — the full-collision G1 task registrations + configs, including
  `DomeG1DREventCfg`, the domain-randomization set
- `terrain/` — Phase 4b converters (cloud → OBJ → USD; procedural configs) — placeholder

## Claude skills for this track

Five official NVIDIA skills from `isaac-sim/isaacsim` are installed in `.claude/skills/`
(copy-mode, no symlinks): `isaac-sim-headless-deployment`, `isaac-sim-troubleshooting`,
`isaac-sim-validator`, `physics-simulation`, `urdf-mjcf-to-usd-conversion`. Note: they
document Isaac Sim 6.0/Kit 110 — most content applies to our pinned 5.1, but check
version-specific claims (e.g. Newton solver backends are 6.0-only) before acting on them.
