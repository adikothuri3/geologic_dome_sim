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
| `Dome-G1FullCollision-Flat-Heading-v0` | the same DR physics under **upstream's task definition** (heading-derived yaw, forward-only, upstream's `feet_air_time` gate). The **positive control**, added 2026-08-08 |
| `Dome-G1FullCollision-Flat-DR-Play-v0` | 16 envs, clean sensors, no pushes — what `play_g1_flat.py` evaluates in |
| `Dome-G1FullCollision-Flat-Heading-Play-v0` | the control's eval variant. `heading_command` is **off** here, so the eval harness's command pin actually holds and the control is scored on the same sweep as everything else |

> [!note] The heading task is an instrument, not a candidate
> A joystick policy whose yaw you cannot command directly is not the Phase-2 task and is not
> what DimOS will drive — rejected on those grounds in `notes/decisions.md` and still rejected.
> It exists so that "our task is hard" can be told apart from "our harness is broken": two
> full-size runs have now failed to produce locomotion, and until upstream's own recipe is
> shown to walk on this robot and this randomization, neither failure can be attributed.

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
`notes/experiments.md` row, success or failure), plus the guards below. `--variant dr` is
the default; `--variant heading` is the positive control and `--variant baseline` the no-DR
one. Defaults are **4096 envs x 3000 iterations** — upstream's own env count, which fits in
5.05 GB here (see the table below).

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

### The guards (`scripts/train_guards.py`)

`DomeOnPolicyRunner` is RSL-RL's `OnPolicyRunner` with `log()` overridden — nothing about PPO
changes. It closes the two gaps the legacy MJX trainer did not have:

| File | What it is |
| --- | --- |
| `progress.jsonl` | one JSON line per iteration, **appended**. An external kill and a Colab disconnect are both SIGKILL, which no `finally` block survives — that is how the 1,792-iteration run ended up with no experiments row at all. A line already on disk cannot be lost |
| `best.json` + `model_best.pt` | the best checkpoint, MuJoCo parity. `play_g1_flat.py --checkpoint best` loads it |
| `outcome.json` | the machine-readable verdict — `ok`, `collapsed` or `FAILED` — for an automated driver |

> [!important] The best checkpoint is chosen by reward *terms*, never by mean reward
> ```
> walk_score = Episode_Reward/track_lin_vel_xy_exp + Episode_Reward/track_ang_vel_z_exp
> gate       = Episode_Reward/feet_air_time >= 0.02
> ```
> Mean reward rose −30 → +4.11 across the whole of the failed 4096-env run, entirely on the
> yaw term, while the robot never took a step. `best.json` carries `"gated": false` when the
> run never stepped, so a non-walking policy cannot present itself as the run's best.

`--abort-if-flat 500` (the default) automates the manual criterion below: if `feet_air_time`
has not lifted above 0.02 anywhere in the trailing 200 iterations, the run stops itself. A
provably dead run costs ~25 minutes instead of three hours.

> [!warning] The exit code of these scripts is not under their control
> Measured 2026-08-08: `raise SystemExit(3)` after `simulation_app.close()` exits **0**. Kit
> owns process shutdown — the same reason `sys.exit("message")` raises `TypeError` inside a
> running app. Anything automating these scripts must read `outcome.json`, not `$?`. A missing
> `outcome.json` means the process never reached its own teardown, which is the case to retry.

### Colab — `colab/isaac_g1_flat_colab.ipynb`

Runs the whole Phase-4a experiment (all three variants, scored, with video) on a Colab GPU.
It shells out to these same scripts unmodified, and clones the repo at a pinned ref rather
than carrying its own copy.

> [!important] Training does not use the renderer, and now says so
> `check_isaac.py` and `train_g1_flat.py` both launch Kit with
> **`--/app/renderer/enabled=false`** (opt out with `--keep-renderer`). Not a micro-
> optimisation: `isaaclab.python.headless.kit` still sets `renderer.enabled = "rtx"`, so
> "headless" alone leaves the RTX subsystem loaded for frames nobody ever looks at.
> Measured on the dev box, renderer off vs on:
>
> | | renderer on | renderer off |
> | --- | --- | --- |
> | gate b @ 4096 envs | 28,975 env-steps/s | **33,596 env-steps/s** |
> | gate a startup | ~8 s | **6.5 s** |
> | 10-iteration smoke, final mean reward | −7.21169098127972 | −7.21169098127972 (identical) |
>
> The reward matching to the last digit is the point: disabling the renderer changes the
> *speed* and nothing about the *result*. It is also what makes a datacentre GPU work — see
> below. `play_g1_flat.py --video` is unaffected; it is a separate script and genuinely does
> render.

> [!warning] The A100 question, precisely
> NVIDIA's Isaac Sim 5.1 requirements page states verbatim: *"GPUs without RT Cores (A100,
> H100) are not supported."* That sentence is about the **RTX renderer**, and it does not
> mean training fails:
>
> | | A100 / H100 | L4, L40S, A10, RTX |
> | --- | --- | --- |
> | headless training (PhysX + CUDA, no renderer) | **reported working**, and faster | works |
> | `--video` / `--enable_cameras` (offscreen RTX) | **froze** on an A100-PCIE-40GB in [IsaacLab #2584](https://github.com/isaac-sim/IsaacLab/issues/2584) — same flags as ours, with `Vulkan 1.1 is not supported` | works |
>
> **Train on the A100.** With `--/app/renderer/enabled=false` the subsystem NVIDIA calls
> unsupported is never loaded, so the gates and the trainer have nothing left to trip over;
> the statement above constrains `--video` and nothing else. Render on an L4 afterwards —
> `runs/` on Drive makes that a runtime switch, not a retrain. The notebook times a hung
> render out at 30 minutes and still writes the tracking numbers, which need no renderer.
> Note also that Colab has been observed substituting L4 for a requested A100.

Two things Colab needs that a rented box does not, both in `setup_colab_gpu.sh`:
**Python 3.11** (Colab ships 3.12, for which no `isaacsim` wheel exists — pip then reports the
package as unfindable, which reads like a typo) and the **Vulkan ICD / EGL vendor manifests**
(Colab ships the driver libraries but not the JSON that tells the loaders where they are, so
Kit finds no device). The ICD contents follow `j3soon/isaac-sim-colab`, the only documented
working Isaac-on-Colab recipe — note it targets **4.5**, not our pinned 5.1, so treat the
first run as unproven. The notebook's preflight cell is a hard gate on driver ≥ 580.65.06 and
≥ 40 GB free, and names the fallback: a rented L40S/A10, where `setup_isaac_cloud.sh` runs
unchanged.

### The cloud run

A box with **≥24 GB VRAM** (Lambda / Brev / AWS; an A10 or L40S both work — prefer these over
an A100 for the RT-core reason above). Isaac's own minimum is 16 GB, and the headroom matters
because Phase 5 adds terrain mesh collision.

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

**2 — Train, all three variants.** 4096 envs × 3000 iterations is 295M samples, twice
upstream's own flat recipe. The budget is generous because compute is no longer the
constraint and "undertrained" has already been blamed once too often.

```bash
# A — the gate fix, the hypothesis under test
$PY sims/isaac/scripts/train_g1_flat.py --num_envs 4096 --max_iterations 3000

# C — the positive control: upstream's task on our randomized physics
$PY sims/isaac/scripts/train_g1_flat.py --variant heading --num_envs 4096 --max_iterations 3000

# B — the fallback lever, only interesting if A stays flat
$PY sims/isaac/scripts/train_g1_flat.py --reward-scale action_rate_l2=-0.001 \
    --num_envs 4096 --max_iterations 3000
```

> [!warning] Watch `feet_air_time`, not mean reward
> This is the failure the 2026-08-08 run hit, and it is invisible in mean reward — which kept
> *rising* the whole time, on the yaw term, while the robot pivoted on the spot and never took
> a step. The tell is one reward term:
>
> ```bash
> python -c "import json,sys; [print(json.loads(l)['iteration'], json.loads(l)['feet_air_time']) \
>   for l in open('runs/isaac/<run_id>/progress.jsonl')][-20:]"
> ```
>
> **Healthy:** climbing past ~0.05 by iteration 300–500 and still rising, with
> `track_lin_vel_xy_exp` above 0.5 (standing still scores **0.37**).
> **Dead:** pinned near 0.01 while `track_ang_vel_z_exp` climbs. In the failed run it sat at
> ~0.01 from iteration 199 to 1599 — flat for 1,400 iterations.
>
> **`--abort-if-flat` now enforces this**, so you no longer have to watch: the run stops
> itself and writes `outcome.json` with `"status": "collapsed"`. It is a *result*, not a
> crash, and gets a normal experiments row saying which lever to pull next.
>
> `feet_air_time_joystick` in `tasks/dome_g1/mdp.py` is the fix for that failure and it is
> **not yet validated** — run A is its first real test.

**3 — Score it, render it, plot it.**

```bash
$PY sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id> --checkpoint best --video
$PY sims/isaac/scripts/plot_play.py    runs/isaac/<run_id>
```

`--checkpoint best` loads `model_best.pt` rather than the last checkpoint; `--video` writes
**one clip per command** into `<run_dir>/videos/` with an `index.json` mapping clip to command.
The eval task is read from the run's own `config.json`, so the heading control is
automatically scored in its own play variant.

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
  `train_guards.py` (best-checkpoint tracking + the collapse/NaN watchdogs),
  `play_g1_flat.py` (rollout + velocity-tracking scorer), `plot_play.py`
- `tasks/dome_g1/` — the full-collision G1 task registrations + configs, including
  `DomeG1DREventCfg`, the domain-randomization set
- `terrain/` — Phase 4b converters (cloud → OBJ → USD; procedural configs) — placeholder

## Claude skills for this track

Five official NVIDIA skills from `isaac-sim/isaacsim` are installed in `.claude/skills/`
(copy-mode, no symlinks): `isaac-sim-headless-deployment`, `isaac-sim-troubleshooting`,
`isaac-sim-validator`, `physics-simulation`, `urdf-mjcf-to-usd-conversion`. Note: they
document Isaac Sim 6.0/Kit 110 — most content applies to our pinned 5.1, but check
version-specific claims (e.g. Newton solver backends are 6.0-only) before acting on them.
