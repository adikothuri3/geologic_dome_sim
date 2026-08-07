# `sims/isaac/` — the primary sim track (Isaac Sim + Isaac Lab)

> **Status: scaffolded 2026-08-07, not yet installed.** This is the primary simulator going
> forward — it mirrors the Robot Everest team's actual stack ("Mapping the full Everest route
> using Lingbot-Map into IsaacSim, with domain randomization over snow, ice friction, and wind
> gust"). See the 2026-08-07 entry in `notes/decisions.md` for the full pivot rationale.
> The MuJoCo track lives on as working legacy + sim2sim validator in `sims/mujoco/`.

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
# (a) Isaac Sim opens headless and closes clean. First run is SLOW (shader cache) — be patient.
& $env:USERPROFILE\venvs\isaac\Scripts\python.exe -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); app.close()"

# (b) The built-in G1 velocity task loads and steps with zero actions.
python scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 8   # from the IsaacLab checkout

# (c) A 10-iteration training run completes and writes logs.
python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 64 --max_iterations 10
```

Point training logs at `runs/isaac/<YYYY-MM-DD-slug>/` and append a row to
`notes/experiments.md` for every run — success or failure — per the `training-run` skill.

## This box vs. Isaac's minimums

Official minimum is **16 GB VRAM / 32 GB RAM**; this machine has **8 GB VRAM / 16 GB RAM** —
below spec on both. The working plan (decided 2026-08-07):

- **Local = prototyping only.** Always `--headless` (the RTX renderer alone eats ~7 GB),
  `num_envs` 64–256, `--/app/content/emptyStageOnStart=true`, close browsers/VS Code first.
  Local answers "does it load and step", not "train a policy".
- **Cloud = real training.** Phase 4a/5 training runs go to a rented GPU (Lambda/Brev/AWS,
  ≥24 GB VRAM). This is the documented normal path even for better-equipped boxes.
- If smoke step (a) fails on 16 GB RAM, that is data for going cloud-first, not a blocker.

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

- `tasks/` — Phase 4a task/env configs (G1 velocity overrides, unitree_rl_lab integration)
- `terrain/` — Phase 4b converters (cloud → OBJ → USD; procedural configs)

Both are placeholders until the Isaac work starts.
