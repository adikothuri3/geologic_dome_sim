---
title: Machine setup — RTX 4060 Ti + WSL2
updated: 2026-08-03
status: current
---

# Machine setup

The dev box for the whole home-phase pipeline (see [[pipeline]]). One machine, 8 GB VRAM, everything else is tactics.

## Hardware & OS (verified 2026-08-02)

- **GPU:** NVIDIA GeForce RTX 4060 Ti, **8 GB VRAM** (8188 MiB), driver 610.62
- **CPU:** AMD Ryzen 7 5700 (8 cores) · **Board:** Gigabyte B550 UD AC, AMI BIOS FEc
- **OS:** Windows 11 Home 10.0.26200, UEFI
- **Installed on Windows:** Python 3.12.8, git 2.55.0

## WSL2 install state (2026-08-03) — complete

SVM was off in firmware, which is why `wsl --install` did nothing for a while — Gigabyte
ships B550 boards that way, and no OS-side workaround exists on consumer Gigabyte boards
(no WMI BIOS provider, unlike Dell/Lenovo/HP business lines). Enabled by hand at
**Tweaker → Advanced CPU Settings → SVM Mode**.

| Step | State |
| --- | --- |
| AMD SVM in firmware | ✅ `VirtualizationFirmwareEnabled = True` |
| WSL features + runtime | ✅ 2.7.11, kernel 6.18.33.2 |
| Ubuntu 24.04 distro | ✅ default user `aditya`, `systemd=true` |
| GPU passthrough | ✅ `nvidia-smi` in WSL: RTX 4060 Ti, 8188 MiB, driver 610.62 |
| Offscreen rendering | ✅ **hardware EGL, first try** — no NVIDIA ICD file, no osmesa fallback |
| JAX-CUDA | ✅ jax **0.9.2, pinned** (see below), `[CudaDevice(id=0)]` |
| Playground + MJX | ✅ playground 0.2.0, brax 0.14.2, `G1JoystickFlatTerrain` loads |
| Open3D (Phase 4) | ✅ 0.19.0 |
| Phase 1 parity on Linux | ✅ all three scripts pass, MP4 renders |

Reproduce from scratch with `scripts/setup_wsl_stage2.ps1` (Windows: distro + user) then
`scripts/setup_wsl.sh --all` (Linux toolchain). Stage 2 installs Ubuntu with `--no-launch`,
because the normal first run opens an interactive console asking for a username and would
hang an unattended script.

> [!warning] WSL's network stalls connections, not transfers
> Roughly **one in five fresh HTTPS connections** from this WSL instance to GitHub hangs
> until timeout, while sustained transfers run fine at 6 MB/s. This killed three install
> attempts at three different steps before it was understood. `setup_wsl.sh` now wraps every
> network call in `retry()` and exports `GIT_HTTP_LOW_SPEED_TIME=20` so a dead socket aborts
> in 20 s instead of hanging. If an install stalls silently, this is the first suspect.

> [!warning] jax is pinned to 0.9.2 — do not `-U` it
> brax 0.14.2 (newest, and what Playground requires) still calls `jax.device_put_replicated`,
> which jax **removed in 0.10.0**. Unpinning resolves to 0.11.0 and every PPO run dies at
> startup with `AttributeError`. See [[decisions]]; unpin only when brax ships a fix, and
> re-verify with `python scripts/train_g1.py --smoke`.

Two consequences worth knowing before debugging anything in WSL:

- **`/tmp` does not survive.** `systemd=true` mounts it as tmpfs, so an idle-terminated
  distro takes the logs with it. Write install logs to a `/mnt/c/...` path.
- **Playground clones its own Menagerie** (pinned commit `1b86ece`, ~500 MB) into
  `~/src/playground/mujoco_playground/external_deps/` on the *first* `registry.load()`. That
  is a one-time cost, already paid; a first training run on a fresh box will pay it again and
  can look like a hang.

## The robot: Menagerie `unitree_g1`, and only that

> [!important] One source of robot geometry
> Every simulated G1 in this project is Menagerie's `unitree_g1` — meshes, kinematics,
> inertials — with our own MJCF layered on top. **If it isn't in
> `mujoco_menagerie/unitree_g1/assets`, it isn't the robot.** Playground's
> `g1_mjx_feetonly.xml` references those STLs directly, so its G1 and the Phase 1 G1 are the
> same machine; Playground only adds MJX-friendly collision primitives, sensors and actuators.
> See [[decisions]].

Our own layers, all in `sim/`, all generated rather than hand-drawn:

| File | Made by | What it is |
| --- | --- | --- |
| `scene_g1_hfield.xml` | hand, Phase 1 | G1 on a numpy heightfield, no floor plane |
| `g1_full_collision.xml` | `scripts/make_full_collision_xml.py` | every link collidable; each box sized from that body's Menagerie mesh |
| `scene_g1_full_collision.xml` | hand | the flat-terrain scene, including the above instead of the feet-only body |

Gate the generated model with `python scripts/check_full_collision.py` before training on it:
it asserts the environment contract survives, that nothing self-collides at the nominal pose,
that only the feet touch the ground when settled, and that a toppled robot is actually caught
by the new geometry.

## The Windows side (still live, still useful)

Phase 1 was built here while WSL2 was blocked on firmware, and it still works: MuJoCo physics
is CPU and Windows renders through WGL with `MUJOCO_GL` unset (see [[decisions]]). Keep it —
it needs no hypervisor, it is the fallback if WSL breaks, and its Menagerie clone is what
seeds the Linux one. Phase 1 now passes identically on both.

- **`.venv`** at the repo root: `mujoco 3.11.0`, `numpy 2.5.1`, `imageio`, `imageio-ffmpeg`
- **Menagerie** sparse-cloned to `C:\Users\Aditya\src\menagerie` (`unitree_g1` only), exposed
  to the scene as the `sim/menagerie` junction
- Video is written via `imageio` + `imageio-ffmpeg`, **not** `mediapy` — mediapy shells out to
  a system ffmpeg that Windows does not have; imageio bundles its own binary and is identical
  on both OSes

## The Linux toolchain (`scripts/setup_wsl.sh`)

JAX-CUDA does not run natively on Windows and DimOS targets Linux, so everything from
Phase 2 on lives in WSL. Stages are selectable and later ones are non-fatal — a DimOS
failure must not cost a working Phase 2 box:

| Flag | Installs | Into | For |
| --- | --- | --- | --- |
| `--base` | apt GL libs, uv, py3.12 venv, MuJoCo, Menagerie, render probe | `~/venvs/dome` | Phase 1 parity |
| `--phase2` | `jax[cuda12]`, MuJoCo Playground (editable clone), MJX | `~/venvs/dome` | locomotion policy |
| `--phase4` | Open3D | `~/venvs/dome` | point cloud → terrain |
| `--phase6` | DimOS | **`~/venvs/dimos`** | robot OS integration |
| `--all` | all of the above | | |

DimOS lives in its own venv on purpose — see [[decisions]]. The Menagerie step copies
`unitree_g1` out of the Windows clone at `C:\Users\Aditya\src\menagerie` when one exists, so
the common path makes no network call at all.

The render probe tries `egl`, installs the NVIDIA EGL ICD at
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` if glvnd can't find the WSL driver, then
falls back to `osmesa`, persisting whichever works to `~/.bashrc`.

## Phase 2 VRAM budget

**The usable budget is ~6 GB, not 8.** This GPU also drives the Windows desktop, which holds
roughly 2 GB. JAX preallocates 75 % of total VRAM by default — 0.75 × 8188 MiB ≈ 6.0 GiB —
and that allocation *fails*, after which XLA retries down a ladder (5.4 → 4.9 → 4.4 GiB) and
proceeds with a fragmented pool. So `check_phase2.py` and `train_g1.py` both set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before JAX initialises its backend, and allocate on
demand instead. JAX then reports `vram 6.0 GiB`.

Playground's G1 defaults to **8192 parallel envs**, which will OOM.
`scripts/train_g1.py` caps it at 2048 per the `training-run` skill and preserves brax's
`batch_size × num_minibatches == num_envs` relation (upstream `256 × 32 = 8192`) by holding
`num_minibatches = 32` and deriving `batch_size` — so gradient maths is unchanged and only
parallelism shrinks. Watch `nvidia-smi` in the first minutes; at the wall drop to 1024 envs
first, cloud second. Closing VS Code buys back most of a gigabyte.

## What runs on 8 GB, and how

| Workload | Runs locally? | How |
| --- | --- | --- |
| MuJoCo (CPU physics + viewer) | Yes, easily | Core physics is CPU-based; GPU only helps rendering. Phases 1 and 4 fully local. |
| LingBot-Map inference | Yes | Runs at 518×378. Control KV-cache growth with `keyframe_interval`, use windowed mode, filter by confidence. 8 GB handles trail-length clips; chunk very long videos. |
| MJX / Playground RL training | Yes, with settings | Cut parallel envs (8192 → 1024–2048) and batch size; G1/Go1 joystick policies still train locally, just slower. Big Phase 5 sweeps → cloud GPU or Colab Pro. |
| Open3D / point-cloud work | Yes, easily | Mostly CPU + light GPU. |
| DimOS + replay datasets | Yes | Replay needs no hardware; agents/modules are CPU-light. |

> [!warning] The two VRAM eaters
> LingBot-Map's KV cache and the MJX parallel-env count are the two things that hit the 8 GB ceiling. Keyframe/windowed mode for recon; fewer envs for training; rent a cloud GPU for Phase 5 sweeps and during the expedition window. Whether 8 GB suffices end-to-end is an open question — see [[open-questions]].

## Claude Code tooling (installed 2026-08-01)

**MCP servers** — user scope (`claude mcp add -s user -t http …`), both verified connected via `claude mcp list`. Neither needs an API key or filesystem access.

- **deepwiki** (`https://mcp.deepwiki.com/mcp`) — repo-level Q&A over the codebases this pipeline sits on: `dimensionalOS/dimos`, `google-deepmind/mujoco`, `mujoco_playground`, `mujoco_menagerie`, LingBot-Map, `isl-org/Open3D`.
- **context7** (`https://mcp.context7.com/mcp`) — version-current API docs; query before writing code against `mujoco`, `mjx`, `jax`, `open3d`, or `onnx` APIs so calls aren't stale.

**Custom skills** in `.claude/skills/` (project conventions, one page each — update them as conventions change):

- `mjcf-terrain` — point cloud → hfield/mesh → MJCF, Phase 4 conventions (5–10 cm cells, robust max-z, hole filling, <200k faces, scale calibration, contact checks)
- `open3d-cleanup` — outlier removal, voxel downsample, ground-plane alignment defaults
- `training-run` — MJX fine-tune wrapper: 8 GB env-count limits, config + commit capture, mandatory [[experiments]] row
- `lingbot-recon` — reconstruction defaults: `keyframe_interval`, windowed mode, confidence filtering, ≤10-min chunking

`obsidian-markdown` (from kepano/obsidian-skills) keeps vault notes valid Obsidian-flavored markdown. The rest of that bundle (`obsidian-cli`, `obsidian-bases`, `json-canvas`, `defuddle`) was removed as unused — reinstall from the same repo if ever needed.

**Third-party skills** (from the skills.sh registry, reviewed before install):

- `mujoco` (coolbeevip/mujoco-skills) — MJCF scene building + robot-control/viewer workflows with helper scripts; useful for Phase 1 fluency and scene debugging. Installed to `.agents/skills/mujoco` (junction at `.claude/skills/mujoco`); security scans clean (Gen/Socket/Snyk).
- Rejected: `letta-ai@tune-mjcf` (deleted upstream since registry indexing — nothing to install), `plurigrid@urdf2mjcf` (thin content, security warning, URDF conversion not on our path — Menagerie ships `unitree_g1` as MJCF), `onnx-converter` (auto-generated boilerplate). Nothing relevant found for open3d / point-cloud.

**Subagents** in `.claude/agents/` (added 2026-08-01, one page each — delegation targets for the main agent):

- `docs-researcher` — read-only external-library lookups (mujoco, mjx, playground, dimos, lingbot-map, open3d) via deepwiki/context7/web; returns a sourced synthesis instead of doc dumps
- `terrain-validator` — QA gate for real2sim terrain assets (scale, cell size, holes, face budget, MJCF load, G1 settle test, slope/roughness stats); writes reports under `reports/` only
- `run-auditor` — after every training/reconstruction run, parses logs and appends the mandatory [[experiments]] row; flags reward hacking, train/eval divergence, VRAM near the 8 GB ceiling
- `vault-keeper` — keeps this vault synced with code changes at end of session; edits `notes/` only

Note: `npx skills add` needed `http.sslBackend=schannel` (via `GIT_CONFIG_*` env vars, not persisted) — the global git config pins `openssl` + Git's CA bundle, which fails TLS verification against GitHub on this box.

## Repo layout

- `notes/` — this Obsidian vault (documentation only)
- `lab-notebook/` — weekly markdown lab notebook, outside the vault
- `scripts/` — `setup_wsl_stage2.ps1` (Windows: distro + user), `setup_wsl.sh` (Linux env),
  `check_render.py` (GL gate), `inspect_model.py`, `check_phase2.py` (JAX GPU + G1 env),
  `train_g1.py` (Phase 2 training, writes the [[experiments]] row),
  `make_full_collision_xml.py` + `check_full_collision.py` (full-body collision model and its
  gate), `render_policy.py` (trained policy → MP4)
- `terrain/` — `make_hfield.py` (numpy → `hfield_data`, `sample_height`), `drop_test.py` (contact gate)
- `sim/` — `scene_g1_hfield.xml` (G1 + heightfield, no floor plane), `pose_and_render.py` (the demo)
- `runs/` — per-run `config.json` + `progress.json` + checkpoints (gitignored)
- `.venv/`, `sim/menagerie`, `runs/`, `*.mp4` are gitignored
