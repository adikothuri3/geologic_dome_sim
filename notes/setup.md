---
title: Machine setup — RTX 4060 Ti + WSL2
updated: 2026-08-02
status: current
---

# Machine setup

The dev box for the whole home-phase pipeline (see [[pipeline]]). One machine, 8 GB VRAM, everything else is tactics.

## Hardware & OS (verified 2026-08-02)

- **GPU:** NVIDIA GeForce RTX 4060 Ti, **8 GB VRAM** (8188 MiB), driver 610.62
- **CPU:** AMD Ryzen 7 5700 (8 cores) · **Board:** Gigabyte B550 UD AC, AMI BIOS FEc
- **OS:** Windows 11 Home 10.0.26200, UEFI
- **Installed on Windows:** Python 3.12.8, git 2.55.0

## WSL2 install state (2026-08-02)

SVM was off in firmware, which is why `wsl --install` did nothing for a while — Gigabyte
ships B550 boards that way, and no OS-side workaround exists on consumer Gigabyte boards
(no WMI BIOS provider, unlike Dell/Lenovo/HP business lines). Enabled by hand at
**Tweaker → Advanced CPU Settings → SVM Mode**.

| Step | State |
| --- | --- |
| AMD SVM in firmware | ✅ `VirtualizationFirmwareEnabled = True` |
| `Microsoft-Windows-Subsystem-Linux` feature | ✅ Enabled |
| `VirtualMachinePlatform` feature | ✅ Enabled |
| WSL runtime | ✅ 2.7.11 installed |
| **Reboot** | ⬜ **required — features are not live until then** |
| Ubuntu 24.04 distro | ⬜ pending (`scripts/setup_wsl_stage2.ps1`) |
| JAX-CUDA + Playground | ⬜ pending (`scripts/setup_wsl.sh --all`) |

After the reboot, one command finishes everything:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_wsl_stage2.ps1
```

It installs Ubuntu with `--no-launch` (the normal first run opens an interactive console
asking for a username, which would hang an unattended script), creates the `aditya` user
non-interactively, then hands off to `scripts/setup_wsl.sh --all` inside WSL.

## What actually runs today (Phase 1, native Windows)

Phase 1 needs no GPU, no JAX and no hypervisor — MuJoCo physics is CPU and Windows renders
through WGL with `MUJOCO_GL` unset. So Phase 1 was built and demoed natively while WSL2
stays blocked (see [[decisions]]).

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

| Flag | Installs | For |
| --- | --- | --- |
| `--base` | apt GL libs, uv, py3.12 venv, MuJoCo, Menagerie, render probe | Phase 1 parity |
| `--phase2` | `jax[cuda12]`, MuJoCo Playground (editable clone), MJX | locomotion policy |
| `--phase4` | Open3D | point cloud → terrain |
| `--phase6` | DimOS | robot OS integration |
| `--all` | all of the above | |

The render probe tries `egl`, installs the NVIDIA EGL ICD at
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` if glvnd can't find the WSL driver, then
falls back to `osmesa`, persisting whichever works to `~/.bashrc`.

## Phase 2 VRAM budget

Playground's G1 defaults to **8192 parallel envs**, which will OOM on 8 GB.
`scripts/train_g1.py` caps it at 2048 per the `training-run` skill and preserves brax's
`batch_size × num_minibatches == num_envs` relation (upstream `256 × 32 = 8192`) by holding
`num_minibatches = 32` and deriving `batch_size` — so gradient maths is unchanged and only
parallelism shrinks. Watch `nvidia-smi` in the first minutes; at the wall reduce envs
first, cloud second.

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
  `train_g1.py` (Phase 2 training, writes the [[experiments]] row)
- `terrain/` — `make_hfield.py` (numpy → `hfield_data`, `sample_height`), `drop_test.py` (contact gate)
- `sim/` — `scene_g1_hfield.xml` (G1 + heightfield, no floor plane), `pose_and_render.py` (the demo)
- `runs/` — per-run `config.json` + `progress.json` + checkpoints (gitignored)
- `.venv/`, `sim/menagerie`, `runs/`, `*.mp4` are gitignored
