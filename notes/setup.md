---
title: Machine setup — RTX 4060 Ti + WSL2
updated: 2026-08-01
status: current
---

# Machine setup

The dev box for the whole home-phase pipeline (see [[pipeline]]). One machine, 8 GB VRAM, everything else is tactics.

## Hardware & OS (verified 2026-08-01)

- **GPU:** NVIDIA GeForce RTX 4060 Ti, **8 GB VRAM** (8188 MiB), driver 610.62
- **OS:** Windows 11 Home 10.0.26200
- **Installed on Windows:** Python 3.12.8, git 2.55.0
- **WSL2: NOT installed yet** (`wsl --status` → "not installed"). This is the first blocker — JAX-CUDA (which MJX needs) does not run natively on Windows, and DimOS targets Linux.

## Day-one install order (not done yet)

1. `wsl --install` → Ubuntu 24.04 (native dual-boot would be even better, WSL2 is the plan)
2. Inside WSL: NVIDIA driver passthrough + CUDA-enabled JAX — `pip install -U "jax[cuda12]"` — verify with `jax.devices()`
3. Install `uv`, Python 3.12, `mujoco`; clone Menagerie + Playground
4. Watch VRAM with `nvidia-smi`; when the 8 GB wall hits, reduce envs/resolution first, cloud second

Update this list as things actually get installed — this note must always reflect the *current* machine state, not the plan.

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

Note: `npx skills add` needed `http.sslBackend=schannel` (via `GIT_CONFIG_*` env vars, not persisted) — the global git config pins `openssl` + Git's CA bundle, which fails TLS verification against GitHub on this box.

## Repo layout

- `notes/` — this Obsidian vault (documentation only)
- `lab-notebook/` — weekly markdown lab notebook, outside the vault
- Pipeline code — none yet (Phase 1 starts Aug 3, 2026)
