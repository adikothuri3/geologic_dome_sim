---
title: Machine setup — RTX 4060 Ti (Windows-native Isaac + WSL2 legacy)
updated: 2026-08-08
status: current
---

# Machine setup

The dev box for the whole home-phase pipeline (see [[pipeline]]). One machine, 8 GB VRAM, everything else is tactics.

## Isaac Sim / Isaac Lab (primary sim — pivoted 2026-08-07)

**Status: installed and smoke-tested 2026-08-07.** All gates green on this box — see the [[experiments]] row `2026-08-07-isaac-g1fc-flat-smoke` and `runs/isaac/gates.log`. Gate **c** (added 2026-08-08) checks that the domain randomization in `Dome-G1FullCollision-Flat-DR-v0` reaches PhysX; see [[locomotion-policy]] for why that needs asserting.

- **Installed:** isaacsim **5.1.0.0**, isaaclab **0.54.2**, isaaclab-tasks 0.11.12, isaaclab-assets 0.2.4, rsl-rl-lib 3.1.2, torch **2.7.0+cu128** (CUDA confirmed), **Python 3.11.9**. **Native Windows 11 — no WSL.** Not Isaac Lab 3.0 beta (Ubuntu-only).
- **Venv:** `%USERPROFILE%\venvs\isaac` · **Isaac Lab clone:** `C:\Users\Aditya\src\IsaacLab` (tag v2.3.2)
- Installer: `sims/isaac/setup_isaac.ps1` (idempotent). It installs the Isaac Lab source packages with **direct pip** — `isaaclab.bat --install` only detects conda or its bundled kit-python, not a plain venv — and pins **`tensordict==0.8.3`** (the latest wheel is built against a newer torch ABI and access-violates on import with torch 2.7.0).
- **Measured on this box** (below Isaac's 16 GB VRAM / 32 GB RAM minimum): headless SimulationApp up in **~8 s**; full-collision G1 env (8 envs) builds in 26 s; 10 RSL-RL iterations at 64 envs run at **~690 steps/s**. So local headless smoke work is genuinely fine; **real training still goes to a rented cloud GPU** (≥24 GB) — never open the GUI viewport here.
- Training logs: `runs/isaac/<YYYY-MM-DD-slug>/`, one [[experiments]] row per run (`sims/isaac/scripts/train_g1_flat.py` writes both).

> [!warning] numpy is pinned to 1.26.0 — do not let anything upgrade it
> `numba 0.59.1`, which `isaacsim-core` depends on, requires `numpy <1.27,>=1.22`.
> On 2026-08-08 the venv was found holding **numpy 2.4.6 files under a numpy-1.26.0
> dist-info** — a 2.x wheel unpacked over a 1.x install without a clean uninstall, so
> pip believed the pin was satisfied. Plain `import numpy` worked and reported 2.4.6;
> `import numpy.ma.mrecords` did not, which meant `trimesh` failed, which meant
> `isaaclab_assets` and `isaaclab_tasks` failed to start **as Kit extensions** — several
> hundred lines of `AttributeError: module 'numpy' has no attribute '_core'` before
> `AppLauncher` even returned. Repair, and the check worth running after any pip install
> into this venv:
>
> ```powershell
> $py = "$env:USERPROFILE\venvs\isaac\Scripts\python.exe"
> & $py -m pip uninstall -y numpy; & $py -m pip install "numpy==1.26.0"
> & $py -c "import numpy, numpy.ma.mrecords, trimesh; print(numpy.__version__)"
> ```
>
> The last line is the real gate: `import numpy` alone passes on a broken tree.

> [!warning] Three Windows/box quirks, all handled in the installer
> - **Norton MITMs TLS.** Python/OpenSSL can't verify anything until you point it at
>   `%USERPROFILE%\venvs\ca-bundle-norton.pem` (pip's certifi + the exported Norton root;
>   set as `PIP_CERT`/`SSL_CERT_FILE`). git needs `-c http.sslBackend=schannel`. This is the
>   same root cause as the WSL-era git TLS workaround below.
> - **The Omniverse EULA blocks headless first launch.** `OMNI_KIT_ACCEPT_EULA=YES` is
>   persisted as a user env var; new shells have it, pre-existing ones don't.
> - **Kit hijacks python stdout** — prints vanish. Gate verdicts go to stderr +
>   `runs/isaac/gates.log`; and Kit teardown (`env.close()`) can die with a native access
>   violation, so `train_g1_flat.py` writes its experiments row *before* closing anything.

## Hardware & OS (verified 2026-08-02)

- **GPU:** NVIDIA GeForce RTX 4060 Ti, **8 GB VRAM** (8188 MiB), driver 610.62
- **CPU:** AMD Ryzen 7 5700 (8 cores) · **Board:** Gigabyte B550 UD AC, AMI BIOS FEc
- **OS:** Windows 11 Home 10.0.26200, UEFI
- **Installed on Windows:** Python 3.12.8, git 2.55.0

## WSL2 install state (2026-08-03) — complete (legacy MuJoCo track + recon)

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
| Open3D (Phase 4) | ✅ 0.19.0 (+ viser 1.0.30, for `recon/view_viser.py`) |
| LingBot-Map (Phase 3) | ✅ own venv `~/venvs/lingbot`, torch 2.8.0+cu128 |
| Phase 1 parity on Linux | ✅ all three scripts pass, MP4 renders |

Reproduce from scratch with `sims/mujoco/scripts/setup_wsl_stage2.ps1` (Windows: distro + user)
then `sims/mujoco/scripts/setup_wsl.sh --all` (Linux toolchain). Stage 2 installs Ubuntu with `--no-launch`,
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
> re-verify with `python sims/mujoco/scripts/train_g1.py --smoke`.

Two consequences worth knowing before debugging anything in WSL:

- **`/tmp` does not survive.** `systemd=true` mounts it as tmpfs, so an idle-terminated
  distro takes the logs with it. Write install logs to a `/mnt/c/...` path.
- **Playground clones its own Menagerie** (pinned commit `1b86ece`, ~500 MB) into
  `~/src/playground/mujoco_playground/external_deps/` on the *first* `registry.load()`. That
  is a one-time cost, already paid; a first training run on a fresh box will pay it again and
  can look like a hang.

## LingBot-Map (Phase 3)

Installed by `recon/setup_lingbot.sh` into **its own venv, `~/venvs/lingbot`** — upstream
pins torch 2.8.0/cu128 and `~/venvs/dome` holds the jax 0.9.2 pin brax needs, so one venv
would force a CUDA-stack fight. Same reasoning as the dimos venv, see [[decisions]].

- Source clone: `~/src/lingbot-map` (editable install, `[vis]` extras → viser 1.0.30)
- Checkpoints: **both downloaded**, `~/ckpt/lingbot-map/` — `lingbot-map.pt` and
  `lingbot-map-long.pt`, 4,632,303,465 bytes each, **1.16 B params, flat fp32 state dict** (no
  optimizer state). They load into the *same* architecture with no missing keys, so switching is
  just `--model_path`. Default to `-long` (see the `lingbot-recon` skill).
- **Fetch checkpoints on the Windows side.** `huggingface_hub` inside WSL stalled at 196 KB and
  never recovered — the documented ~1-in-5 HTTPS stall. `curl.exe` on Windows pulled the same
  4.63 GB at ~12 MB/s in 6 minutes; then `cp` it into `~/ckpt/` over drvfs.
- **Sky segmentation:** `~/ckpt/skyseg.onnx` (176 MB, from `JianyuanWang/skyseg` on HF, same
  Windows-side `curl.exe` route). Needed by `reconstruct.py --mask_sky` for outdoor scenes;
  `onnxruntime` 1.23.2 is already in the lingbot venv. It runs on **CPU on purpose** — the GPU
  is at its ceiling from the aggregator, and 320×320 segmentation over a few hundred frames
  costs seconds. Upstream only applies sky masking in its *viewer*, so our exported `.ply`
  needed its own path.
- **FlashInfer is deliberately not installed** — it is a long build and we pass `--use_sdpa`
  anyway. Every run prints `flashinfer not available`; that line is expected, not a warning.

> [!warning] The checkpoint does not fit the naive load path
> Upstream's `load_model()` does `torch.load(map_location=device)`. That fails twice on this
> box: 4.63 GB of checkpoint on an 8 GB card *plus* a second allocation when `model.to(device)`
> runs, and 4.63 GB of checkpoint *plus* 4.63 GB of freshly built model in 7.7 GB of WSL RAM.
> `recon/reconstruct.py` loads with `mmap=True` and casts the aggregator to bf16 **before** the
> GPU transfer, which lands weights at **2.81 GB** resident.
>
> Confirmed empirically 2026-08-06: running upstream `demo.py` unmodified dies with
> `RuntimeError: CUDA driver error: device not ready` in `load_model`. **Upstream's demo simply
> cannot run on this box** — always go through `recon/reconstruct.py`, including when reproducing
> upstream's own example scenes.

> [!warning] Overshooting VRAM can take the whole WSL VM down
> Exceeding VRAM under WSL does not reliably raise a clean torch OOM — it surfaced as
> `CUDA driver error: device not ready`, and once killed the distro outright with
> `Wsl/Service/E_UNEXPECTED` (recover with `wsl --shutdown`). `reconstruct.py` therefore sets
> `torch.cuda.set_per_process_memory_fraction` (`--vram_fraction`, default 0.85) so the failure
> is a catchable Python exception. Keep it.

Working limits on this card at 518×294, measured (see [[experiments]]):

| Knob | Ceiling | Why |
| --- | --- | --- |
| `--window_size` | **24 keyframes** (6.21 GB peak); 32 OOMs | dominant VRAM driver in windowed mode |
| `--kv_cache_sliding_window` | **≤24** in streaming mode (6.55 GB peak); 32 OOMs | the cache *is* the model's memory horizon — see below |
| sequence length | ~25 s of footage per consistent scene, either mode | drift, not memory — see below |
| frame count | ~660 at 518×294 | predictions accumulate on CPU in fp32; `reconstruct.py` preflights this and aborts in seconds rather than being OOM-killed 20 min in |

### Streaming mode works; its limit is drift, not memory

Streaming runs a **bounded** KV cache of recent keyframes and slices frames to the GPU one at a
time, so peak VRAM is flat regardless of sequence length — the whole 660-frame clip completed at
**5.63 GB**. This is the property Phases 6–7 depend on: a live camera feed never has more than a
frame in flight, which is how upstream claims stability past 10,000 frames.

> [!warning] Leave `images` on the CPU in **both** modes
> An early version of `recon/reconstruct.py` moved the whole tensor to the GPU for streaming, on
> the false assumption that only windowed mode slices per iteration. Both do
> (`gct_stream.py`: *"we slice-then-move per iteration so peak GPU memory is O(scale) or O(1)
> frames rather than O(S)"*). The bug made streaming look unusable past ~100 frames when it in
> fact handles the entire clip.

What actually bounds a reconstruction is **drift**, and the KV cache size sets the memory
horizon — anything older than the cache is forgotten. On this card the cache maxes at ~24
keyframes ≈ 14 s of memory, and quality decays with clip length (`traj_length_over_extent`,
≲3 healthy):

| Clip length | streaming | windowed |
| --- | --- | --- |
| 25 s | **2.76** | **2.76** (identical output) |
| 49 s | 11.8 | 9.7 |
| 132 s | 34–46 | 31–38 |

So both modes hit the same wall for the same reason, and neither is a workaround for the other.
A larger GPU buys a larger cache and therefore a longer horizon — this is a concrete reason for
the cloud-GPU line item in [[open-questions]], not just a training-throughput one.

### The rented-GPU path: `colab/lingbot_map_colab.ipynb`

Built 2026-08-06 to test that claim rather than assert it. It shells out to the same
`recon/*.py` scripts, unmodified, so the only variable between it and a local run is the card —
which is the only way its numbers mean anything. What it does: both upstream demo scenes at
upstream's *exact* config, a `kv_cache_sliding_window` ladder from 16 to 128, heavy Open3D
cleanup, and a printed compute-bound / not-compute verdict. `colab/README.md` has the how.

Three things it needs that a local run does not:

- **`LINGBOT_SRC`**, an env var `reconstruct.py` already reads, pointed at the Colab clone —
  it imports upstream's `demo.py` before touching torch.
- **`recon/` uploaded as a zip.** The GitHub remote is private, so the notebook tries Drive,
  then a `GD_TOKEN` Colab Secret, then a `files.upload()` of `recon.zip` (~80 KB).
- **`keyframe_interval` pinned to 1.** Upstream's auto is `(n+319)//320`; ours is
  `ceil(n/240)`. They agree on loop's 237 frames and *disagree* on courthouse's 286, so
  leaving it on auto would have silently changed the thing being measured.

Watch for one thing on a T4: `reconstruct.py` picks bf16 only on sm_80+ and drops to fp16
below it, so a Turing card changes the numeric path as well as the VRAM. Use A100 or L4.

## The robot: Menagerie `unitree_g1`, and only that

> [!important] One source of robot geometry
> Every simulated G1 in this project is Menagerie's `unitree_g1` — meshes, kinematics,
> inertials — with our own MJCF layered on top. **If it isn't in
> `mujoco_menagerie/unitree_g1/assets`, it isn't the robot.** Playground's
> `g1_mjx_feetonly.xml` references those STLs directly, so its G1 and the Phase 1 G1 are the
> same machine; Playground only adds MJX-friendly collision primitives, sensors and actuators.
> See [[decisions]].

Our own layers, all in `sims/mujoco/xmls/`, all generated rather than hand-drawn:

| File | Made by | What it is |
| --- | --- | --- |
| `scene_g1_hfield.xml` | hand, Phase 1 | G1 on a numpy heightfield, no floor plane |
| `g1_full_collision.xml` | `sims/mujoco/scripts/make_full_collision_xml.py` | every link collidable; each box sized from that body's Menagerie mesh |
| `scene_g1_full_collision.xml` | hand | the flat-terrain scene, including the above instead of the feet-only body |

Gate the generated model with `python sims/mujoco/scripts/check_full_collision.py` before training on it:
it asserts the environment contract survives, that nothing self-collides at the nominal pose,
that only the feet touch the ground when settled, and that a toppled robot is actually caught
by the new geometry.

## The Windows side (still live, still useful)

Phase 1 was built here while WSL2 was blocked on firmware, and it still works: MuJoCo physics
is CPU and Windows renders through WGL with `MUJOCO_GL` unset (see [[decisions]]). Keep it —
it needs no hypervisor, it is the fallback if WSL breaks, and its Menagerie clone is what
seeds the Linux one. Phase 1 now passes identically on both.

- The old repo-root `.venv` (Phase-1-only: mujoco + numpy + imageio) was **deleted 2026-08-07**
  — everything MuJoCo now runs in WSL's `~/venvs/dome`; recreate a Windows venv only if needed
- **Menagerie** sparse-cloned to `C:\Users\Aditya\src\menagerie` (`unitree_g1` only), exposed
  to the scene as the `sims/mujoco/xmls/menagerie` junction
- Video is written via `imageio` + `imageio-ffmpeg`, **not** `mediapy` — mediapy shells out to
  a system ffmpeg that Windows does not have; imageio bundles its own binary and is identical
  on both OSes

## The Linux toolchain (`sims/mujoco/scripts/setup_wsl.sh`)

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

## Phase 2 VRAM budget (legacy MuJoCo track)

**The usable budget is ~6 GB, not 8.** This GPU also drives the Windows desktop, which holds
roughly 2 GB. JAX preallocates 75 % of total VRAM by default — 0.75 × 8188 MiB ≈ 6.0 GiB —
and that allocation *fails*, after which XLA retries down a ladder (5.4 → 4.9 → 4.4 GiB) and
proceeds with a fragmented pool. So `check_phase2.py` and `train_g1.py` both set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` before JAX initialises its backend, and allocate on
demand instead. JAX then reports `vram 6.0 GiB`.

Playground's G1 defaults to **8192 parallel envs**, which will OOM.
`sims/mujoco/scripts/train_g1.py` caps it at 2048 per the `training-run` skill and preserves brax's
`batch_size × num_minibatches == num_envs` relation (upstream `256 × 32 = 8192`) by holding
`num_minibatches = 32` and deriving `batch_size` — so gradient maths is unchanged and only
parallelism shrinks. Watch `nvidia-smi` in the first minutes; at the wall drop to 1024 envs
first, cloud second. Closing VS Code buys back most of a gigabyte.

## What runs on 8 GB, and how

| Workload | Runs locally? | How |
| --- | --- | --- |
| **Isaac Sim / Isaac Lab (primary)** | Smoke tests only | Below official minimum (16 GB VRAM / 32 GB RAM). Headless, `num_envs` 64–256, load-and-step checks. Real training → cloud GPU. |
| MuJoCo (CPU physics + viewer, legacy) | Yes, easily | Core physics is CPU-based; GPU only helps rendering. Phases 1 and 4 (legacy chain) fully local. |
| LingBot-Map inference | Yes | Runs at 518×378. Control KV-cache growth with `keyframe_interval`, use windowed mode, filter by confidence. 8 GB handles trail-length clips; chunk very long videos. |
| MJX / Playground RL training (legacy) | Yes, with settings | Cut parallel envs (8192 → 1024–2048) and batch size; G1 joystick policies still train locally, just slower. |
| Open3D / point-cloud work | Yes, easily | Mostly CPU + light GPU. |
| DimOS + replay datasets | Yes | Replay needs no hardware; agents/modules are CPU-light. |

> [!warning] The VRAM eaters
> LingBot-Map's KV cache, Isaac Sim's renderer + env count, and (legacy) the MJX parallel-env count are what hit the 8 GB ceiling. Keyframe/windowed mode for recon; headless + few envs for Isaac smoke tests; rent a cloud GPU for real Isaac training, Phase 5 sweeps, and during the expedition window. See [[open-questions]].

## Claude Code tooling (installed 2026-08-01)

**MCP servers** — user scope (`claude mcp add -s user -t http …`), both verified connected via `claude mcp list`. Neither needs an API key or filesystem access.

- **deepwiki** (`https://mcp.deepwiki.com/mcp`) — repo-level Q&A over the codebases this pipeline sits on: `isaac-sim/IsaacLab`, `unitreerobotics/unitree_rl_lab`, `leggedrobotics/rsl_rl`, `dimensionalOS/dimos`, LingBot-Map, `isl-org/Open3D`, plus the legacy `google-deepmind/mujoco` / `mujoco_playground` / `mujoco_menagerie`.
- **context7** (`https://mcp.context7.com/mcp`) — version-current API docs; query before writing code against `isaacsim`, `isaaclab`, `open3d`, `onnx`, or the legacy `mujoco`/`mjx`/`jax` APIs so calls aren't stale.

**Custom skills** in `.claude/skills/` (project conventions, one page each — update them as conventions change):

- `mjcf-terrain` — **legacy MuJoCo track**: point cloud → hfield/mesh → MJCF, Phase 4 conventions (5–10 cm cells, robust max-z, hole filling, scale calibration, contact checks)
- `open3d-cleanup` — outlier removal, voxel downsample, ground-plane alignment defaults
- `training-run` — run discipline for **both sims**: commit-first, config capture, mandatory [[experiments]] row; Isaac local/cloud split + legacy MJX env-count limits
- `lingbot-recon` — reconstruction defaults: `keyframe_interval`, windowed mode, confidence filtering, ≤10-min chunking

`obsidian-markdown` (from kepano/obsidian-skills) keeps vault notes valid Obsidian-flavored markdown. The rest of that bundle (`obsidian-cli`, `obsidian-bases`, `json-canvas`, `defuddle`) was removed as unused — reinstall from the same repo if ever needed.

**Third-party skills:** the vendored `mujoco` skill (coolbeevip/mujoco-skills) was **removed
2026-08-07** — robot-arm oriented, referenced by nothing here, and its `.claude/skills/mujoco`
junction was a standing Windows/WSL git hazard. Previously rejected from the registry:
`letta-ai@tune-mjcf`, `plurigrid@urdf2mjcf`, `onnx-converter`.

**Subagents** in `.claude/agents/` (added 2026-08-01, one page each — delegation targets for the main agent):

- `docs-researcher` — read-only external-library lookups (isaac sim/lab, rsl-rl, dimos, lingbot-map, open3d, legacy mujoco/mjx) via deepwiki/context7/web; returns a sourced synthesis instead of doc dumps
- `terrain-validator` — QA gate for real2sim terrain assets in either sim (scale, cell size, holes, face budget, scene load, G1 settle test, slope/roughness stats); writes reports under `reports/` only
- `run-auditor` — after every training/reconstruction run, parses logs and appends the mandatory [[experiments]] row; flags reward hacking, train/eval divergence, VRAM near the 8 GB ceiling
- `vault-keeper` — keeps this vault synced with code changes at end of session; edits `notes/` only

Note: `npx skills add` needed `http.sslBackend=schannel` (via `GIT_CONFIG_*` env vars, not persisted) — the global git config pins `openssl` + Git's CA bundle, which fails TLS verification against GitHub on this box.

## Repo layout (restructured 2026-08-07)

- `notes/` — this Obsidian vault (documentation only)
- `lab-notebook/` — weekly markdown lab notebook, outside the vault
- `recon/` — **sim-agnostic** Phase 3 toolchain: `extract_frames.py`, `fetch_grandtour.py`,
  `measure_flow.py`, `reconstruct.py` (LingBot-Map → `cloud.ply` + `trajectory.npz` + `run.json`),
  `eval_ate.py`, `calibrate_scale.py`, `clean_cloud.py`, `inspect_cloud.py`, `view_viser.py`,
  `setup_lingbot.sh`. Its output seam is a clean, metric, ground-aligned cloud
  (`cloud_clean.ply` + `scale.json`); everything downstream is per-sim.
- `sims/isaac/` — **primary sim track** (Isaac Sim / Isaac Lab): setup script, task and
  terrain-import scaffolding. See `sims/isaac/README.md`.
- `sims/mujoco/` — **legacy MuJoCo/MJX track**, kept runnable: `scripts/` (training, gates,
  installers), `terrain/` (`make_hfield.py`, `drop_test.py`, `cloud_to_hfield.py`, `assets/`),
  `xmls/` (MJCF scenes + `menagerie` junction). See `sims/mujoco/README.md`.
- `colab/` — rented-GPU LingBot-Map notebook (`recon/` scripts baked in by `embed_recon.py`)
- `reports/` — rendered evidence (contact sheets, gate renders, demo videos)
- `runs/` — gitignored run outputs: `runs/recon/` (reconstructions), `runs/mujoco/`
  (legacy training runs), `runs/isaac/` (Isaac Lab logs, once training starts). Pre-pivot
  runs sit loose at `runs/` top level; history is not moved.
- `sims/mujoco/xmls/menagerie`, `runs/`, `data/`, `*.mp4` are gitignored
