---
title: Decision log
updated: 2026-08-05
status: current
---

# Decision log

Append-only. Every entry: the choice, the reasoning, and the rejected alternative. Newest at the bottom. Entries are never edited — if a decision is reversed, append a new entry that supersedes it.

---

## 2026-08-01 — Streaming reconstruction with LingBot-Map, not offline COLMAP

**Choice:** LingBot-Map (streaming, monocular RGB) as the reconstruction engine.
**Why:** The trek is one long continuous walk — exactly the long-streaming regime LingBot-Map was built for (stable past 10,000 frames, ~20 FPS, leads Oxford Spires / Tanks & Temples / ETH3D). It needs no LiDAR or depth camera, which matches what Pemba actually carries, and enables same-day turnaround during the expedition window.
**Rejected:** COLMAP-style offline SfM — it's the baseline being beaten; an offline batch run can't support the daily [[runbook]] cadence and struggles at trek-length sequences.

## 2026-08-01 — Stand on MuJoCo Playground's G1 environment, don't build locomotion from scratch

**Choice:** Fine-tune the existing Playground G1 joystick-locomotion environment (MJX) rather than writing a G1 RL environment from zero.
**Why:** Playground's G1 env has demonstrated sim-to-real transfer; the project's novelty is the terrain loop, not the locomotion baseline. Time budget is ~12–15 hrs/week with hard expedition deadlines.
**Rejected:** Custom environment from scratch (months of reward-shaping work that already exists), and Isaac-family GPU simulators (MuJoCo's contact dynamics are more realistic for legged contact, and MJX still gives the GPU parallelism).

## 2026-08-01 — Two terrain paths, heightfield first

**Choice:** Build both conversion paths, in order: `hfield` heightfield first, static collision mesh second (see [[pipeline]] for parameters).
**Why:** Heightfields are fast, robust, and sufficient for most walking terrain; meshes are only *needed* for overhangs and big boulders, and cost more (Poisson reconstruction, decimation, contact tuning).
**Rejected:** Mesh-only (slower, fragile contacts for the common case) and heightfield-only (can't represent overhangs at all).

## 2026-08-01 — WSL2 Ubuntu 24.04 as the dev environment

**Choice:** WSL2 + Ubuntu 24.04 on the Windows 11 box (see [[setup]]).
**Why:** JAX-CUDA, which MJX requires, doesn't run natively on Windows, and DimOS targets Linux.
**Rejected:** Native Windows (blocked by JAX-CUDA); native Ubuntu dual-boot (acknowledged as better, deferred for setup speed — revisit if WSL2 GPU passthrough causes pain).

## 2026-08-01 — Default posture: sim-validated recommendations, not mid-expedition redeployment

**Choice:** Frame Phase 7 output as sim-validated recommendations plus sim2sim evidence; redeploying policies onto Pemba mid-expedition happens only if expedition leads green-light it.
**Why:** The G1 policy stack is partly closed (Unitree's factory controller ≠ our policy) and hardware risk decisions belong to expedition leads. Daily 3D reconstructions + terrain-difficulty analytics are already a first-of-kind contribution even without the closed loop.
**Rejected:** Committing to closed-loop redeployment as the success criterion.

## 2026-08-01 — Documentation split: vault for state, lab notebook for history

**Choice:** This vault (`notes/`) holds only current, distilled documentation; the weekly lab notebook lives in `lab-notebook/` outside the vault; git holds history (stale vault content is deleted, not appended).
**Why:** The vault's one job is giving an agent or human enough context to work on the pipeline in <5 min per note. Journaling inside it would rot that guarantee.
**Rejected:** One combined PKM-style vault with daily notes.

## 2026-08-02 — Menagerie `unitree_g1` as the asset source

**Choice:** Build on `mujoco_menagerie/unitree_g1` (`g1.xml`, 29 position actuators, `nq=36`, one `stand` keyframe).
**Why:** It is the canonical, officially maintained G1. Menagerie also ships `g1_mjx.xml` / `scene_mjx.xml` with `home` and `knees_bent` keyframes, so Phase 2's MJX work can stay in the same repo — choosing Menagerie costs nothing downstream.
**Rejected:** Playground's `g1_mjx_feetonly.xml` as the asset. Same robot with feet-only collision geoms, but it lives inside the Playground tree and would split our asset source across two repos for no Phase 1 gain. This settles the *asset* only — the 2026-08-01 decision to fine-tune Playground's G1 joystick *environment* rather than write one from scratch is unchanged.

## 2026-08-02 — Phase 1 built on native Windows; WSL2 deferred to Phase 2

**Choice:** Run Phase 1 (load → keyframe posing → heightfield → render) natively on Windows. WSL2 remains the Phase 2+ environment; this scopes the 2026-08-01 WSL2 decision rather than reversing it.
**Why:** `wsl --install` is hard-blocked — AMD SVM is disabled in the B550's firmware and cannot be changed from the OS (see [[setup]]). Phase 1 needs no GPU, no JAX and no hypervisor: MuJoCo physics is CPU and Windows renders through WGL with `MUJOCO_GL` unset. Waiting on a BIOS trip would have idled a whole phase for nothing; the demo landed the same day instead.
**Rejected:** Blocking Phase 1 on the BIOS fix. Also rejected: making Windows the permanent environment — JAX-CUDA still does not run natively there and DimOS targets Linux, so MJX training genuinely needs WSL2 before Aug 16.

## 2026-08-02 — `imageio` for video, not `mediapy`

**Choice:** Write frames with `imageio` + `imageio-ffmpeg`.
**Why:** `mediapy` shells out to a system `ffmpeg` binary, which exists after `apt install` on Linux but not on Windows. `imageio-ffmpeg` ships its own binary as a wheel, so one code path covers both OSes — which matters because this project will keep straddling them.
**Rejected:** `mediapy` plus a manual ffmpeg install on Windows (an extra install step and a PATH dependency, for no benefit).

## 2026-08-03 — DimOS gets its own venv, isolated from the Phase 2 environment

**Choice:** `--phase6` installs `dimos[base,unitree]` into `~/venvs/dimos`, not the shared `~/venvs/dome` that MJX training uses.
**Why:** DimOS resolves ~289 packages, including torch, a second full CUDA stack, and its own numpy pin. Installing that alongside jax/MJX lets a Phase 6 dependency silently re-resolve numpy or a CUDA library underneath the locomotion policy. Phase 2 is due Aug 16 and Phase 6 not until Sept 27 — a working training environment is worth far more right now than import convenience later, and the two only need to exchange terrain files, not live objects.
**Rejected:** One shared venv (simplest, but stakes a working Phase 2 on a Phase 6 dependency resolution); deferring DimOS entirely (leaves an unknown install risk sitting in front of a September deadline).

## 2026-08-03 — Allocate VRAM on demand, not by preallocation

**Choice:** Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in `check_phase2.py` and `train_g1.py` before JAX initialises its GPU backend.
**Why:** This 4060 Ti also drives the Windows desktop, which holds ~2 GB. JAX's default 75 % preallocation asks for 6.0 GiB of 8 GiB, fails outright, and falls back down a retry ladder to a fragmented pool. On-demand allocation leaves the display its share and makes `num_envs` the single honest lever for VRAM (see [[setup]]).
**Rejected:** Tuning `XLA_PYTHON_CLIENT_MEM_FRACTION` to a fixed slice — it hardcodes an assumption about what the desktop is using, which changes every time VS Code or a browser opens. Also rejected: leaving the default and living with the retry ladder, which works but starts every run fragmented.

## 2026-08-03 — Pin `jax[cuda12]==0.9.2` until brax catches up

**Choice:** Pin jax to 0.9.2 in `scripts/setup_wsl.sh` rather than tracking latest.
**Why:** brax 0.14.2 — the newest release, and what Playground requires — still calls `jax.device_put_replicated` in `ppo/train.py:756` and three other places. jax deprecated that in 0.8.1 and **removed** it in 0.10.0 as part of the `pmap` → `jit(shard_map)` migration, so an unpinned install resolves to 0.11.0 and every PPO run dies at startup. brax declares only `jax>=0.4.6`, so nothing upstream prevents the bad resolution. 0.9.2 is the last release with the API; MJX, Warp and the G1 env all verified working on it, with `--smoke` reaching reward −3.07.
**Rejected:** Shimming `device_put_replicated` ourselves — the drop-in needs explicit sharding, and hand-rolling device placement under brax's PPO to dodge a version pin is a poor trade. Also rejected: pinning jax without pinning down *why*, which is how a pin outlives its reason — unpin when brax ships a fix, and re-verify with `train_g1.py --smoke`.

## 2026-08-04 — One robot: Menagerie `unitree_g1` plus our own configs, nothing else

**Choice:** Every simulated G1 in this project is Menagerie's `unitree_g1` geometry — meshes, kinematics, inertials — with our own MJCF on top. No second robot source, no vendor URDF import, no hand-modelled links.
**Why:** A real2sim2real pipeline is only as trustworthy as the claim that the sim robot *is* the real robot. One source makes that claim checkable in a line: if it isn't in `mujoco_menagerie/unitree_g1/assets`, it isn't the robot. This is already true in practice and is now a rule — Playground's `g1_mjx_feetonly.xml` references Menagerie's STLs directly (`../../../../../mujoco_menagerie/unitree_g1/assets/*.STL`), so its G1 and our Phase 1 G1 are the same machine; Playground only wraps it with MJX-friendly collision primitives, sensors and actuators. Our additions live in `sim/` and are generated, not hand-drawn: `scripts/make_full_collision_xml.py` sizes every collision box from the Menagerie mesh it wraps, so the collision volume cannot drift from the real geometry.
**Rejected:** Importing Unitree's own URDF (a second, divergent source of truth for the same robot); hand-authoring collision capsules (fast to write, impossible to verify, and drifts silently when upstream updates); and using Menagerie's `type="mesh"` collision geoms verbatim, which are the highest-fidelity option but are convex-mesh collisions that make GPU training impractically slow — see the full-collision entry below.

## 2026-08-04 — Full-body collision, with primitives derived from the meshes

**Choice:** Train on a G1 where every link can touch the world, not just the feet. `scripts/make_full_collision_xml.py` generates `sim/g1_full_collision.xml`; `scripts/check_full_collision.py` gates it; `train_g1.py --full-collision` trains it.
**Why:** Playground collides feet only — fast and adequate on a flat plane, and wrong for this project. On reconstructed Everest terrain a shin, knee, hip or torso meeting rock is a real event, and a policy that has never felt one has a blind spot the simulator never showed it. Generating each primitive from its body's Menagerie mesh bounding box means the collision volume follows the real robot instead of hand-guessed capsules. The gate caught two faults that would otherwise have been invisible: upstream's `left_thigh` and `left_shin` capsules interpenetrate by 16 mm at the nominal pose (harmless while nothing collides, a robot-toppling force once broadphase is on), and the shin capsules are radius 0.08 — a 16 cm-thick shin — which grazes the floor while merely standing and would have poisoned `feet_slip`, `feet_air_time` and the contact sensors with phantom ground contact.
**Consequence worth knowing:** enabling collision also revives `left_foot_right_shin_found` / `right_foot_left_shin_found`. Those sensors are read by `_get_termination()` to catch leg crossing, but their contact `<pair>`s are commented out upstream, so that half of the termination test could never fire in the feet-only baseline.
**Rejected:** Menagerie's mesh collision geoms (highest fidelity; convex-mesh collision is far too slow for 200M GPU steps — revisit for a low-env-count Phase 5 validation run). Also rejected: hand-placing capsules, where one wrong offset gives a robot resting on invisible geometry and the render looks fine.

## 2026-08-05 — Reconstruct into our own export, and only trust one window

**Choice:** `recon/reconstruct.py` wraps LingBot-Map's model API rather than calling upstream's `demo.py`, writes `cloud.ply` + `trajectory.npz` + `run.json` itself, and — until cross-window alignment is fixed — clips are sized to fit a **single window** (~124 frames at `window_size 24`, `keyframe_interval 6`).
**Why:** `demo.py` exports nothing; it runs inference and opens a viser viewer, so closing the tab discards the reconstruction. Phase 4 needs a cloud and a camera trajectory on disk, so an export layer has to exist somewhere and owning it lets us record the run config, VRAM peak and alignment diagnostics in the same place. The single-window rule is empirical, not cautious: stitched runs produced camera paths **34–38×** the scene extent with a hard pose jump at a window boundary, and tripling the overlap made it worse — the failure is scale estimation on low-texture overlap, not overlap size. The same clip in one window lands at 2.8×. `run.json` records `traj_length_over_extent` so the check is automatic rather than remembered.
**Why not just use their viewer:** we still do — `recon/view_viser.py` serves the *exported* files, so any past run reopens in seconds without a GPU, instead of re-running inference to look at a result.
**Rejected:** `demo_render/batch_demo.py --save_predictions`, which does export, but pulls in kaolin plus a CUDA extension build for a rendering path we don't need. Also rejected: accepting the stitched full-clip cloud as the deliverable — it is visibly wrong, and shipping a broken artefact into Phase 4 would surface as mysterious terrain geometry three weeks later.
**Corrected 2026-08-05:** this entry originally also rejected streaming mode "because it holds every frame on the GPU." That was our bug, not the model's behaviour — streaming slices per frame exactly as windowed does, and runs the full clip at flat 5.63 GB. The single-window rule survives the correction, but for a different reason than first written: the limit is ~25 s of footage per consistent scene in *either* mode, set by drift once the scene leaves the KV cache's memory horizon, not by window stitching alone. See [[setup]].

## 2026-08-05 — LingBot-Map gets its own venv

**Choice:** `scripts/setup_lingbot.sh` installs into `~/venvs/lingbot`, not the shared `~/venvs/dome`.
**Why:** Upstream pins torch 2.8.0/cu128; `~/venvs/dome` holds the `jax==0.9.2` pin that brax needs. One venv means two CUDA stacks negotiating over the same numpy, and a Phase 3 install could silently break the Phase 2 training environment. Same reasoning as the DimOS venv above, and it cost nothing — the two exchange PLY files, not live objects.
**Consequence:** the viewer needs viser *and* Open3D, which live in different venvs; viser was added to `~/venvs/dome` (the one with Open3D) rather than Open3D to the recon venv, to keep the working inference environment untouched.
