---
title: Decision log
updated: 2026-08-01
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
