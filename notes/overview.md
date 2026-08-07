---
title: Robot Everest 2026 — Overview
updated: 2026-08-07
status: current
---

# Overview

**Mission:** build and validate a pipeline that takes camera footage from Pemba (a Unitree G1 humanoid) on the Everest trek, reconstructs the terrain in 3D with LingBot-Map, converts the reconstruction into an **Isaac Sim** simulation, trains locomotion policies on the actual terrain the robot is facing (Isaac Lab + RSL-RL), and feeds the improved behavior back through DimensionalOS. See [[pipeline]] for the architecture.

> [!important] Simulator pivot, 2026-08-07
> The primary simulator is now **Isaac Sim / Isaac Lab** (`sims/isaac/`) — the Robot Everest
> team's actual stack ("Mapping the full Everest route using Lingbot-Map into IsaacSim, with
> domain randomization over snow, ice friction, and wind gust"). MuJoCo was the onboarding
> stand-in: it carried Phases 1–2 and stays runnable in `sims/mujoco/` as the 50 Hz sim2sim
> validator. Full rationale and rejected alternatives: the 2026-08-07 entry in [[decisions]].

**Why it matters:** nobody has closed the real2sim2real loop on a live expedition, on natural high-altitude terrain, with a *streaming* (not offline) reconstruction model. Existing work (DISCOVERSE, RL-GSBridge, Scalable Real2Sim) is almost entirely indoor manipulation. This is outdoor legged locomotion, and the whole pipeline gets built at home before the team leaves for Kathmandu.

**Key dates:**

- Team departs: ~September 5, 2026
- Mission window: Everest south side, **October 5–20, 2026**
- SF hackathon: **August 29, 2026** (the Phase 4 demo is the entry)
- Commitment: ~12–15 hrs/week; every phase ends in a demo, not a slide deck

> [!info] Current phase
> **Phase 4 — Isaac Sim bring-up** (re-targeted Aug 7). **Isaac is installed and the whole
> smoke ladder is green** (Aug 7, same day): headless SimulationApp in ~8 s, the
> **full-collision G1** (`Dome-G1FullCollision-Flat-v0`, our own task — stock Isaac uses
> the stripped `G1_MINIMAL_CFG`) builds and steps, and a 10-iteration RSL-RL training
> smoke ran at ~690 steps/s on the 8 GB card — see [[setup]] and [[experiments]].
> Next: the real Phase 4a flat-plane policy (cloud GPU), then 4b terrain import —
> LingBot-Map recon if it proves usable, stock/procedural mountain terrain otherwise.
> Phase 3 (LingBot-Map) continues in parallel and is used **only if it works**; it is no
> longer on the critical path.
>
> Phases 1 and 2 both landed early in MuJoCo: Phase 1 on Aug 2, Phase 2 on Aug 4 with a
> self-trained joystick policy walking and turning under command on a full-body-collision
> G1 — see [[locomotion-policy]] for every reward term and [[experiments]] for the runs.
>
> **The whole MuJoCo chain runs end to end** (Aug 6): video → cloud → scale → cleanup →
> heightfield → G1 standing on it, every stage gated. Proven on upstream's `example/loop`
> office walkthrough: 4.07M points → 964k cleaned and metric → a 572×357 hfield at 5 cm,
> with the G1 settling on it at 0.3 mm foot penetration. Commands in [[pipeline]].
>
> **Outdoor trail footage found** (Aug 6): GrandTour **EIG-1** — an alpine rock-and-gravel
> descent with stairs, and a survey-grade CPT7 ground truth, pulled by
> `recon/fetch_grandtour.py` (see [[pipeline]]). A 25 s segment reconstructs at
> `traj_length_over_extent` **1.27** against courthouse's 24.9, with **ATE 1.168 m over 23.0 m**
> — the project's first externally-scored reconstruction, and evidence that the earlier
> underperformance was the footage, not the model. Still **blocked on our *own* footage** for a
> demo that satisfies [[capture-protocol]]; GrandTour is benchmark input, not expedition input.
>
> Two hard limits surfaced along the way, both in [[open-questions]]: this box cannot reach
> upstream's default memory horizon, and an *indoor* scene needs `--surface ground` because
> robust max-z turns furniture into a canyon.

## Milestones

| Due | Phase | Demo | Status |
| --- | --- | --- | --- |
| Aug 9 | 1 — MuJoCo fluency | G1 standing on a numpy-generated heightfield, rendered as video | **done** (Aug 2, MuJoCo) |
| Aug 16 | 2 — First locomotion policy | Self-trained joystick policy walking, every reward term explained | **done** (Aug 4, MuJoCo/MJX) |
| Aug 23 | 3 — LingBot-Map reconstruction | Phone video → dense point cloud of a local trail, camera trajectory overlaid | **toolchain done** (Aug 5); **demonstrated on outdoor trail footage with ground truth** (Aug 6, GrandTour EIG-1); used going forward **only if it works** — no longer on the critical path |
| Aug 30 | 4 — Isaac Sim real2sim terrain | **4a:** full-body-collision G1 trained on a flat plane in Isaac Lab (headless local smoke, cloud for the real run). **4b:** terrain imported — LingBot recon mesh via MeshConverter if usable, else stock/procedural mountain terrain. Hackathon demo Aug 29 | **Isaac installed, smoke ladder green** (Aug 7): full-collision G1 task loads, steps, and trains locally. 4a real policy + 4b outstanding. MuJoCo chain done Aug 6, recorded in `sims/mujoco/` |
| Sept 13 | 5 — Sim2Real training loop | Fine-tuned policy beats baseline on (recon or mountain) terrain in Isaac Lab, with metrics table | not started |
| Sept 27 | 6 — DimOS integration | One command: replayed robot session → Isaac-ready terrain asset (USD) | not started |
| Oct 5–20 | 7 — Live expedition pipeline | Daily recon + terrain analytics from Pemba's Everest footage | not started |
| Nov | Final | Public repo + write-up: "A streaming real2sim2real pipeline from the Everest trek" | not started |

Phase 7 requires two documents written before the window opens: [[capture-protocol]] and [[runbook]] (both currently stubs).

## Vault map

- [[pipeline]] — the real→sim→real architecture and where each tool sits
- [[locomotion-policy]] — what Phase 2 trains: every reward term, the observation space, what domain randomization varies
- [[setup]] — the RTX 4060 Ti + WSL2 machine: what's installed, VRAM tactics
- [[decisions]] — append-only log of choices, reasoning, rejected alternatives
- [[experiments]] — append-only run log (training and reconstruction runs)
- [[capture-protocol]] — filming rules for Pemba on the trek (Phase 7 deliverable)
- [[runbook]] — daily footage→report turnaround (Phase 7 deliverable)
- [[glossary]] — project-specific terms
- [[open-questions]] — known hard problems, not yet resolved

The weekly lab notebook lives in `lab-notebook/` at the repo root, **outside** this vault — one markdown file per week of what was tried and what broke. This vault holds only distilled, current documentation.
