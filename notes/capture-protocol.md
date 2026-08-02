---
title: Capture protocol for Pemba
updated: 2026-08-01
status: current
---

# Capture protocol for Pemba

> [!todo] Stub — Phase 7 deliverable
> Target: a single page of filming rules the expedition team can follow without us. Due before the team departs (~Sept 5, 2026). The items below are the roadmap's starting constraints; each one gets validated or revised by the failure modes logged in Phases 3–4 (see [[experiments]] and [[open-questions]]).

Draft rules to validate:

- **Resolution / fps:** RGB video at 20–30 fps (LingBot-Map's native operating regime is 518×378 — exact capture settings TBD after Phase 3 testing).
- **Exposure:** locked. Exposure swings are a known LingBot-Map failure mode; snow makes it worse.
- **Camera:** forward-facing, fixed mount.
- **Speed:** walking-speed limits to control motion blur — thresholds TBD from Phase 3 failure logging.
- **Turns:** overlap views on turns; avoid fast pans.
- **Chunking:** footage in ≤10-minute chunks (drift control + VRAM-friendly recon, see [[setup]]).
- **Scale:** GPS/odometry logs recorded alongside every chunk — this is the scale-calibration source (see [[pipeline]]).
- **Delivery:** daily upload format TBD (constraint: whatever comes down over Starlink).
