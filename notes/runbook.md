---
title: Daily turnaround runbook
updated: 2026-08-01
status: current
---

# Daily turnaround runbook

> [!todo] Stub — Phase 7 deliverable
> Target: a single page that turns a day's footage into a morning report, runnable by one person during the Oct 5–20 window. The skeleton below is from the roadmap; each step gets replaced with exact commands as Phases 3–6 land (the Phase 6 demo — one command from replayed session to terrain file — becomes the core of steps 2–3).

Daily loop (skeleton):

1. **Footage lands** (over Starlink, per [[capture-protocol]]).
2. **Same-day reconstruction** — LingBot-Map on the day's chunks; confidence-filter; log a row in [[experiments]].
3. **Terrain build** — Open3D cleanup → scale calibration from GPS/odometry → heightfield (+ mesh where needed), per [[pipeline]].
4. **Overnight fine-tune** on cloud GPU (the 8 GB local card is for debugging only during the window — see [[setup]]).
5. **Morning report:** reconstruction quality, slope/roughness histograms, policy metrics (recon vs. flat, before vs. after), flagged hazards.

Even if closed-loop redeployment stays off the table (see [[decisions]]), daily 3D reconstructions of the route + terrain-difficulty analytics are the deliverable.
