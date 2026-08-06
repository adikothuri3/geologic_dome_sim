---
title: Capture protocol for Pemba
updated: 2026-08-06
status: current
---

# Capture protocol for Pemba

> [!todo] Stub — Phase 7 deliverable
> Target: a single page of filming rules the expedition team can follow without us. Due before the team departs (~Sept 5, 2026). The items below are the roadmap's starting constraints; each one gets validated or revised by the failure modes logged in Phases 3–4 (see [[experiments]] and [[open-questions]]).

> [!danger] Rule zero: translate, don't rotate
> **Parallax is the whole ballgame.** LingBot-Map recovers geometry by triangulating a point across camera positions. Rotation moves every pixel but reveals *no* depth, so footage shot by pivoting in place gives the model nothing to triangulate — it invents depth, and the invented surfaces stack into an unusable smear. No setting, checkpoint or cleanup pass can recover parallax that was never filmed.
>
> Measured Aug 5 — the same box, same code, two clips:
>
> | | upstream `example/loop` (clean result) | our indoor clip (unusable) |
> | --- | --- | --- |
> | Total rotation | 429° | 578° |
> | Translation path | 36.02 | 5.36 |
> | Camera-centre spread | 14.15 | 1.38 |
> | **Rotation per unit translation** | **11.9 °/unit** | **107.9 °/unit** |
>
> **Target ≤ ~15 °/unit; treat >40 as a reshoot.** The failing clip rotated through more than 1.5 full turns while translating 6.7× less than the working one. Note both clips scored a *healthy* `traj_length_over_extent` (2.47 vs 3.36) — that metric does not detect this, so it must be checked separately ([[experiments]]).
>
> In practice: keep walking, point the camera along the direction of travel, and turn only while moving. Never plant your feet and sweep the camera across a scene.

Draft rules to validate:

- **Resolution / fps:** RGB video at 20–30 fps (LingBot-Map's native operating regime is 518×378 — exact capture settings TBD after Phase 3 testing).
- **Orientation: film landscape.** Confirmed Aug 5, not TBD. LingBot-Map fits width to 518 and only crops height, so a landscape frame becomes 518×294 while the same portrait frame pads to 518×518 — 43% of the compute spent on blank pixels, for identical field of view. Measured cost of getting this wrong: +1.8 GB VRAM and −28% throughput ([[experiments]]).
- **Exposure:** locked. Exposure swings are a known LingBot-Map failure mode; snow makes it worse.
- **HDR: prefer it off.** Phone/robot cameras recording HLG or Dolby Vision need a real tonemap before inference, not a pixel-format cast, or frames arrive washed out and low-contrast. `recon/extract_frames.py` handles it, but SDR capture removes a whole failure mode.
- **Camera:** forward-facing, fixed mount — which is exactly what rule zero requires. A head/chest mount on a walking robot is close to ideal; a handheld operator standing still and looking around is the worst case.
- **Framing:** aim at mid-distance, roughly 1.5–4 m of usable depth. Large blank surfaces filling the frame at arm's length give neither texture nor parallax; near-field floor plus far background in the same frame is what the depth head wants.
- **Speed:** walking-speed limits to control motion blur — thresholds TBD from Phase 3 failure logging.
- **Turns:** overlap views on turns; **avoid fast pans, and never pan from a standstill** (see rule zero). Arc into direction changes while still moving.
- **Don't close the loop.** There is no loop closure or pose-graph optimisation in LingBot-Map (upstream issues #60/#78, both open), so returning to the starting viewpoint puts the same surface back at a drifted position as a second offset layer. End each clip somewhere new.
- **Chunking:** footage in ≤10-minute chunks (drift control + VRAM-friendly recon, see [[setup]]). **Note:** a reconstruction is currently only globally consistent within one window (~124 frames on our box) — chunk length is bounded by stitching, not just by drift, until [[open-questions]] resolves it.
- **Scale:** GPS/odometry logs recorded alongside every chunk — this is the scale-calibration source (see [[pipeline]]). Film the calibration markers *into* the start of each clip rather than measuring after the fact.
- **Delivery:** daily upload format TBD (constraint: whatever comes down over Starlink).
