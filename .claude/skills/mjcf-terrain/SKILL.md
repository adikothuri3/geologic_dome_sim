---
name: mjcf-terrain
description: LEGACY MuJoCo track — convert a cleaned point cloud into MuJoCo terrain (hfield) inside an MJCF scene, per the pre-pivot Phase 4 conventions. The primary Isaac terrain path (cloud → OBJ → USD via MeshConverter) is documented in notes/pipeline.md and sims/isaac/terrain/. Use after open3d-cleanup when building terrain for the sims/mujoco track (e.g. sim2sim scenes).
---

# mjcf-terrain

> Legacy since the 2026-08-07 Isaac pivot (`notes/decisions.md`): the primary terrain path is
> now cloud → Open3D mesh → OBJ → USD (`sims/isaac/terrain/`). This skill remains for
> `sims/mujoco/` terrain — the sim2sim validator and the proven Aug 6 chain.

Input: a cleaned, ground-aligned, **scale-calibrated** point cloud (run the `open3d-cleanup` skill first). Output: an MJCF scene with a terrain asset the `unitree_g1` (Menagerie) can stand on. Implementation: `sims/mujoco/terrain/cloud_to_hfield.py`.

## Scale calibration is mandatory — verify before converting

Monocular reconstruction has arbitrary scale. Refuse to build terrain until the cloud has been rescaled against a known length (home: two markers a measured distance apart; expedition: Pemba's dimensions or GPS track length). Sanity bound: terrain features should be human-trail sized — a "step" over ~0.5 m or a cloud spanning km-scale means the calibration is wrong.

## Path 1 — heightfield (default, build first)

1. Grid the XY plane at **5–10 cm cells** (start at 10 cm; go finer only if steps look aliased).
2. Height per cell = **robust max-z** (e.g. 95th percentile of the cell's points, not raw max — raw max bakes in outlier spikes).
3. **Fill holes** (empty cells): interpolate from neighbors; never leave NaN/zero pits.
4. Normalize to [0,1] and emit a MuJoCo `hfield` asset (`.png` or binary) + `<hfield>` with correct `size="x y z_top z_bottom"` so real-world meters are preserved.

## Path 2 — mesh (only for overhangs / large boulders)

Poisson or ball-pivoting surface reconstruction → **decimate to <200k faces** → static collision mesh geom. More expensive and fragile in contact; don't reach for it when the hfield suffices.

## Contact sanity checks (always run before handing off)

- Drop a box onto the terrain (`sims/mujoco/terrain/drop_test.py`); confirm it rests without jitter or tunneling.
- Load the G1 from Menagerie, place above terrain, settle under gravity — no penetration, no explosion (`sims/mujoco/scripts/settle_g1_recon.py`).
- Check `solref`/`solimp` are project-standard (tuned in Phase 4 — record chosen values in `notes/pipeline.md` when they stabilize).
- Timestep survives at the 50 Hz control-loop convention.

Log any conversion driven by a real reconstruction as a row in `notes/experiments.md`.
