# `sims/isaac/terrain/` — Phase 4b terrain builders

Two sources feed Isaac terrain; both end at `TerrainImporterCfg(terrain_type="usd")`:

1. **Real-DEM terrain:** real mountains from survey-grade elevation data — no
   reconstruction involved, so no scale ambiguity and no drift ceiling. Two assets:
   the **Eiger Trail** (built, gated 2026-08-07: Eigergletscher→Alpiglen, 5.4 km,
   713 m descent, swissALTI3D 0.5 m grid — the same alpine rock terrain family as
   the GrandTour EIG-1 benchmark footage) and the **Everest summit patch** (built
   2026-08-08: 2 km around the summit pyramid, NASA HMA 8 m DEM — the expedition
   objective itself).
2. **LingBot recon mesh (per-segment only):** `cloud_clean.ply` (from `recon/`) →
   Open3D mesh → OBJ → USD. Recon is capped at ~25 s / ~25 m per consistent scene
   (see `notes/open-questions.md`), so it can never supply a *long* trail — the DEM
   path exists precisely to cover that.

## The Eiger Trail pipeline (run in the isaac venv)

```powershell
$py = "$env:USERPROFILE\venvs\isaac\Scripts\python.exe"
& $py sims\isaac\terrain\fetch_eiger_trail.py     # OSM route + 11 DEM tiles -> data/
& $py sims\isaac\terrain\build_trail_terrain.py   # straightened strip -> npz/OBJ/origins/meta
& $py sims\isaac\terrain\mesh_to_usd.py           # OBJ -> USD (headless SimulationApp)
& $py sims\isaac\scripts\check_trail.py           # gate: loads, robots stand, obs finite
```

Everything lands under `data/eiger_trail/` (gitignored; the scripts are the
provenance — `meta.json` records every parameter and stat).

**The strip is straightened:** x = arc length along the trail, y = cross-track
(±12 m), z = real DEM elevation at that physical point. Elevation profile,
cross-slope, rock steps are real; map-view curvature is not (a straight strip
makes env origins and spawning trivial). The centerline is smoothed (σ 12 m) to
bound fold-over at switchbacks. Near-vertical cells in the strip are *real* —
the trail passes under the Eiger north face cliff bases.

**Spawn origins are filtered:** the OSM centerline sits a few metres off the
actual trail bench in places, so `build_trail_terrain.py` drops candidate
origins with >1 m relief over a ±1 m patch (1,089 of 2,671 survive). Robots
spawn standing; the cliffs stay in the mesh as terrain to walk into.

**Collision is the exact triangle mesh** (`TriangleMeshPropertiesCfg`, static,
~1.03 M tris — PhysX cooks it in ~5 s on the local box).

> **Face-budget waiver (2026-08-08, pending ratification):** terrain-validator FAILs
> this asset on its 200k-triangle charter budget (5.1×) while passing every
> correctness check (see `reports/terrain-validation-eiger_trail-2026-08-07.md`).
> That budget was written for ~25 m per-segment recon scenes; a full-trail strip is
> length-driven at a modest 8 tris/m², cooks in 4.6 s, and load+settle gates pass on
> the 8 GB box. Treating the budget as per-density for trail strips rather than
> per-asset; decimating to 200k would coarsen cells to ~1.1 m and erase the steps
> the strip exists to represent. If cloud training shows cooking/VRAM pain at 4096
> envs, the fallback is tiling the strip into sub-USDs, not decimation.

The task consuming this: `sims/isaac/tasks/dome_g1/trail_env_cfg.py`
(`Dome-G1FullCollision-EigerTrail-v0`), which also documents the three
usd-terrain deviations from the stock rough task (custom env origins, no
terrain-level curriculum, full-collision G1).

## The Everest summit pipeline (run in the isaac venv)

```powershell
$py = "$env:USERPROFILE\venvs\isaac\Scripts\python.exe"
& $py sims\isaac\terrain\fetch_everest_dem.py       # HMA 8 m tile-677 (370 MB) -> data/hma_dem/
& $py sims\isaac\terrain\build_everest_terrain.py --origin-max-slope 20   # patch -> npz/OBJ/origins/meta
& $py sims\isaac\terrain\render_everest.py          # 4 evidence renders -> reports/
& $py sims\isaac\terrain\mesh_to_usd.py --obj data\everest\terrain\everest_terrain.obj --out-dir data\everest\terrain\usd --name everest_summit.usd
& $py sims\isaac\scripts\check_everest.py           # gate: loads, robots stand, obs finite
```

Everything lands under `data/everest/` (gitignored; scripts are the provenance).
The fetch needs a free NASA Earthdata login in `%USERPROFILE%\_netrc` (one line:
`machine urs.earthdata.nasa.gov login <user> password <pass>`) or an
`EARTHDATA_TOKEN` env var.

**No straightening here:** the patch is axis-aligned to the DEM grid (x = grid
east, y = grid north, centered on the summit; the HMA custom Albers grid is
~1.2° off true north at this longitude). Nothing is resampled — the heightfield
*is* the cropped 8 m grid. 250×250 cells, 1750 m of relief, 124k tris (under
the 200k budget, no waiver needed).

**Spawn origins are slope-filtered, not relief-filtered:** the Eiger ±1 m-relief
criterion is sub-cell at 8 m posting. Cells with slope ≤ 20° survive (the 15°
default found only 18 — the summit pyramid's median slope is 46.5°); 42 of
14,884 candidates pass, mostly benches on the east face and SE ridge. The
fallback ladder (relax to 20° → extend toward the South Col flats) is recorded
in `meta.json`.

**Voids:** 0.11% of the patch (71 cells), filled by 8-neighbor mean dilation in
2 passes — `micro_noise: none` still holds, interpolation only. Builds with
>15% voids abort.

The task consuming this: `sims/isaac/tasks/dome_g1/everest_env_cfg.py`
(`Dome-G1FullCollision-Everest-v0`), reusing the Eiger `TerrainImporter`
subclass; gate: `sims/isaac/scripts/check_everest.py`.
