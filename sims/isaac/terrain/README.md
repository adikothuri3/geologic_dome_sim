# `sims/isaac/terrain/` — Phase 4b terrain builders

Two sources feed Isaac terrain; both end at `TerrainImporterCfg(terrain_type="usd")`:

1. **Real-DEM trail (built, gated 2026-08-07):** a long real mountain trail from
   survey-grade elevation data — no reconstruction involved, so no scale ambiguity
   and no drift ceiling. First asset: the **Eiger Trail** (Eigergletscher→Alpiglen,
   5.4 km, 713 m descent, swissALTI3D 0.5 m grid) — the same alpine rock terrain
   family as the GrandTour EIG-1 benchmark footage.
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
