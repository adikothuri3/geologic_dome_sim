# Terrain validation — eiger_trail (Isaac track) — 2026-08-07

**Asset:** `data/eiger_trail/terrain/` (heightfield.npz, origins.npz, trail_terrain.obj, usd/eiger_trail.usd, meta.json)
**Source:** swissALTI3D 0.5 m GeoTIFFs in `data/swissalti3d/` (11 tiles, all pixel scale exactly 0.5×0.5 m), OSM relation 19465820 centerline (`data/eiger_trail/trail_lv95.json`)
**Builder:** `sims/isaac/terrain/build_trail_terrain.py` (straightened corridor, 12 m centerline smoothing)
**Validator commands:** see "Repro" at the bottom. All checks read-only; no assets modified.

## Verdict: **FAIL** — one check fails (mesh budget: 1,029,216 collision tris > 200,000 charter limit). All other checks PASS.

The failing check is a budget-policy breach, not a correctness defect: PhysX cooked the full
~1M-tri triangleMesh in 4.6 s and 8 envs run the settle gate cleanly on the 8 GB card
(`runs/isaac/gates.log`, `[gate trail]` 2026-08-07 23:28–23:29). Declaring the asset
training-ready requires either an explicit budget waiver for full-trail strips, or
decimation / per-section tiling to get under 200k.

## Checks

### 1. Scale calibration — PASS
Survey-grade metric source (swisstopo), so scale reduces to "is the geometry read out correctly":
- All 11 GeoTIFF tiles report pixel scale exactly (0.5, 0.5) m in LV95.
- Independent per-tile bilinear reads at 7 centerline points (s = 0, 1072, 2144, 2680.5, 3216.5, 4288.5, 5360.5 m) vs strip z + z_offset: **max error 0.0 mm**.
- Known elevations: strip start (Eigergletscher end) 2328.1 m abs vs station ~2320 m; strip end (Alpiglen end) 1614.6 m vs station ~1615 m. Total drop 713.5 m over 5360.5 m ⇒ 13.3 % mean grade.
- Known distance: raw OSM polyline 5771.9 m; smoothed strip 5360.5 m (7.1 % shortening — the expected geometric effect of 12 m gaussian smoothing on switchbacks, documented in the builder). Smoothed-centerline resample spacing 0.4989–0.5000 m vs 0.5 m arc parameterization.

### 2. Grid resolution / vertex density — PASS (mesh branch, with caveat)
- Sim asset is a mesh (USD triangleMesh), so the mesh branch applies: grid spacing exactly 0.5 m both axes (arc spacing min=max=0.500000 m; cross min=max=0.500000 m), **4.0 verts/m², 8.0 tris/m²**.
- Caveat: 0.5 m is the source DEM's native resolution — sub-0.5 m rock texture is not represented. `meta.json` records `micro_noise: none — terrain/domain randomization is Phase 5`. The charter's 5–10 cm hfield rule is not met by the intermediate npz, but the npz is a build artifact, not the collision asset; 0.5 m cannot be exceeded without inventing data.

### 3. Holes / NaNs / integrity — PASS
- heightfield: **0 NaN, 0 inf** in H[10722, 49]; z range 0–741.73 m, z_offset 1611.786 m; abs max 2353.517 m matches meta bit-exact. (Builder raises on any NaN DEM sample, and none survived.)
- OBJ: 525,378 verts / 1,029,216 faces, both match meta exactly; face count equals the closed-form 2·(10721)·(48); all face indices valid; **0 degenerate faces** (min tri area 0.125 m²); **0 downward-facing normals** (consistent CCW-from-above winding); edge audit: 1,533,055 edges shared by exactly 2 faces, 21,538 boundary edges = exactly the open-strip perimeter, **0 non-manifold edges**. No missing floor patches. (Strip is intentionally open — a static terrain sheet, not a closed solid; fine for triangleMesh collision.)
- OBJ verts vs npz grid: max deviation x 0, y 0, z 5.0e-4 m = exactly the `%.3f` write quantization.

### 4. Mesh budget — **FAIL**
- Collision mesh: **1,029,216 faces > 200,000 charter limit** (5.1×). USD is 6.3 MB usdc with exact triangleMesh collision on all of it.
- Mitigating evidence: PhysX cooking 4.6 s, 8-env settle gate passes on the 8 GB card. Per-area density (8 tris/m² over 5360×24 m) is modest; the breach is total count driven by strip length. Options: waiver, quadric decimation of low-slope sections, or splitting into per-curriculum-section tiles.

### 5. Headless scene load — PASS (evidence: gate log, not re-run)
- `runs/isaac/gates.log` `[gate trail]` 2026-08-07 23:28:57–23:29:13: SimulationApp up 7.4 s, env (Dome-G1FullCollision-EigerTrail-v0, `sims/isaac/tasks/dome_g1/trail_env_cfg.py`) built in 4.6 s including PhysX cooking of ~1M tris, 8 envs, no errors/warnings logged, gate PASS.
- File-level sanity here: `usd/eiger_trail.usd` magic `PXR-USDC`, 6.3 MB, MeshConverter config + asset hash present.

### 6. G1 settle test — PASS (evidence: gate log, not re-run)
- Same gate: origins z-span assert (x span 14 m, z span 2 m over 8 envs), root sits 0.74 m above origin after reset, **worst drop below origin 0.18 m after 100 zero-action steps**, observations finite, no solver warnings / NaN / contact explosions. Note: 0.18 m is settle drop relative to origin z (includes conforming to local terrain), not a direct penetration measurement; per-contact max penetration was not separately instrumented.

### 7. Statistics vs source — PASS
- Meta stats reproduce **bit-exact** from heightfield.npz: drop 713.507 m, grade 13.310 %, along-track slope p50 17.224 % / p95 55.125 % / max 4214.28 %.
- Claimed extreme slopes confirmed real and localized: centerline slope p50 16.5 % / p95 48.2 % / **p99 75.4 % (claim: 75 %)**, max 357.5 %, **exactly 34 centerline cells > 100 % (claim: 34)**. Full-strip cells with |grad| > 100 %: 20,844 / 514,608 (4.05 %) — cliff bases at strip edges.
- Strip vs source DEM slope distribution at the *same map-frame footprint* (bilinear central differences on the tile mosaic, every 2nd row): p50 48.0 vs 48.0 %, p90 79.3 vs 78.4 %, p95 94.2 vs 93.0 %, p99 168.7 vs 162.7 %, mean 53.0 vs 52.9 % — match within differencing-scheme noise. Re-sampling the DEM at the strip's exact footprint reproduces H to **0.03 mm**.
- Roughness (4-neighbor Laplacian residual): p50 1.94 cm, p95 9.14 cm, max 7.59 m (cliff cells) — consistent with raw 0.5 m DEM, no smoothing/filtering artifacts in z.

### Origins walkability (task-specific) — PASS
- origins.npz: [1089, 3] float32, 0 NaN, x 10–5350 m, all y = 0 (centerline), z on surface (z equals H[i, 24] exactly at all 10 spot-checked origins).
- Builder filter reproduced independently: 2671 candidates, 1089 kept — **byte-identical** to the file (max |Δ| = 0.0). Kept-origin max ±1 m patch relief 1.000 m (limit 1.0); rejected 1582 candidates had relief p50 1.31 m / max 7.05 m. The ≤1.0 m walkability claim holds.

### Straightening artifacts (task-specific) — PASS with quantified caveat
- Smoothed centerline min turn radius **2.3 m** (max |κ| 0.432 /m) < 12 m half-width ⇒ fold-over exists at the sharpest switchbacks despite 12 m smoothing.
- Sweep Jacobian (1 − κ·y): min −4.19; **1891 / 525,378 cells (0.36 %) fold over (J ≤ 0)**, 6113 cells (1.16 %) compressed ≥2×; fold-over confined to **207 / 10,722 rows (1.93 %)**, always at the outer strip edges.
- Map-frame footprint duplication: 13.41 % of samples share a 0.5 m DEM cell overall, 10.71 % even in the inner ±2 m corridor — the inner figure is dominated by rounding collisions of adjacent 0.5 m samples, so J ≤ 0 (0.36 %) is the honest fold-over metric.
- All 1089 spawn origins sit at y = 0 where J ≡ 1: spawns and the trail bench itself are unaffected; artifacts are cosmetic edge terrain a policy may brush against, matching the builder's documented tradeoff.

## Repro
```
# gate evidence (already run 2026-08-07 23:28): tail runs/isaac/gates.log
%USERPROFILE%\venvs\isaac\Scripts\python.exe <scratchpad>\validate_eiger.py
#   (script archived logic: npz/meta/OBJ cross-check, edge/degeneracy audit,
#    origin-filter reproduction, per-tile bilinear DEM spot checks at
#    s = {0, 1072, 2144, 2680.5, 3216.5, 4288.5, 5360.5} m,
#    curvature/Jacobian fold-over census, map-frame slope-distribution comparison)
# rebuild for comparison (do not overwrite blindly):
%USERPROFILE%\venvs\isaac\Scripts\python.exe sims\isaac\terrain\build_trail_terrain.py
```

Not verified: Isaac USD stage re-open (relied on tonight's `[gate trail]` PASS, not re-run); per-contact max penetration (gate reports settle drop 0.18 m, not penetration depth); MuJoCo legacy load path (not applicable to this Isaac asset).
