# Terrain validation — everest (Isaac track) — 2026-08-08 — PASS 2 of 2 (FINAL)

**Asset:** `data/everest/terrain/` (heightfield.npz, origins.npz, everest_terrain.obj, meta.json) + `data/everest/terrain/usd/everest_summit.usd` (754,249 B, MeshConverter 2026-08-08 18:58:40)
**Source:** NASA/NSIDC HMA_DEM8m_MOS v1 tile-677 (`data/hma_dem/HMA_DEM8m_MOS_20170716_tile-677.tif`, 12500×12500 px, 8 m, WorldView/GeoEye stereo)
**Builder:** `sims/isaac/terrain/build_everest_terrain.py` (axis-aligned summit crop, no resampling; clone of the gated Eiger pipeline); USD via `sims/isaac/terrain/mesh_to_usd.py` (MeshConverter)
**Scope:** Pass 1 (earlier today, same file, see git history) validated the npz/OBJ/origins/meta artifacts and deferred checks 5–6 pending the USD. This final pass verifies the USD + gate results (`sims/isaac/scripts/check_everest.py --num_envs 42`, logged in `runs/isaac/gates.log` 18:59) and closes the report.
All checks read-only; no assets modified.

## Verdict: **PASS (final)** — all seven applicable checks pass with measured numbers. Asset is training-ready.

Standing caveats (do not fail any stated criterion):
(a) 8 m posting is 80–160× coarser than the charter's 5–10 cm hfield rule (native DEM
resolution, `micro_noise: none` by design); (b) 38/42 origins have at least one ≥20°
one-sided step to an 8 m neighbor — the settle gate exercised all 42 (including the
three worst) and none fell through; keep all 42 origins (rationale under check 6);
(c) contact-level penetration depth was measured by the gate's root-vs-origin proxy,
not per-contact PhysX depth (see check 6).

## Checks

### 1. Scale calibration — PASS
- GeoTIFF pixel scale tag: exactly **8.0 × 8.0 m** (isotropy asserted by the builder; re-read here).
- Projection: geokeys assert Albers method (3075=11) with exactly the HMA constants (25/47, 36N 85E, FE=FN=0); builder round-trips the tiepoint to <1 cm (re-ran: passes).
- Known-point test (independent of the builder's projection code — Vincenty on WGS84): DEM global max lands at grid (x=+20.0, y=+4.0) = **20.4 m** from patch center at bearing 79°; published summit (27.988056 N, 86.925278 E) is **28.0 m** from the requested center (27.988, 86.925) at bearing ~77°. Position error vector ≈ **7.6 m < 1 cell** — sub-cell georegistration.
- Known elevation: DEM summit **8817.12 m** = expected ~8817 for this DEM bit-exact vs meta (survey 8849 m is snow/method-dependent; −32 m stereo-vs-snow bias is documented HMA behavior).
- Albers equal-area distance distortion at 28 N (analytic): k = 0.99216 along parallels, 1.00790 along meridians — grid distances are within **0.79 %** of ground truth, an inherent, bounded property of the chosen CRS.
- USD carries no rescale: `config.yaml` scale = (1.0, 1.0, 1.0); stage `metersPerUnit = 1.0`; USD bounds x,y ∈ [−996, +996], z ∈ [0, 1750.358] — identical to the OBJ/npz (verified this pass).
- Long-baseline probe (supplementary, inconclusive): at the expected South Summit offset (Vincenty 270.8 m from summit, convergence-rotated) the DEM reads 8564 m vs ~8717 expected — attributable to ±11 m precision of published South Summit coordinates on a knife ridge plus crest aliasing at 8 m posting; does not contradict the sub-cell summit fix above.

### 2. Grid resolution / vertex density — PASS (mesh branch, with major caveat)
- Grid spacing exactly **8.000000 m** both axes (dx and dy min=max=8.0 across all 249 intervals); x and y span −996..+996 m (250 cells = 2000 m footprint).
- Mesh density: **0.0156 verts/m², 0.0310 tris/m²**.
- Caveat (larger than Eiger's): 8 m is the source DEM's native posting — the charter's 5–10 cm hfield rule is off by ~100×, and the G1 (0.2 m feet) will see 8 m planar facets. `meta.json` records `micro_noise: none` deliberately (honesty rule: no synthesized detail). Terrain/domain randomization remains the Phase 5 answer; cannot exceed 8 m without inventing data.

### 3. Holes / NaNs / integrity — PASS
- heightfield: **0 NaN, 0 inf** in H[250,250] float32; z 0–1750.358 m, z_offset 7066.758 m; abs max 8817.116 — all bit-exact vs meta.
- Void fill audit vs raw tile crop: raw crop has exactly **71 NaN cells (0.1136 %)** = meta's `void_cells_filled`/`void_fraction_initial`; largest 4-connected void blob **14 cells** (2 fill passes sufficed, matching `fill_max_pass_used: 2`); voids sit on steep faces at 8089–8740 m abs. All **71/71 filled values lie within the local 7×7 raw [min, max]** — no extrapolation spikes. Filled-vs-raw on the 62,429 valid cells: **max |diff| = 0.000000 m** (the build is a pure crop, verified).
- OBJ: 62,500 verts / 124,002 faces, both match meta and the closed-form 2·(249)²; all face indices valid; **0 zero-area faces** (min tri area 32.01 m²); **0 downward normals** (consistent CCW-from-+z winding); edge audit: 185,505 edges shared by exactly 2 faces, **996 boundary edges = exactly the patch perimeter**, **0 non-manifold edges**. No missing floor patches (open sheet by design, fine for triangleMesh collision).
- OBJ verts vs npz grid: max deviation x 0, y 0, z **5.0e-4 m** = exactly the `%.3f` write quantization.
- USD-vs-OBJ (this pass, usd-core, no Kit): single mesh prim `/everest_terrain/geometry/mesh`, **62,500 points / 124,002 faces, all triangles**; face-index arrays **byte-identical** to the OBJ; max vertex coordinate delta **6.055e-05 m** (float32 quantization of the OBJ's `%.3f` values) — the USD is the OBJ, bit-consistent. (`.asset_hash` = `da00b33290bd984b63c78c9b128d3cbc` is MeshConverter's config+asset digest, not a plain file MD5, so geometry-level comparison was used instead — it is the stronger check.)

### 4. Mesh budget — PASS
- **124,002 faces ≤ 200,000** charter limit (62 % of budget). No waiver needed — unlike the Eiger strip (1.03 M, FAIL/waiver-pending). Confirmed identical count inside the USD (check 3).

### 5. Headless scene load — PASS
- `config.yaml` records exactly what was requested: `mesh_approximation_name: none` (exact triangle-mesh collision; `physx_func: pxr.PhysxSchema:PhysxTriangleMeshCollisionAPI`), `scale: (1.0, 1.0, 1.0)`, identity translation/rotation, **`mass_props: null`, `rigid_props: null`** (static collider, as required), `collision_enabled: true`, `make_instanceable: false`.
- Independent stage open with usd-core (no Kit): stage opens cleanly; **Z-up, metersPerUnit 1.0**; 3 prims, exactly 1 mesh; identity xformOps at every level; applied schemas on the mesh: `PhysicsCollisionAPI` + `PhysicsMeshCollisionAPI` with **`physics:approximation = "none"`**; **no RigidBodyAPI/MassAPI anywhere** on the stage. Geometry bit-consistent with the OBJ (check 3).
- Full Isaac load (gate, 18:59): **SimulationApp up in 6.5 s; env built in 2.3 s including PhysX cooking the ~125k-tri collision mesh**; the height-scan RayCaster found the mesh (finite observations, check 6); env origins are the patch spawn points, not the stock z=0 grid (**xy span 1680 m, z span 873 m over 42 envs** — matches origins.npz exactly: x −716..964 → 1680 m, z abs 7944–8817 → 873 m). Zero errors in `runs/isaac/gates.log`; the gate logs `FAILED` + traceback on any exception and exits 1 — it logged `PASS`.
- Caveat: Kit's stderr warning stream is not archived in gates.log, so "zero warnings" is verified to the extent of the gate's assertions plus the clean log, not a captured console transcript.

### 6. G1 settle test — PASS
- Command: `check_everest.py --num_envs 42`. The importer tiles origins modulo num_envs (per `everest_env_cfg.py` docstring), so 42 envs = **each of the 42 origins exercised exactly once, including the three flagged worst: (164, −12), (324, −316), (308, −284)**.
- After reset: root sits **0.74..0.74 m above origin z** on every env (uniform — all spawns valid).
- Settle window: **100 zero-action steps = 2.0 s sim time** (step_dt = sim.dt 0.005 s × decimation 4, stock `LocomotionVelocityRoughEnvCfg` values, verified in the Isaac Lab source install). Gate wall time **22.1 s total**.
- Results: **observations finite at all 100 steps** (includes the height scan — no NaN state, no raycast misses); no contact explosions or solver failures logged; **worst drop below origin 3.72 m**, under the 5 m fall-through threshold. **PASS.**
- Assessment of the 3.72 m worst drop — **topple, not penetration; acceptable**:
  - Under zero actions the G1 collapses everywhere; on a flat cell the root ends ~0.3–0.4 m below its 0.74 m spawn height, i.e. still *above* origin z. A robot 3.72 m *below* origin z has come to rest on lower neighboring terrain — exactly what sliding/toppling off an 8 m ledge bench produces, and consistent with the flagged origins (one-sided neighbor steps 65.7–84.8°, 3×3 local relief up to 131.9 m).
  - Penetration would not stop at 3.72 m: a robot falling *through* the mesh free-falls ≈19.6 m in the 2 s window and keeps going (the gate's own criterion notes fall-through robots end up "hundreds of metres down", because env resets teleport surviving robots back to origins — a *persistent* large drop is the penetration signature). 3.72 m is bounded and static.
  - Corroboration: the height-scan observations stayed finite for all envs all 100 steps — every robot remained above the collision mesh.
- **Verdict on origins: keep all 42.** The stated criterion was penetration, not toppling; env resets handle fallen robots by design; and standable cells adjacent to drops are representative of this terrain, not defects. Recommendation for Phase 5: monitor per-origin early-termination rates during training and revisit only if specific origins prove unlearnable — drop origins then rather than rebuilding.
- Caveat: max *contact* penetration depth (per-contact PhysX depth) was not directly measured; the gate's proxy is root height vs origin (bounded at 3.72 m, threshold 5 m) plus finite observations. This satisfies the gate's stated criterion.

### 7. Statistics vs source — PASS
- Meta stats reproduce **bit-exact** from heightfield.npz: slope p50 46.504°, p95 68.143°, max 86.483°; relief 1750.358 m.
- Slope histogram, filled patch vs raw source crop (same footprint, same 8 m central differences): p10 36.61/36.62, p25 40.41/40.41, p50 46.50/46.49, p75 53.74/53.70, p90 62.23/62.13, p95 68.14/67.97, p99 77.34/76.95°; mean 47.93 vs 47.90°. Differences are confined to the 71 filled cells (0.11 %) — distributions match.
- Roughness (|4-neighbor Laplacian|): patch p50 0.362 m, p95 3.501 m, max 95.96 m; raw source p50 0.360, p95 3.435, max 95.95 m — match. The 96 m max is a coherent multi-cell ridge block at (x=204, y=12, ~8728 m abs) present in the raw DEM (Kangshung-rim serac/pinnacle or stereo artifact) — **not** a build artifact. Pinnacle census: **0 cells** stand >20 m above all 8 neighbors, so no single-cell spikes anywhere.
- Plausibility for the summit pyramid: p50 46.5° / mean 47.9° matches the ~45–55° faces of the upper pyramid; E-W profile through the summit (`reports/everest_profile.png`) shows the 8817 m apex with steep ridge notches to the east, consistent with the map/oblique renders.

### Origins standability (task-specific) — PASS (settle-gate confirmed)
- origins.npz: [42, 3] float32, 0 NaN; x −716..964, y −620..836 m; z abs 7944–8817 m (873 m spread); all 42 at unique 16 m sites; min edge margin 32 m (≥24 required).
- Filter reproduced independently: 122² = **14,884 candidates** (matches meta), step 2 cells, margin 3 cells; slope ≤ 15° → **18** (rung 0, below the 40 floor — confirms the recorded fallback), slope ≤ 20° → **42** kept, **byte-identical** to the file (`np.array_equal` True). 42 ≥ 40 floor. Origin z equals H at the exact cell for all 42 (max err 0.0). Origin cell slopes 5.23–19.78° — all within the 20° criterion.
- One-sided 8-neighbor slope census (criterion limitation, deliberate per the recorded ladder decision): **38/42 origins have ≥1 neighbor step >20°, 29/42 >30°, 11/42 >45°, 3/42 >60°**; 3×3 local relief p50 10.2 m, max **131.9 m**. Worst three: (164, −12) one-sided 84.8°, (324, −316) 73.3°, (308, −284) 65.7°.
- **Pass-2 resolution:** the settle gate exercised all 42 (one env per origin); every spawn held — uniform 0.74 m reset height, no fall-through, worst post-collapse drop 3.72 m (a topple onto lower adjacent terrain, within tolerance). **All 42 origins retained.**
- Note: meta.json records `origin_max_slope_deg: 20.0` but not the ladder itself (15°→18 origins→rung 1); the rung-0 count is verified above, so the narrative is confirmed even though the meta field is absent.

### Provenance / meta — PASS
- All meta fields cross-checked against artifacts: built 2026-08-08, CRS constants asserted from geokeys, center_albers = (187871.740, −899585.701), size 2000 m, resolution 8 m, void stats exact, z_offset 7066.7578125 bit-exact, vertex/triangle counts exact. Renders `reports/everest_{map,patch,oblique,profile}.png` visually consistent with the data (42 dots, summit marker ~20 m E of center, 1750 m pyramid, 8817 m profile apex).
- USD provenance: `data/everest/terrain/usd/{everest_summit.usd, config.yaml, .asset_hash}` all written 2026-08-08 18:58; `config.yaml` `asset_path` points at this repo's `everest_terrain.obj`; geometry verified identical (check 3).

## Repro
```
# pass-1 audit (npz/meta/OBJ cross-check, raw-tile crop reproduction, void-fill
# bounds, edge/degeneracy audit, origin-filter byte reproduction, one-sided
# origin slope census, Vincenty summit georegistration, slope/roughness
# histograms vs source):
%USERPROFILE%\venvs\isaac\Scripts\python.exe <scratchpad>\validate_everest.py
%USERPROFILE%\venvs\isaac\Scripts\python.exe <scratchpad>\validate_everest2.py

# pass-2 USD verification (usd-core stage open, mesh-vs-OBJ bit diff,
# collision/physics API audit — no Kit needed):
%USERPROFILE%\venvs\isaac\Scripts\python.exe <scratchpad>\verify_usd.py

# pass-2 gate (all 42 origins; appends to runs\isaac\gates.log):
%USERPROFILE%\venvs\isaac\Scripts\python.exe sims\isaac\scripts\check_everest.py --num_envs 42

# rebuild for comparison (do not overwrite blindly; note --origin-max-slope 20):
%USERPROFILE%\venvs\isaac\Scripts\python.exe sims\isaac\terrain\build_everest_terrain.py --origin-max-slope 20
```

Gate evidence: `runs/isaac/gates.log`, entries 2026-08-08 18:59:08–18:59:24
(`[gate everest] ... PASS (total 22.1s)`).

Not verified (final): captured Kit warning transcript for check 5 (gate asserts + clean log only); per-contact PhysX penetration depth for check 6 (root-vs-origin proxy used, per the gate's stated criterion); South Summit long-baseline distance (inconclusive: published-coordinate precision on a knife ridge); MuJoCo legacy load path (not applicable — Isaac asset).
