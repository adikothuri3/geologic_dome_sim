# Terrain validation report — no asset found — 2026-08-01

**Asset:** none (smoke test — repo-wide search)
**Source point cloud:** none
**Overall verdict: FAIL — nothing to validate.**

## Search performed

Searched the entire working tree at `C:\Users\Aditya\VSCode\GeologicDome` (tracked, untracked, and gitignored paths; `.git` excluded) for:

- MJCF: `*.xml`, `*.mjcf`
- Heightfields: `*.png`, `*.npy`, `*.npz`, `*.tif`, `*.tiff`, `*.h5`
- Collision meshes: `*.obj`, `*.stl`, `*.ply`
- Point clouds: `*.ply`, `*.pcd`, `*.las`, `*.laz`

**Result: 0 matching files.** The repo currently contains only documentation (`notes/`, `lab-notebook/`), agent/skill definitions (`.claude/`, `.agents/`), and config. This is consistent with `notes/overview.md`, where Phase 3 (reconstruction) and Phase 4 (real2sim terrain) are both marked **not started** — no reconstruction output or terrain asset exists yet.

Repro:

```bash
cd /c/Users/Aditya/VSCode/GeologicDome
find . -path ./.git -prune -o -type f \( -iname '*.xml' -o -iname '*.mjcf' \
  -o -iname '*.png' -o -iname '*.npy' -o -iname '*.npz' -o -iname '*.h5' \
  -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.obj' -o -iname '*.stl' \
  -o -iname '*.ply' -o -iname '*.pcd' -o -iname '*.las' -o -iname '*.laz' \) -print
# → no output
```

## Checklist results

| # | Check | Result | Measured |
|---|-------|--------|----------|
| 1 | Scale calibration (known-distance test) | FAIL | no asset / no cloud |
| 2 | Grid resolution 5–10 cm | FAIL | no hfield |
| 3 | Holes / NaNs in height data | FAIL | no height data |
| 4 | Mesh budget < 200k faces | FAIL | no mesh |
| 5 | MJCF loads headless, zero errors/warnings | FAIL | no MJCF |
| 6 | G1 settle test (penetration, settle time) | FAIL | no scene to load |
| 7 | Slope/roughness stats vs. source cloud | FAIL | no asset, no cloud |

All checks FAIL for the same reason: **no terrain asset and no source point cloud exist in the repository.** No measurements were fabricated.

## What would make this pass

Once Phase 3/4 produce artifacts, re-run this validator pointing at:

1. The cleaned point cloud (post `open3d-cleanup`), and
2. The generated MJCF scene + hfield PNG/array and/or collision mesh (post `mjcf-terrain`).

Expected per project conventions (`notes/pipeline.md`, `mjcf-terrain` skill): 5–10 cm hfield cells, robust max-z with hole filling, collision mesh under 200k faces, scale calibration applied before terrain build.

## Not verified

All seven checks were skipped for lack of inputs: scale calibration, grid resolution, hole/NaN scan, mesh face budget, headless MJCF load, G1 settle test, slope/roughness statistics comparison.
