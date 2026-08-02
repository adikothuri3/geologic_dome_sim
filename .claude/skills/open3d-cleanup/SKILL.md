---
name: open3d-cleanup
description: Clean a raw LingBot-Map point cloud with Open3D — outlier removal, voxel downsample, ground-plane alignment — using this project's default parameters. Use on any reconstruction output before terrain building.
---

# open3d-cleanup

Input: raw point cloud from `lingbot-recon` (already confidence-filtered at export). Output: a cloud ready for the `mjcf-terrain` skill.

Order matters — run exactly this sequence:

## 1. Statistical outlier removal

`pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)` — project default starting point. Tighten `std_ratio` toward 1.5 only if floater clusters survive; loosen if real thin structures vanish. Inspect before/after in the viewer.

## 2. Voxel downsample

`pcd.voxel_down_sample(voxel_size=0.02)` (2 cm). Keep voxel size well below the terrain grid cell (5–10 cm in `mjcf-terrain`) so the robust max-z per cell still has multiple points to draw from.

## 3. Ground-plane alignment

`pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)` on the lower portion of the cloud → rotate so the plane normal is +Z, translate plane to z=0. On sloped trail terrain fit the plane to a local flat patch (start of the walk), not the whole cloud — a whole-cloud fit tilts the world.

## Notes

- Scale calibration is **not** done here — it belongs to `mjcf-terrain`'s precondition; but if the calibration reference (marker distance) is known, apply the uniform scale before step 2 so voxel sizes are in true meters.
- Save output as `.ply` next to the input with a `_clean` suffix; keep the raw cloud.
- These defaults are the current convergence point — when a better setting proves out on real data, update this file (and note the change in `lab-notebook/`).
