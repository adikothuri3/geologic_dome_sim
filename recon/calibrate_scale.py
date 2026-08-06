"""Turn a monocular reconstruction's arbitrary units into metres.

Monocular reconstruction has no absolute scale, and `.claude/skills/mjcf-terrain`
refuses to build terrain without it: terrain 10% too large changes the step
heights a policy trains on. This is the unresolved item 2 in
`notes/open-questions.md`; this script is the home-lab half of the answer, and
the expedition half (markers / GPS / Pemba's dimensions) plugs into the same
`--factor` path.

Anchors, in order of how much they should be trusted:

  ``camera``   Median camera height above the fitted ground plane, against an
               assumed eye/chest height. Averaged over every pose in the clip,
               so it is the most robust anchor we have -- *provided the poses
               are trustworthy*. It is worthless on a run whose trajectory
               drifted, so the script refuses when the drift ratio says so.

  ``factor``   A number you worked out yourself (measured markers, a known
               object, a published building dimension). Always available.

Whatever the anchor, the derived scene is cross-checked against
human-plausible bounds and the run is refused if it lands outside them --
a wrong scale that silently proceeds is worse than no scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from clean_cloud import fit_ground_plane

# A reconstruction whose camera path is many times its own extent has drifted,
# and its poses cannot anchor anything. Loop sits at 3.4; the courthouse runs
# at 25 are exactly the case this guard exists to catch.
MAX_DRIFT_RATIO = 6.0
# Walking eye/chest height for a handheld or chest-mounted camera.
DEFAULT_CAMERA_HEIGHT_M = 1.5
# Anything outside these is a calibration failure, not a scene.
MIN_SCENE_M, MAX_SCENE_M = 1.0, 500.0


def load(run_dir: Path, cloud_name: str):
    cloud = run_dir / cloud_name
    if not cloud.exists():
        raise SystemExit(f"missing {cloud}")
    pcd = o3d.io.read_point_cloud(str(cloud))
    if len(pcd.points) == 0:
        raise SystemExit("cloud is empty")
    traj = run_dir / "trajectory.npz"
    C = None
    if traj.exists():
        C = np.load(traj)["extrinsic"].astype(np.float64)[:, :3, 3]
    return pcd, C


def camera_anchor(pcd, C, target_h: float):
    """metres-per-unit from median camera height above the ground plane."""
    if C is None:
        raise SystemExit("--anchor camera needs trajectory.npz")

    pts = np.asarray(pcd.points)
    extent = pts.max(0) - pts.min(0)
    cands = fit_ground_plane(pcd, 0.01 * float(extent.max()), all_candidates=True)
    if not cands:
        raise SystemExit("no ground plane found; use --anchor factor")

    # Pick the floor, not the biggest plane. A person walking holds a roughly
    # constant height above the floor and a wildly varying distance from any
    # wall, so the plane that minimises spread in camera height is the floor --
    # even when a wall collects far more inliers.
    best = None
    for model, n_in in cands:
        if n_in < 0.02 * len(pts):
            continue
        a, b, c, d = model
        n = np.array([a, b, c], float)
        nrm = np.linalg.norm(n)
        n, d = n / nrm, d / nrm

        h_pts, h_cam = pts @ n + d, C @ n + d
        if np.median(h_cam) < 0:
            n, d, h_pts, h_cam = -n, -d, -h_pts, -h_cam

        floor = np.percentile(h_pts, 1.0)
        h = h_cam - floor
        cam = float(np.median(h))
        if cam <= 0:
            continue
        spread = float(np.percentile(h, 95) - np.percentile(h, 5)) / cam
        if best is None or spread < best[0]:
            best = (spread, n, d, cam, floor, h, h_pts, n_in)

    if best is None:
        raise SystemExit("no usable ground plane; use --anchor factor")

    spread, n, d, cam, floor, h, h_pts, n_in = best
    print(f"  ground plane: {n_in:,} inliers, normal {n.round(3)} "
          f"(chosen from {len(cands)} candidates by camera-height consistency)")
    print(f"  camera height above floor: {cam:.4f} units "
          f"(p5-p95 {np.percentile(h, 5):.3f}-{np.percentile(h, 95):.3f}, "
          f"spread {100*spread:.0f}% of median)")
    print(f"  structure reaches {float(h_pts.max() - floor):.4f} units above floor")
    if spread > 0.6:
        print("  WARNING: camera height varies a lot -- either the plane is not the "
              "floor, or the walk changed level. Treat the scale as soft.")
    return target_h / cam, ([round(float(v), 6) for v in n], round(float(d), 6))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--anchor", choices=["camera", "factor"], default="camera")
    ap.add_argument("--camera-height", type=float, default=DEFAULT_CAMERA_HEIGHT_M)
    ap.add_argument("--factor", type=float, default=None,
                    help="metres per unit, for --anchor factor")
    ap.add_argument("--cloud", default="cloud.ply")
    ap.add_argument("--force", action="store_true",
                    help="write scale.json even if the sanity checks fail")
    a = ap.parse_args()

    pcd, C = load(a.run_dir, a.cloud)
    pts = np.asarray(pcd.points)
    extent = pts.max(0) - pts.min(0)
    print(f"{a.run_dir.name}: {len(pts):,} points, extent {extent.round(3)} (arbitrary)")

    run = a.run_dir / "run.json"
    ratio = json.loads(run.read_text()).get("traj_length_over_extent") if run.exists() else None

    if a.anchor == "camera":
        if ratio is not None and ratio > MAX_DRIFT_RATIO:
            raise SystemExit(
                f"refusing the camera anchor: traj_length_over_extent = {ratio} "
                f"(> {MAX_DRIFT_RATIO}). The poses drifted, so camera height means "
                f"nothing here. Re-run with --anchor factor --factor <m per unit>.")
        m_per_unit, ground = camera_anchor(pcd, C, a.camera_height)
    else:
        if a.factor is None:
            raise SystemExit("--anchor factor needs --factor")
        m_per_unit, ground = a.factor, (None, None)

    scene = extent * m_per_unit
    print(f"\n  scale = {m_per_unit:.4f} m/unit")
    print(f"  scene extent  {scene.round(2)} m")

    nn = np.median(pcd.compute_nearest_neighbor_distance()) * m_per_unit
    print(f"  point spacing {nn*100:.2f} cm  "
          f"({'ok' if nn < 0.02 else 'COARSE for a 2 cm voxel'})")

    ok = MIN_SCENE_M <= float(scene.max()) <= MAX_SCENE_M
    if not ok:
        print(f"\n  IMPLAUSIBLE: largest dimension {scene.max():.1f} m is outside "
              f"{MIN_SCENE_M}-{MAX_SCENE_M} m")
        if not a.force:
            raise SystemExit(2)

    out = a.run_dir / "scale.json"
    out.write_text(json.dumps({
        "m_per_unit": round(float(m_per_unit), 6),
        "anchor": a.anchor,
        "camera_height_m": a.camera_height if a.anchor == "camera" else None,
        "extent_m": [round(float(v), 3) for v in scene],
        "point_spacing_m": round(float(nn), 5),
        "plausible": bool(ok),
        # Handed to clean_cloud so it aligns to the plane we *verified* is the
        # floor, instead of re-fitting and risking the biggest-plane-is-a-wall
        # trap. Stored in pre-scale units; multiply d by m_per_unit after scaling.
        "ground_normal": ground[0], "ground_offset": ground[1],
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
