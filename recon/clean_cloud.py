"""Clean a raw LingBot-Map cloud with Open3D, and render it properly.

Implements the `open3d-cleanup` skill's sequence: statistical outlier removal ->
voxel downsample -> ground-plane alignment. Adds a radius-outlier pass, because
the dominant artefact in these reconstructions is not isolated noise but
"flying pixels": thin radial streaks shed from depth discontinuities at
occlusion edges (doorframes, furniture silhouettes). Statistical removal alone
leaves them, since each streak is locally dense along its own filament.

One deviation from the skill's constants: monocular reconstruction has arbitrary
scale, so a hardcoded 2 cm voxel is meaningless here -- a cloud whose whole extent
is 2.4 units would be obliterated or untouched depending on the unknown scale
factor. Sizes are therefore expressed as a fraction of the cloud's extent unless
--absolute is passed. Once Phase 4 scale calibration exists, use --absolute and
the skill's true-metre defaults.

Ground alignment also earns its place visually: rendering a cloud whose floor is
an arbitrary oblique plane is most of why a decent reconstruction can look like
garbage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

# Fraction-of-extent defaults, for a cloud whose scale is still arbitrary.
RADIUS_FRAC, VOXEL_FRAC = 0.004, 0.0015
# True-metre defaults once --scale is known. VOXEL_M is the open3d-cleanup skill's
# 2 cm: well under the 5-10 cm hfield cell, so robust max-z still has points to pick
# from. RADIUS_M is tight enough to strip occlusion-edge streaks without eating
# genuinely thin structure.
RADIUS_M, VOXEL_M, GROUND_M = 0.03, 0.02, 0.05


def fit_ground_plane(pcd: o3d.geometry.PointCloud, dist: float,
                     all_candidates: bool = False):
    """RANSAC the dominant ground plane. Returns ``(model, n_inliers)`` or None.

    Fit on the lower slab only. A whole-cloud fit happily returns a wall -- in a
    room the walls carry more points than the floor.

    Inlier count alone is *not* enough to identify the floor: in a corridor the
    walls carry more points than the carpet even within the lower slab, and the
    winning plane comes back vertical. With ``all_candidates`` the caller gets
    every axis-direction fit, sorted by inlier count, so it can apply a better
    test -- ``calibrate_scale`` picks whichever plane keeps the camera at a
    near-constant height, which is what "floor" physically means for a walk.

    Split out from ``align_ground`` because scale calibration needs the plane
    without the rotation: camera height above the ground is our metric anchor.
    """
    pts = np.asarray(pcd.points)

    # "Lower" is unknown before alignment, so try each axis direction and keep
    # the plane supported by the most inliers.
    cands = []
    for axis in range(3):
        for sign in (1, -1):
            proj = sign * pts[:, axis]
            cut = np.percentile(proj, 40)
            sel = np.where(proj <= cut)[0]
            if len(sel) < 100:
                continue
            sub = pcd.select_by_index(sel)
            try:
                model, inliers = sub.segment_plane(
                    distance_threshold=dist, ransac_n=3, num_iterations=1000
                )
            except RuntimeError:
                continue
            cands.append((model, len(inliers)))

    if not cands:
        return None
    cands.sort(key=lambda c: -c[1])
    return cands if all_candidates else cands[0]


def align_ground(pcd: o3d.geometry.PointCloud, dist: float, plane=None):
    """Rotate so the dominant floor plane is +Z and sits at z=0.

    ``plane`` short-circuits the fit with a model calibrate_scale already
    confirmed is the floor -- worth using, because picking by inlier count alone
    can land on a wall (see ``fit_ground_plane``).
    """
    pts = np.asarray(pcd.points)
    best = (plane, -1) if plane is not None else fit_ground_plane(pcd, dist)

    if best is None:
        print("  ground plane: no fit found, leaving orientation unchanged")
        return pcd, None

    a, b, c, d = best[0]
    n = np.array([a, b, c], dtype=float)
    d /= np.linalg.norm(n)
    n /= np.linalg.norm(n)
    # Point the normal "up" relative to the bulk of the cloud.
    centroid = pts.mean(0)
    if np.dot(n, centroid) + d < 0:
        n, d = -n, -d

    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, target)
    s, cth = np.linalg.norm(v), float(np.dot(n, target))
    if s < 1e-8:
        R = np.eye(3) if cth > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / (s**2))

    pcd.rotate(R, center=(0, 0, 0))
    pts = np.asarray(pcd.points)
    pcd.translate((0, 0, -np.percentile(pts[:, 2], 1)))
    print(f"  ground plane: {best[1]:,} inliers, rotated floor to +Z")
    return pcd, best[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--nb-neighbors", type=int, default=20)
    ap.add_argument("--std-ratio", type=float, default=2.0)
    ap.add_argument("--radius-frac", type=float, default=None,
                    help=f"radius-outlier search radius; fraction of extent, or metres "
                         f"under --scale/--absolute (defaults {RADIUS_FRAC} / {RADIUS_M} m)")
    ap.add_argument("--min-neighbors", type=int, default=12,
                    help="points with fewer neighbours in that radius are streaks")
    ap.add_argument("--voxel-frac", type=float, default=None,
                    help=f"voxel size, same units as --radius-frac "
                         f"(defaults {VOXEL_FRAC} / {VOXEL_M} m)")
    ap.add_argument("--scale", default=None,
                    help="metres per arbitrary unit, or 'auto' to read scale.json; "
                         "implies --absolute and switches sizes to the skill's true metres")
    ap.add_argument("--absolute", action="store_true",
                    help="treat --radius-frac/--voxel-frac as true units, not fractions")
    ap.add_argument("--no-align", action="store_true")
    a = ap.parse_args()

    src = a.run_dir / "cloud.ply"
    if not src.exists():
        raise SystemExit(f"missing {src}")

    pcd = o3d.io.read_point_cloud(str(src))
    n0 = len(pcd.points)
    if n0 == 0:
        raise SystemExit("cloud is empty")

    # Scale first, so every size below is a real metre and matches the
    # open3d-cleanup skill's constants instead of a fraction of an arbitrary box.
    m_per_unit, ground_plane = None, None
    if a.scale is not None:
        if a.scale == "auto":
            sf = a.run_dir / "scale.json"
            if not sf.exists():
                raise SystemExit(f"--scale auto needs {sf}; run recon/calibrate_scale.py first")
            rec = json.loads(sf.read_text())
            m_per_unit = float(rec["m_per_unit"])
            if rec.get("ground_normal"):
                # Stored pre-scale: n.x + d = 0 becomes n.x + m*d = 0 after scaling.
                ground_plane = list(rec["ground_normal"]) + [rec["ground_offset"] * m_per_unit]
        else:
            m_per_unit = float(a.scale)
        pcd.scale(m_per_unit, center=(0, 0, 0))
        a.absolute = True
        print(f"scaled by {m_per_unit:.4f} m/unit -> cloud is now in metres")

    extent0 = np.asarray(pcd.points).max(0) - np.asarray(pcd.points).min(0)
    scale = 1.0 if a.absolute else float(np.max(extent0))
    radius = a.radius_frac if a.radius_frac is not None else (RADIUS_M if a.absolute else RADIUS_FRAC)
    voxel = a.voxel_frac if a.voxel_frac is not None else (VOXEL_M if a.absolute else VOXEL_FRAC)
    ground_dist = GROUND_M if a.absolute else 0.01 * float(np.max(extent0))
    print(f"input: {n0:,} points, extent {extent0.round(3)}"
          f"{' m' if a.absolute else ' (arbitrary)'}")

    print("1. statistical outlier removal")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=a.nb_neighbors, std_ratio=a.std_ratio
    )
    n1 = len(pcd.points)
    print(f"   {n0:,} -> {n1:,}  (-{100*(n0-n1)/n0:.1f}%)")

    print("2. radius outlier removal (flying pixels)")
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=a.min_neighbors, radius=radius * scale
    )
    n2 = len(pcd.points)
    print(f"   {n1:,} -> {n2:,}  (-{100*(n1-n2)/max(n1,1):.1f}%)")

    print("3. voxel downsample")
    pcd = pcd.voxel_down_sample(voxel_size=voxel * scale)
    n3 = len(pcd.points)
    print(f"   {n2:,} -> {n3:,}")

    if not a.no_align:
        print("4. ground-plane alignment"
              + ("  (using calibrated plane)" if ground_plane else ""))
        pcd, plane = align_ground(pcd, ground_dist, plane=ground_plane)

    out = a.run_dir / "cloud_clean.ply"
    o3d.io.write_point_cloud(str(out), pcd)
    extent1 = np.asarray(pcd.points).max(0) - np.asarray(pcd.points).min(0)
    print(f"\nwrote {len(pcd.points):,} points -> {out}")

    (a.run_dir / "clean_stats.json").write_text(json.dumps({
        "n_in": int(n0), "n_after_statistical": int(n1),
        "n_after_radius": int(n2), "n_out": int(len(pcd.points)),
        "removed_pct": round(100 * (n0 - len(pcd.points)) / n0, 1),
        "extent_in": [round(float(v), 3) for v in extent0],
        "extent_out": [round(float(v), 3) for v in extent1],
        "ground_aligned": not a.no_align,
        "m_per_unit": m_per_unit,
        "units": "m" if m_per_unit else "arbitrary",
        "radius": radius, "voxel": voxel,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
