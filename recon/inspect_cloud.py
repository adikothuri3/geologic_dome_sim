"""Open3D inspection of a LingBot-Map reconstruction.

Phase 3 deliverable: look at the cloud with the camera trajectory overlaid, and
get numbers rather than vibes about whether the reconstruction is usable.

This is inspection only -- no filtering is written back. Cleanup (statistical
outlier removal, voxel downsample, ground-plane alignment) is Phase 4 and lives
behind the `open3d-cleanup` skill.

Renders offscreen by default so it works over WSL without an X server; pass
--interactive for a live window.

Note on scale: monocular reconstruction is scale-ambiguous, so every distance
printed here is in arbitrary units. Scale calibration is a Phase 4 step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_trajectory(path: Path):
    d = np.load(path, allow_pickle=True)
    return d["extrinsic"], d["intrinsic"], d["cam_centers"]


def trajectory_geometries(cam_centers: np.ndarray, extrinsic: np.ndarray, frustum_every: int,
                          scene_extent: float):
    """A polyline through the camera centres plus periodic camera frustums."""
    geoms = []

    pts = cam_centers.astype(np.float64)
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    traj = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    # Colour-ramp the path along time so direction of travel is readable.
    t = np.linspace(0, 1, max(len(lines), 1))[:, None]
    traj.colors = o3d.utility.Vector3dVector(
        np.hstack([t, np.zeros_like(t), 1.0 - t])  # blue (start) -> red (end)
    )
    geoms.append(traj)

    size = scene_extent * 0.02
    for i in range(0, len(extrinsic), max(1, frustum_every)):
        c2w = np.eye(4)
        c2w[:3, :4] = extrinsic[i]
        f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        f.transform(c2w)
        geoms.append(f)

    return geoms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="folder containing cloud.ply + trajectory.npz")
    ap.add_argument("--interactive", action="store_true", help="open a live window instead of rendering")
    ap.add_argument("--frustum-every", type=int, default=25)
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="voxel size for the preview downsample only (0 = off)")
    ap.add_argument("--clean", action="store_true",
                    help="inspect cloud_clean.ply (ground-aligned) instead of the raw cloud")
    ap.add_argument("--no-trajectory", action="store_true",
                    help="omit the camera path; the cleaned cloud is ground-aligned but "
                         "the trajectory is not, so they no longer share a frame")
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument("--tag", default="",
                    help="prefix for the rendered filenames, so a --clean pass does "
                         "not overwrite the raw cloud's renders")
    a = ap.parse_args()

    ply = a.run_dir / ("cloud_clean.ply" if a.clean else "cloud.ply")
    npz = a.run_dir / "trajectory.npz"
    if not ply.exists():
        raise SystemExit(f"missing {ply}")
    if not npz.exists():
        raise SystemExit(f"missing {npz}")

    pcd = o3d.io.read_point_cloud(str(ply))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise SystemExit("cloud is empty")

    extrinsic, intrinsic, cam_centers = load_trajectory(npz)

    # ── Numbers ──────────────────────────────────────────────────────────────
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    centroid = pts.mean(axis=0)

    # Nearest-neighbour spacing tells us the effective resolution, which is what
    # decides a sane hfield cell size in Phase 4.
    sample = pts[np.random.default_rng(0).choice(len(pts), size=min(20000, len(pts)), replace=False)]
    tree = o3d.geometry.KDTreeFlann(pcd)
    nn = []
    for p in sample[:5000]:
        k, _, d2 = tree.search_knn_vector_3d(p, 2)
        if k == 2:
            nn.append(np.sqrt(d2[1]))
    nn = np.asarray(nn)

    seg = np.linalg.norm(np.diff(cam_centers, axis=0), axis=1)

    stats = {
        "n_points": int(len(pts)),
        "n_poses": int(len(cam_centers)),
        "extent_arbitrary": [round(float(v), 3) for v in extent],
        "centroid": [round(float(v), 3) for v in centroid],
        "nn_spacing_median": round(float(np.median(nn)), 5) if len(nn) else None,
        "nn_spacing_p95": round(float(np.percentile(nn, 95)), 5) if len(nn) else None,
        "traj_length_arbitrary": round(float(seg.sum()), 3),
        "traj_step_median": round(float(np.median(seg)), 5) if len(seg) else None,
        "traj_step_max": round(float(seg.max()), 5) if len(seg) else None,
        "has_colors": bool(pcd.has_colors()),
    }

    print("── reconstruction stats (arbitrary scale) ──")
    for k, v in stats.items():
        print(f"  {k:24s} {v}")

    # A camera step much larger than the median is the signature of a pose jump --
    # the drift/tracking-failure mode the expedition needs to detect automatically.
    if len(seg) and seg.max() > 8 * np.median(seg):
        idx = int(seg.argmax())
        print(f"\n  ! possible pose jump at frame {idx}->{idx+1}: "
              f"step {seg.max():.4f} vs median {np.median(seg):.4f}")

    (a.run_dir / "inspect_stats.json").write_text(json.dumps(stats, indent=2))

    # ── Visualise ────────────────────────────────────────────────────────────
    preview = pcd.voxel_down_sample(a.voxel) if a.voxel > 0 else pcd
    scene_extent = float(np.max(extent))
    geoms = [preview]
    if not (a.no_trajectory or a.clean):
        geoms += trajectory_geometries(cam_centers, extrinsic, a.frustum_every, scene_extent)

    if a.interactive:
        # Only works with a real display -- on WSL the legacy visualizer cannot
        # initialise GLEW, so this is the Windows-side path.
        o3d.visualization.draw_geometries(geoms, window_name=f"LingBot-Map: {a.run_dir.name}")
        return 0

    # Offscreen: four orbit views so the cloud can be judged from a still image.
    # Must be OffscreenRenderer, not Visualizer(visible=False): the latter still
    # needs GLFW/GLEW and fails under WSL, while this path uses EGL headless.
    rnd = o3d.visualization.rendering.OffscreenRenderer(1600, 1200)
    rnd.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

    mat_pts = o3d.visualization.rendering.MaterialRecord()
    mat_pts.shader = "defaultUnlit"
    mat_pts.point_size = a.point_size
    rnd.scene.add_geometry("cloud", preview, mat_pts)

    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 3.0
    for i, g in enumerate(geoms[1:], start=1):
        if isinstance(g, o3d.geometry.LineSet):
            rnd.scene.add_geometry(f"traj{i}", g, mat_line)
        else:
            m = o3d.visualization.rendering.MaterialRecord()
            m.shader = "defaultUnlit"
            rnd.scene.add_geometry(f"frustum{i}", g, m)

    # At a 60 deg vertical FOV an object of size S fills the frame from ~0.87*S
    # away, so 1.4*S renders it postage-stamp sized with most of the image empty.
    r = scene_extent * 1.0
    # After cleanup the cloud is ground-aligned, so +Z is genuinely up and the
    # orbit should be around Z. The raw cloud has no such guarantee.
    if a.clean:
        views = [
            ("top",   np.array([0.0, -1e-3, r]),              np.array([0.0, 1.0, 0.0])),
            ("front", np.array([0.0, -r, 0.45 * r]),          np.array([0.0, 0.0, 1.0])),
            ("side",  np.array([r, 0.0, 0.45 * r]),           np.array([0.0, 0.0, 1.0])),
            ("obliq", np.array([0.75 * r, -0.75 * r, 0.5 * r]), np.array([0.0, 0.0, 1.0])),
        ]
    else:
        views = [
            ("top",   np.array([0.0, -1e-3, r]),        np.array([0.0, -1.0, 0.0])),
            ("front", np.array([0.0, -0.35 * r, -r]),   np.array([0.0, -1.0, 0.0])),
            ("side",  np.array([r, -0.35 * r, 0.0]),    np.array([0.0, -1.0, 0.0])),
            ("obliq", np.array([0.7 * r, -0.5 * r, -0.7 * r]), np.array([0.0, -1.0, 0.0])),
        ]

    out_paths = []
    for name, eye_off, up in views:
        rnd.setup_camera(60.0, centroid, centroid + eye_off, up)
        img = rnd.render_to_image()
        p = a.run_dir / f"view_{a.tag}{name}.png"
        o3d.io.write_image(str(p), img)
        out_paths.append(p)

    print("\nrendered:")
    for p in out_paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
