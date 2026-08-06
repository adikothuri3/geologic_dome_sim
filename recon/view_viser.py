"""Serve an exported reconstruction in viser: point cloud + camera trajectory.

Phase 3 asks for the reconstruction to be inspected in viser with the camera
trajectory overlaid. Upstream's viewer only works on live predictions still in
memory, which means re-running inference just to look at a result. This reads the
exported `cloud.ply` + `trajectory.npz` instead, so any past run can be reopened
in seconds with no GPU.

Runs inside WSL; the browser on Windows reaches it at http://localhost:<port>
via WSL2's localhost forwarding.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import viser


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="folder with cloud.ply + trajectory.npz")
    ap.add_argument("--clean", action="store_true",
                    help="serve cloud_clean.ply; it is ground-aligned, so the trajectory "
                         "(still in the raw frame) is omitted")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-points", type=int, default=1_500_000,
                    help="random subsample cap; the browser, not the server, is the bottleneck")
    ap.add_argument("--point-size", type=float, default=0.004)
    ap.add_argument("--frustum-every", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=0,
                    help="auto-exit after N seconds (0 = serve until interrupted)")
    a = ap.parse_args()

    ply = a.run_dir / ("cloud_clean.ply" if a.clean else "cloud.ply")
    npz = a.run_dir / "trajectory.npz"
    for p in (ply, npz):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    pcd = o3d.io.read_point_cloud(str(ply))
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    if len(pts) == 0:
        raise SystemExit("cloud is empty")

    if len(pts) > a.max_points:
        idx = np.random.default_rng(0).choice(len(pts), a.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
        print(f"subsampled to {a.max_points:,} points for the browser")

    d = np.load(npz, allow_pickle=True)
    extrinsic, cam_centers = d["extrinsic"], d["cam_centers"]

    server = viser.ViserServer(port=a.port)
    server.scene.add_point_cloud(
        "/cloud", points=pts, colors=cols, point_size=a.point_size
    )

    # Trajectory as a time-coloured polyline: blue at the start, red at the end.
    if len(cam_centers) > 1 and not a.clean:
        segs = np.stack([cam_centers[:-1], cam_centers[1:]], axis=1)
        t = np.linspace(0, 1, len(segs))[:, None]
        seg_cols = np.repeat(
            np.hstack([t, np.zeros_like(t), 1.0 - t])[:, None, :], 2, axis=1
        )
        server.scene.add_line_segments(
            "/trajectory", points=segs, colors=seg_cols, line_width=3.0
        )

    for i in range(0, 0 if a.clean else len(extrinsic), max(1, a.frustum_every)):
        c2w = np.eye(4)
        c2w[:3, :4] = extrinsic[i]
        R = c2w[:3, :3]
        wxyz = _mat_to_wxyz(R)
        server.scene.add_camera_frustum(
            f"/cams/{i:05d}", fov=1.0, aspect=518 / 294, scale=0.03,
            wxyz=wxyz, position=c2w[:3, 3], color=(0.2, 0.9, 0.3),
        )

    print(f"\nviser serving {len(pts):,} points and {len(cam_centers)} poses")
    print(f"open http://localhost:{a.port} in a browser on Windows")

    if a.seconds:
        time.sleep(a.seconds)
        print("auto-exit")
        return 0
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def _mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z) quaternion, viser's convention."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


if __name__ == "__main__":
    raise SystemExit(main())
