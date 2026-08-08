"""Build an Isaac-ready terrain strip from a real mountain trail DEM.

Takes the trail centerline (fetch_eiger_trail.py) and the swissALTI3D 0.5 m
tiles, and produces a *straightened* corridor: the strip's x axis is arc length
along the trail, y is signed cross-track offset, z is the real DEM elevation
sampled at that physical point. Real elevation profile, cross-slope, steps and
rock texture are preserved; the trail's map-view curvature is not (a straight
strip makes env origins and spawning trivial). At switchbacks tighter than the
half-width, the outer samples sweep overlapping real terrain — the centerline
is pre-smoothed (--smooth) to bound that.

Outputs, under --out (data/, gitignored):
    heightfield.npz     H[n_along, n_across] float32 (z, metres, min=0),
                        plus arc/cross coordinate vectors and the centerline
    origins.npz         env spawn origins [K,3] along the centerline
    trail_terrain.obj   the strip as a triangle mesh (x=arc, y=cross, z=up)
    meta.json           provenance + stats (the record terrain-validator reads)

Run in the isaac venv (numpy + tifffile + imagecodecs):
    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims/isaac/terrain/build_trail_terrain.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[3]


def load_mosaic(dem_dir: Path) -> tuple[np.ndarray, float, float]:
    """Paste all 1 km tiles into one float32 grid. Returns (grid, E_min, N_max)."""
    tiles = sorted(dem_dir.glob("*.tif"))
    if not tiles:
        raise FileNotFoundError(f"no GeoTIFFs in {dem_dir} — run fetch_eiger_trail.py first")
    entries = []
    for path in tiles:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            sx, sy, _ = page.tags["ModelPixelScaleTag"].value
            _, _, _, e0, n0, _ = page.tags["ModelTiepointTag"].value
            if not (abs(sx - 0.5) < 1e-6 and abs(sy - 0.5) < 1e-6):
                raise ValueError(f"{path.name}: pixel scale {sx}x{sy}, expected 0.5 m")
            entries.append((e0, n0, page.asarray().astype(np.float32)))
    e_min = min(e for e, _, _ in entries)
    n_max = max(n for _, n, _ in entries)
    e_max = max(e + a.shape[1] * 0.5 for e, _, a in entries)
    n_min = min(n - a.shape[0] * 0.5 for _, n, a in entries)
    grid = np.full((round((n_max - n_min) / 0.5), round((e_max - e_min) / 0.5)),
                   np.nan, dtype=np.float32)
    for e0, n0, arr in entries:
        r = round((n_max - n0) / 0.5)
        c = round((e0 - e_min) / 0.5)
        grid[r:r + arr.shape[0], c:c + arr.shape[1]] = arr
    return grid, e_min, n_max


def bilinear(grid: np.ndarray, e_min: float, n_max: float,
             e: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Sample the mosaic at LV95 points. Pixel centers are at half-cell offsets."""
    fc = (e - e_min) / 0.5 - 0.5
    fr = (n_max - n) / 0.5 - 0.5
    c0 = np.floor(fc).astype(np.int64)
    r0 = np.floor(fr).astype(np.int64)
    if (r0 < 0).any() or (c0 < 0).any() or (r0 + 1 >= grid.shape[0]).any() or (c0 + 1 >= grid.shape[1]).any():
        raise ValueError("sample point outside the DEM mosaic — widen --corridor in fetch")
    wc = fc - c0
    wr = fr - r0
    h = (grid[r0, c0] * (1 - wr) * (1 - wc)
         + grid[r0, c0 + 1] * (1 - wr) * wc
         + grid[r0 + 1, c0] * wr * (1 - wc)
         + grid[r0 + 1, c0 + 1] * wr * wc)
    return h


def gaussian_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    """1-D gaussian smoothing with edge-clamped padding (no scipy in this venv)."""
    radius = max(1, int(4 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_samples) ** 2)
    k /= k.sum()
    padded = np.concatenate([np.full(radius, x[0]), x, np.full(radius, x[-1])])
    return np.convolve(padded, k, mode="valid")


def resample_by_arc(e: np.ndarray, n: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seg = np.hypot(np.diff(e), np.diff(n))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, arc[-1], ds)
    return np.interp(s, arc, e), np.interp(s, arc, n), s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trail", type=Path, default=REPO / "data/eiger_trail/trail_lv95.json")
    ap.add_argument("--dem-dir", type=Path, default=REPO / "data/swissalti3d")
    ap.add_argument("--out", type=Path, default=REPO / "data/eiger_trail/terrain")
    ap.add_argument("--width", type=float, default=24.0, help="strip width (m)")
    ap.add_argument("--res", type=float, default=0.5, help="grid resolution (m)")
    ap.add_argument("--smooth", type=float, default=12.0,
                    help="centerline gaussian sigma (m); bounds switchback fold-over")
    ap.add_argument("--start", type=float, default=0.0, help="start (m along trail)")
    ap.add_argument("--length", type=float, default=None, help="length (m); default full trail")
    ap.add_argument("--origin-spacing", type=float, default=2.0,
                    help="spacing (m) of env origins along the centerline")
    ap.add_argument("--origin-margin", type=float, default=10.0,
                    help="keep origins this far (m) from the strip ends")
    ap.add_argument("--origin-max-relief", type=float, default=1.0,
                    help="drop spawn origins whose ±1 m patch has more height "
                         "relief (m) than this — the OSM centerline can sit a few "
                         "metres off the real trail bench, onto side-slope or cliff")
    args = ap.parse_args()

    trail = json.loads(args.trail.read_text())
    grid, e_min, n_max = load_mosaic(args.dem_dir)
    print(f"mosaic {grid.shape[0]}x{grid.shape[1]} px, {np.isnan(grid).mean():.0%} empty padding")

    # --- centerline: dense resample -> smooth -> exact arc-length resample ---
    e_raw = np.asarray(trail["e"])
    n_raw = np.asarray(trail["n"])
    e_d, n_d, _ = resample_by_arc(e_raw, n_raw, 0.25)
    e_s = gaussian_smooth(e_d, args.smooth / 0.25)
    n_s = gaussian_smooth(n_d, args.smooth / 0.25)
    e_c, n_c, s = resample_by_arc(e_s, n_s, args.res)

    if args.start > 0 or args.length is not None:
        end = s[-1] if args.length is None else args.start + args.length
        keep = (s >= args.start) & (s <= end)
        e_c, n_c, s = e_c[keep], n_c[keep], s[keep] - args.start

    # tangents and left normals
    te = np.gradient(e_c)
    tn = np.gradient(n_c)
    norm = np.hypot(te, tn)
    te, tn = te / norm, tn / norm
    ne, nn = -tn, te

    y = np.arange(-args.width / 2, args.width / 2 + args.res / 2, args.res)
    ee = e_c[:, None] + ne[:, None] * y[None, :]
    nnn = n_c[:, None] + nn[:, None] * y[None, :]
    H_abs = bilinear(grid, e_min, n_max, ee, nnn)
    if np.isnan(H_abs).any():
        raise ValueError(f"{np.isnan(H_abs).sum()} NaN height samples — DEM coverage hole")

    z_offset = float(H_abs.min())
    H = (H_abs - z_offset).astype(np.float32)
    n_along, n_across = H.shape
    print(f"strip {s[-1]:.0f} m x {args.width:.0f} m -> {n_along}x{n_across} cells")

    # slope stats (per-cell gradient magnitude)
    gx = np.diff(H, axis=0) / args.res
    gy = np.diff(H, axis=1) / args.res
    center = H[:, n_across // 2]
    stats = {
        "elevation_min_abs_m": z_offset,
        "elevation_max_abs_m": float(H_abs.max()),
        "total_drop_m": float(center[0] - center[-1]),
        "mean_grade_pct": float((center[0] - center[-1]) / s[-1] * 100),
        "slope_p50_pct": float(np.percentile(np.abs(gx), 50) * 100),
        "slope_p95_pct": float(np.percentile(np.abs(gx), 95) * 100),
        "slope_max_pct": float(max(np.abs(gx).max(), np.abs(gy).max()) * 100),
    }
    print("  " + "  ".join(f"{k}={v:.1f}" for k, v in stats.items()))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "heightfield.npz",
        height=H, arc=s.astype(np.float32), cross=y.astype(np.float32),
        centerline_e=e_c.astype(np.float64), centerline_n=n_c.astype(np.float64),
        z_offset=np.float64(z_offset),
    )

    # --- env origins: along the centerline, on the surface ---
    step = max(1, round(args.origin_spacing / args.res))
    margin = round(args.origin_margin / args.res)
    idx = np.arange(margin, n_along - margin, step)
    pad = round(1.0 / args.res)  # ±1 m patch
    j0 = n_across // 2
    relief = np.array([np.ptp(H[i - pad:i + pad + 1, j0 - pad:j0 + pad + 1]) for i in idx])
    keep = relief <= args.origin_max_relief
    idx = idx[keep]
    origins = np.stack([s[idx], np.zeros(len(idx)), center[idx]], axis=1).astype(np.float32)
    if len(origins) < 50:
        raise ValueError(f"only {len(origins)} walkable origins — threshold too strict or terrain wrong")
    np.savez(args.out / "origins.npz", origins=origins)
    print(f"{len(origins)} walkable env origins (of {len(keep)} candidates every "
          f"{args.origin_spacing} m; relief <= {args.origin_max_relief} m over a 2 m patch)")

    # --- OBJ (x=arc, y=cross, z=up; CCW faces seen from +z) ---
    obj_path = args.out / "trail_terrain.obj"
    xs = np.repeat(s, n_across)
    ys = np.tile(y, n_along)
    zs = H.ravel()
    with open(obj_path, "w", newline="\n") as f:
        f.write("# Eiger Trail strip — built by sims/isaac/terrain/build_trail_terrain.py\n")
        for i0 in range(0, len(xs), 200_000):
            i1 = min(i0 + 200_000, len(xs))
            f.write("".join(f"v {a:.3f} {b:.3f} {c:.3f}\n"
                            for a, b, c in zip(xs[i0:i1], ys[i0:i1], zs[i0:i1])))
        ii, jj = np.meshgrid(np.arange(n_along - 1), np.arange(n_across - 1), indexing="ij")
        v1 = (ii * n_across + jj + 1).ravel()
        v2 = v1 + n_across
        v3 = v2 + 1
        v4 = v1 + 1
        faces = np.empty((len(v1) * 2, 3), dtype=np.int64)
        faces[0::2] = np.stack([v1, v2, v3], axis=1)
        faces[1::2] = np.stack([v1, v3, v4], axis=1)
        for i0 in range(0, len(faces), 400_000):
            i1 = min(i0 + 400_000, len(faces))
            f.write("".join(f"f {a} {b} {c}\n" for a, b, c in faces[i0:i1]))
    print(f"OBJ: {obj_path} ({obj_path.stat().st_size/1e6:.0f} MB, "
          f"{len(xs)} verts, {len(faces)} tris)")

    meta = {
        "built": date.today().isoformat(),
        "source_trail": trail["source"],
        "source_dem": "swisstopo swissALTI3D 0.5 m (data.geo.admin.ch), tiles in data/swissalti3d/",
        "frame": "x = arc length along trail (m, 0 at Eigergletscher end), "
                 "y = cross-track (m, left positive), z = up (m, strip min = 0)",
        "straightened": True,
        "width_m": args.width,
        "resolution_m": args.res,
        "length_m": float(s[-1]),
        "centerline_smooth_sigma_m": args.smooth,
        "start_offset_m": args.start,
        "micro_noise": "none — raw DEM heights; terrain/domain randomization is Phase 5",
        "z_offset_abs_m": z_offset,
        "origin_spacing_m": args.origin_spacing,
        "origin_max_relief_m": args.origin_max_relief,
        "num_origin_candidates": int(len(keep)),
        "num_origins": int(len(origins)),
        "stats": stats,
        "vertices": int(len(xs)),
        "triangles": int(len(faces)),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"meta: {args.out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
