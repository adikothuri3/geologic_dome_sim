"""Build an Isaac-ready terrain patch of the Everest summit pyramid from the HMA 8 m DEM.

Crops an axis-aligned window around the summit (27.988 N, 86.925 E) straight out
of HMA_DEM8m_MOS tile-677 (fetch_everest_dem.py). Unlike the Eiger strip there
is no centerline and no resampling: the patch axes are the DEM grid axes, so the
output heightfield *is* the cropped slice — x = east, y = north, both centered
on the patch middle, z = elevation with the patch minimum at 0.

Voids (optical-stereo holes on steep faces) are filled by iterative 8-neighbor
mean dilation, capped so anything wider than ~100 m aborts the build instead of
being fabricated — the micro_noise: none honesty rule extends to holes.

Outputs, under --out (data/, gitignored):
    heightfield.npz       H[n_y, n_x] float32 (z, metres, min=0) + x/y vectors
    origins.npz           env spawn origins [K,3] on cells with slope <= threshold
    everest_terrain.obj   the patch as a triangle mesh (x=east, y=north, z=up)
    meta.json             provenance + stats (the record terrain-validator reads)

Run in the isaac venv (numpy + tifffile + imagecodecs):
    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims/isaac/terrain/build_everest_terrain.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[3]

# The HMA mosaics use a custom Albers Equal Area on WGS84 ("HMA Albers"):
# standard parallels 25/47, origin 36 N 85 E, FE=FN=0 — read from the tile's
# geokeys (ProjCoordTransGeoKey=11) and asserted in read_geotiff().
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)
_E = math.sqrt(_E2)
_LAT1, _LAT2 = math.radians(25.0), math.radians(47.0)
_LAT0, _LON0 = math.radians(36.0), math.radians(85.0)


def _q(phi: float) -> float:
    s = math.sin(phi)
    return (1 - _E2) * (s / (1 - _E2 * s * s)
                        - (1 / (2 * _E)) * math.log((1 - _E * s) / (1 + _E * s)))


def _m(phi: float) -> float:
    return math.cos(phi) / math.sqrt(1 - _E2 * math.sin(phi) ** 2)


_N = (_m(_LAT1) ** 2 - _m(_LAT2) ** 2) / (_q(_LAT2) - _q(_LAT1))
_C = _m(_LAT1) ** 2 + _N * _q(_LAT1)
_RHO0 = _A * math.sqrt(_C - _N * _q(_LAT0)) / _N


def wgs84_to_hma_albers(lat: float, lon: float) -> tuple[float, float]:
    """Forward ellipsoidal Albers Equal Area (Snyder 1987 eq. 14-1..14-4)."""
    rho = _A * math.sqrt(_C - _N * _q(math.radians(lat))) / _N
    theta = _N * (math.radians(lon) - _LON0)
    return rho * math.sin(theta), _RHO0 - rho * math.cos(theta)


def hma_albers_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """Inverse ellipsoidal Albers (Snyder 1987 eq. 14-8..14-11, iterative)."""
    rho = math.hypot(x, _RHO0 - y)
    theta = math.atan2(x, _RHO0 - y)
    q = (_C - (rho * _N / _A) ** 2) / _N
    phi = math.asin(max(-1.0, min(1.0, q / 2)))
    for _ in range(8):
        s = math.sin(phi)
        phi += ((1 - _E2 * s * s) ** 2 / (2 * math.cos(phi))) * (
            q / (1 - _E2) - s / (1 - _E2 * s * s)
            + (1 / (2 * _E)) * math.log((1 - _E * s) / (1 + _E * s))
        )
    return math.degrees(phi), math.degrees(theta / _N + _LON0)


def read_geotiff(path: Path) -> tuple[np.ndarray, float, float, float, float]:
    """Returns (grid float32 with NaN voids, pixel_m, E0, N0, nodata)."""
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        sx, sy, _ = page.tags["ModelPixelScaleTag"].value
        _, _, _, e0, n0, _ = page.tags["ModelTiepointTag"].value
        if abs(sx - sy) > 1e-6:
            raise ValueError(f"anisotropic pixels {sx}x{sy}")
        keys = page.tags["GeoKeyDirectoryTag"].value

        def geokey(kid):
            return next((keys[i + 3] for i in range(0, len(keys) - 3, 4)
                         if keys[i] == kid), None)

        if geokey(3075) != 11:  # ProjCoordTransGeoKey: 11 = Albers Equal Area
            raise ValueError(f"projection method geokey {geokey(3075)}, expected "
                             "11 (Albers) — the hand-rolled projection would be wrong")
        doubles = page.tags["GeoDoubleParamsTag"].value
        want = {3078: 25.0, 3079: 47.0, 3081: 36.0, 3080: 85.0, 3082: 0.0, 3083: 0.0}
        got = {k: doubles[geokey(k)] for k in want}
        if any(abs(got[k] - v) > 1e-9 for k, v in want.items()):
            raise ValueError(f"Albers parameters {got} differ from the HMA constants "
                             f"{want} baked into this script")
        nd_tag = page.tags.get(42113)  # GDAL_NODATA
        nodata = float(nd_tag.value) if nd_tag is not None else -9999.0
        grid = page.asarray().astype(np.float32)
    grid[(grid == nodata) | (grid < -500.0)] = np.nan
    return grid, float(sx), float(e0), float(n0), nodata


def fill_voids(h: np.ndarray, max_iters: int) -> tuple[np.ndarray, int]:
    """Iterative 8-neighbor mean dilation fill (numpy-only, no scipy).

    Each pass fills NaN cells that touch >=1 valid neighbor, growing the fill
    front one cell per pass. Returns (filled, passes_used); NaNs may survive if
    a hole is wider than 2*max_iters cells — the caller decides that's fatal.
    """
    h = h.copy()
    passes = 0
    for _ in range(max_iters):
        nan = np.isnan(h)
        if not nan.any():
            break
        padded = np.pad(h, 1, constant_values=np.nan)
        stack = np.stack([padded[1 + di:padded.shape[0] - 1 + di, 1 + dj:padded.shape[1] - 1 + dj]
                          for di in (-1, 0, 1) for dj in (-1, 0, 1)
                          if not (di == 0 and dj == 0)])
        cnt = np.sum(~np.isnan(stack), axis=0)
        mean = np.nansum(np.nan_to_num(stack), axis=0) / np.maximum(cnt, 1)
        fillable = nan & (cnt > 0)
        if not fillable.any():
            break
        h[fillable] = mean[fillable]
        passes += 1
    return h, passes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dem", type=Path,
                    default=REPO / "data/hma_dem/HMA_DEM8m_MOS_20170716_tile-677.tif")
    ap.add_argument("--out", type=Path, default=REPO / "data/everest/terrain")
    ap.add_argument("--lat", type=float, default=27.988, help="patch center latitude")
    ap.add_argument("--lon", type=float, default=86.925, help="patch center longitude")
    ap.add_argument("--size", type=float, default=2000.0, help="patch side length (m)")
    ap.add_argument("--max-void-frac", type=float, default=0.15,
                    help="abort if more of the patch than this is nodata")
    ap.add_argument("--fill-max-iters", type=int, default=6,
                    help="void-fill passes (1 cell = 8 m reach each)")
    ap.add_argument("--origin-spacing", type=float, default=16.0,
                    help="candidate origin spacing (m); snapped to whole cells")
    ap.add_argument("--origin-max-slope", type=float, default=15.0,
                    help="keep origins on cells with slope (deg) at most this")
    ap.add_argument("--origin-margin", type=float, default=24.0,
                    help="keep origins this far (m) from the patch edges")
    ap.add_argument("--min-origins", type=int, default=40,
                    help="abort below this many surviving origins")
    args = ap.parse_args()

    grid, px, e0, n0, nodata = read_geotiff(args.dem)
    print(f"tile {grid.shape[0]}x{grid.shape[1]} px at {px:.0f} m, "
          f"tiepoint E={e0:.0f} N={n0:.0f}, nodata={nodata:g}, "
          f"{np.isnan(grid).mean():.1%} void overall")

    # projection self-check: round-trip the tiepoint corner into the granule bbox
    lat_c, lon_c = hma_albers_to_wgs84(e0, n0)
    e_rt, n_rt = wgs84_to_hma_albers(lat_c, lon_c)
    if math.hypot(e_rt - e0, n_rt - n0) > 0.01:
        raise AssertionError("Albers round-trip error > 1 cm — projection code broken")
    if not (27.0 < lat_c < 29.0 and 86.0 < lon_c < 88.0):
        raise AssertionError(f"tiepoint corner maps to {lat_c:.3f}N {lon_c:.3f}E — "
                             "not the Everest tile or wrong projection")

    # --- cell-snapped crop window around the requested center ---
    e_c, n_c = wgs84_to_hma_albers(args.lat, args.lon)
    col_c = int((e_c - e0) / px)
    row_c = int((n0 - n_c) / px)
    half = int(round(args.size / 2 / px))
    fill_margin = args.fill_max_iters  # cells; lets edge fills see real neighbors
    r0, r1 = row_c - half - fill_margin, row_c + half + fill_margin
    c0, c1 = col_c - half - fill_margin, col_c + half + fill_margin
    if r0 < 0 or c0 < 0 or r1 > grid.shape[0] or c1 > grid.shape[1]:
        raise ValueError("patch (plus fill margin) falls outside tile-677 — "
                         "move --lat/--lon or shrink --size")
    patch = grid[r0:r1, c0:c1].copy()
    del grid

    void_frac = float(np.isnan(patch[fill_margin:-fill_margin,
                                     fill_margin:-fill_margin]).mean())
    if void_frac > args.max_void_frac:
        raise ValueError(f"void fraction {void_frac:.1%} exceeds "
                         f"--max-void-frac {args.max_void_frac:.0%} — a patch this "
                         "holey would be mostly interpolation, not Everest")
    if void_frac > 0.05:
        print(f"WARNING: void fraction {void_frac:.1%} (>5%) — check the fill "
              "stats in meta.json before trusting this patch", file=sys.stderr)

    n_voids = int(np.isnan(patch).sum())
    patch, passes = fill_voids(patch, args.fill_max_iters)
    patch = patch[fill_margin:-fill_margin, fill_margin:-fill_margin]
    if np.isnan(patch).any():
        raise ValueError(f"{int(np.isnan(patch).sum())} void cells survive "
                         f"{args.fill_max_iters} fill passes — a hole wider than "
                         f"{2 * args.fill_max_iters * px:.0f} m; not filling that")

    # rows come north->south; flip so +y = north (right-handed, z-up)
    H_abs = patch[::-1].astype(np.float32)
    n_y, n_x = H_abs.shape
    if float(H_abs.max()) < 8000.0:
        raise ValueError(f"patch max elevation {H_abs.max():.0f} m — this is not "
                         "the summit pyramid; check --lat/--lon")

    z_offset = float(H_abs.min())
    H = H_abs - z_offset
    x = (np.arange(n_x) - (n_x - 1) / 2) * px  # 0 at patch center
    y = (np.arange(n_y) - (n_y - 1) / 2) * px
    print(f"patch {n_x * px:.0f} m x {n_y * px:.0f} m -> {n_y}x{n_x} cells, "
          f"elevation {z_offset:.0f}-{H_abs.max():.0f} m a.s.l. "
          f"(relief {H.max():.0f} m)")

    # slope per cell (degrees) from the 8 m grid
    gy, gx = np.gradient(H, px)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    stats = {
        "elevation_min_abs_m": z_offset,
        "elevation_max_abs_m": float(H_abs.max()),
        "relief_m": float(H.max()),
        "slope_p50_deg": float(np.percentile(slope_deg, 50)),
        "slope_p95_deg": float(np.percentile(slope_deg, 95)),
        "slope_max_deg": float(slope_deg.max()),
    }
    print("  " + "  ".join(f"{k}={v:.1f}" for k, v in stats.items()))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "heightfield.npz",
        height=H, x=x.astype(np.float32), y=y.astype(np.float32),
        z_offset=np.float64(z_offset),
    )

    # --- env origins: grid-sampled cells flat enough to stand on ---
    step = max(1, round(args.origin_spacing / px))
    margin = max(1, round(args.origin_margin / px))
    ii, jj = np.meshgrid(np.arange(margin, n_y - margin, step),
                         np.arange(margin, n_x - margin, step), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    keep = slope_deg[ii, jj] <= args.origin_max_slope
    n_candidates = len(keep)
    ii, jj = ii[keep], jj[keep]
    origins = np.stack([x[jj], y[ii], H[ii, jj]], axis=1).astype(np.float32)
    if len(origins) < args.min_origins:
        raise ValueError(
            f"only {len(origins)} origins with slope <= {args.origin_max_slope} deg "
            f"(of {n_candidates} candidates) — below --min-origins {args.min_origins}. "
            "Either relax --origin-max-slope (20 is still standable) or extend the "
            "patch toward the South Col: --lat 27.980 --lon 86.928 --size 3600"
        )
    np.savez(args.out / "origins.npz", origins=origins)
    print(f"{len(origins)} spawn origins (of {n_candidates} candidates every "
          f"{step * px:.0f} m; slope <= {args.origin_max_slope} deg)")

    # --- OBJ (x=east, y=north, z=up; CCW faces seen from +z) ---
    obj_path = args.out / "everest_terrain.obj"
    xs = np.tile(x, n_y)
    ys = np.repeat(y, n_x)
    zs = H.ravel()
    with open(obj_path, "w", newline="\n") as f:
        f.write("# Everest summit patch — built by sims/isaac/terrain/build_everest_terrain.py\n")
        for i0 in range(0, len(xs), 200_000):
            i1 = min(i0 + 200_000, len(xs))
            f.write("".join(f"v {a:.3f} {b:.3f} {c:.3f}\n"
                            for a, b, c in zip(xs[i0:i1], ys[i0:i1], zs[i0:i1])))
        ii2, jj2 = np.meshgrid(np.arange(n_y - 1), np.arange(n_x - 1), indexing="ij")
        v1 = (ii2 * n_x + jj2 + 1).ravel()
        v2 = v1 + 1
        v3 = v2 + n_x
        v4 = v1 + n_x
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
        "source_dem": "NASA/NSIDC HMA_DEM8m_MOS v1 tile-677 (WorldView/GeoEye stereo, "
                      "8 m, ~1.9 m RMSE at Everest), data/hma_dem/",
        "frame": "x = grid east, y = grid north (m, 0 at patch center; Albers grid, "
                 "~1.2 deg from true north here), z = up (m, patch min = 0)",
        "crs": "HMA Albers Equal Area on WGS84 (std parallels 25/47, origin 36N 85E), "
               "hand-rolled, verified by round-trip",
        "center_wgs84": [args.lat, args.lon],
        "center_albers": [e_c, n_c],
        "size_m": args.size,
        "resolution_m": px,
        "micro_noise": "none — raw DEM heights; voids filled by 8-neighbor mean "
                       "dilation only (no synthesized detail)",
        "void_fraction_initial": void_frac,
        "void_cells_filled": n_voids,
        "fill_max_pass_used": passes,
        "z_offset_abs_m": z_offset,
        "origin_spacing_m": step * px,
        "origin_max_slope_deg": args.origin_max_slope,
        "num_origin_candidates": n_candidates,
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
