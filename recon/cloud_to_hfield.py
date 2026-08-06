"""Phase 4, path 1: a cleaned metric point cloud -> a MuJoCo heightfield.

Implements `.claude/skills/mjcf-terrain`'s heightfield path: grid XY at 5-10 cm,
robust max-z per cell, fill holes, normalize to [0, 1], emit an `<hfield>` whose
`size` preserves real metres.

Input must be **scale-calibrated and ground-aligned** -- `recon/calibrate_scale.py`
then `recon/clean_cloud.py --scale auto`. The skill refuses terrain built from
arbitrary units and so does this script.

Two decisions worth knowing about, both driven by what reconstructions actually
look like rather than by the skill text:

  * **Robust max, not raw max.** A single surviving flying pixel above a cell
    becomes a spike the policy can trip on. The 95th percentile of the cell's
    points ignores it while still tracking real steps and kerbs.

  * **Unobserved cells are filled from the nearest observed cell, not
    interpolated across the gap.** A walkthrough only sees a corridor-shaped
    slice of its bounding box; interpolating across a 20 m unseen region invents
    a smooth ramp that never existed. Nearest-fill at least extends the last
    real measurement, and `--crop` trims to the observed footprint so the
    invented fraction stays small. The observation mask is saved either way --
    never let invented terrain masquerade as measured.

The heightfield is written as a `.npy` (normalized [0, 1], `grid[iy, ix]`, row 0
at the -y edge) to be loaded straight into `model.hfield_data`, matching the
convention `terrain/make_hfield.py` established and `terrain/drop_test.py`
validates empirically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage


def rasterize(pts: np.ndarray, cell: float, pct: float, min_pts: int):
    """Robust max-z per XY cell. Returns (height grid, observed mask, origin)."""
    xy_min = pts[:, :2].min(0)
    ix = np.floor((pts[:, 0] - xy_min[0]) / cell).astype(np.int64)
    iy = np.floor((pts[:, 1] - xy_min[1]) / cell).astype(np.int64)
    ncol, nrow = int(ix.max()) + 1, int(iy.max()) + 1
    flat = iy * ncol + ix

    # Sort by cell then by z, so each cell's points are contiguous and ordered;
    # the percentile is then a direct index into each run. Beats grouping 5M
    # points with a Python loop.
    order = np.lexsort((pts[:, 2], flat))
    fs, zs = flat[order], pts[order, 2]
    starts = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]])
    counts = np.diff(np.r_[starts, len(fs)])
    cells = fs[starts]

    take = starts + np.minimum((counts * pct / 100.0).astype(np.int64), counts - 1)
    z = zs[take]

    grid = np.zeros(nrow * ncol, np.float64)
    seen = np.zeros(nrow * ncol, bool)
    keep = counts >= min_pts
    grid[cells[keep]] = z[keep]
    seen[cells[keep]] = True
    return grid.reshape(nrow, ncol), seen.reshape(nrow, ncol), xy_min


def fill_holes(grid: np.ndarray, seen: np.ndarray, mode: str, floor: float) -> np.ndarray:
    """Fill unobserved cells. Never leave NaN/zero pits (mjcf-terrain step 3).

    ``nearest`` extends the closest real measurement -- right for small gaps
    inside a surface the camera did see.

    ``floor`` sets unobserved cells to ground level. Right when the observed
    fraction is small, which is the normal case for a walkthrough: the camera
    saw a corridor-shaped sliver of its own bounding box. Nearest-fill there
    would smear metre-tall walls across regions nobody looked at and fence the
    robot in with invented geometry. Flat-unknown is both safer to walk and
    honest about what was measured.
    """
    if seen.all():
        return grid
    if mode == "floor":
        out = grid.copy()
        out[~seen] = floor
        return out
    _, idx = ndimage.distance_transform_edt(~seen, return_indices=True)
    return grid[tuple(idx)]


def crop_to_observed(grid, seen, xy_min, cell, margin_cells: int):
    rows, cols = np.where(seen)
    r0, r1 = max(rows.min() - margin_cells, 0), min(rows.max() + margin_cells + 1, grid.shape[0])
    c0, c1 = max(cols.min() - margin_cells, 0), min(cols.max() + margin_cells + 1, grid.shape[1])
    return (grid[r0:r1, c0:c1], seen[r0:r1, c0:c1],
            xy_min + np.array([c0 * cell, r0 * cell]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cloud", default="cloud_clean.ply")
    ap.add_argument("--name", default=None, help="asset basename (default: run dir name)")
    ap.add_argument("--cell", type=float, default=0.05,
                    help="grid cell in metres; the skill's range is 0.05-0.10")
    ap.add_argument("--surface", choices=["top", "ground"], default="top",
                    help="'top' = robust max-z, the skill's default and correct for "
                         "outdoor trail terrain where the upper surface is what you "
                         "walk on. 'ground' takes a low percentile instead, i.e. the "
                         "floor beneath furniture -- use it indoors, where max-z turns "
                         "every desk and partition into a metre-tall column and the "
                         "heightfield becomes a canyon rather than a walkable surface.")
    ap.add_argument("--percentile", type=float, default=None,
                    help="override the per-cell percentile (default 95 for top, 10 for ground)")
    ap.add_argument("--min-points", type=int, default=3,
                    help="cells with fewer points count as unobserved")
    ap.add_argument("--max-height", type=float, default=2.5,
                    help="clip heights this far above the floor; walls and facades "
                         "would otherwise become metre-tall columns of no use to a walker")
    ap.add_argument("--ground-tol", type=float, default=0.4,
                    help="ground mode only: cells this far above the floor are treated "
                         "as structure, not terrain, and refilled")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="gaussian sigma in cells applied after hole filling; removes "
                         "reconstruction depth noise that a keyframe-posed robot cannot "
                         "stand on. Report it -- it changes the terrain the policy sees")
    ap.add_argument("--median", type=int, default=None,
                    help="median-filter window in cells, to kill one-cell spikes "
                         "(default 3 in ground mode, off in top mode)")
    ap.add_argument("--fill", choices=["auto", "nearest", "floor"], default="auto",
                    help="how to fill unobserved cells; auto picks floor below 50%% observed")
    ap.add_argument("--crop", action="store_true",
                    help="trim to the observed footprint (recommended)")
    ap.add_argument("--margin", type=int, default=4, help="cells of margin when cropping")
    ap.add_argument("--base-z", type=float, default=0.5,
                    help="hfield thickness below z=0; must be non-zero")
    ap.add_argument("--force", action="store_true",
                    help="build even if the cloud is not scale-calibrated")
    a = ap.parse_args()

    name = a.name or a.run_dir.name
    src = a.run_dir / a.cloud
    if not src.exists():
        raise SystemExit(f"missing {src}; run clean_cloud.py first")

    stats = a.run_dir / "clean_stats.json"
    units = json.loads(stats.read_text()).get("units") if stats.exists() else None
    if units != "m" and not a.force:
        raise SystemExit(
            f"{src.name} is not scale-calibrated (units={units!r}).\n"
            "  mjcf-terrain refuses arbitrary units -- terrain 10% off changes the\n"
            "  step heights a policy trains on. Run:\n"
            "    recon/calibrate_scale.py <run_dir>\n"
            "    recon/clean_cloud.py <run_dir> --scale auto\n"
            "  (--force overrides, for geometry debugging only)")

    pts = np.asarray(o3d.io.read_point_cloud(str(src)).points)
    print(f"{name}: {len(pts):,} points, extent {(pts.max(0)-pts.min(0)).round(2)} m")

    # Ground wants a *near*-minimum, not a modest low percentile: a cell straddling
    # a partition is mostly panel points, so even p10 lands partway up the wall and
    # the real floor is thrown away as "non-ground". Outlier removal already ran, so
    # the bottom of the distribution is trustworthy.
    pct = a.percentile if a.percentile is not None else (95.0 if a.surface == "top" else 3.0)
    print(f"surface: {a.surface} (p{pct:g} of each cell's z)")
    grid, seen, xy_min = rasterize(pts, a.cell, pct, a.min_points)
    print(f"grid {grid.shape[0]}x{grid.shape[1]} at {a.cell*100:.0f} cm  "
          f"observed {100*seen.mean():.1f}%")

    if a.crop:
        grid, seen, xy_min = crop_to_observed(grid, seen, xy_min, a.cell, a.margin)
        print(f"cropped to {grid.shape[0]}x{grid.shape[1]}  "
              f"observed {100*seen.mean():.1f}%")

    floor = float(np.percentile(grid[seen], 1.0))

    if a.surface == "ground":
        # A cell containing only a partition face has no floor in it, so even a low
        # percentile sits high up -- that is a wall, not ground. Reject those cells
        # rather than clipping them, or they survive as the spires that make an
        # indoor heightfield a canyon. Consequence worth stating: furniture and
        # walls are absent from the terrain, so obstacles must come from elsewhere.
        wall = seen & (grid > floor + a.ground_tol)
        if wall.any():
            print(f"  dropped {wall.sum():,} non-ground cells "
                  f"({100*wall.mean():.1f}% of field, >{a.ground_tol} m above floor)")
            seen = seen & ~wall
        if not seen.any():
            raise SystemExit("no ground cells survived; raise --ground-tol")

    mode = a.fill
    if mode == "auto":
        mode = "floor" if seen.mean() < 0.5 else "nearest"
        print(f"fill: {mode} (observed {100*seen.mean():.0f}%)")
    grid = fill_holes(grid, seen, mode, floor)
    grid = np.clip(grid - floor, 0.0, a.max_height)

    if a.smooth > 0:
        # Monocular depth noise puts ~5 cm of high-frequency roughness on what was
        # a flat carpet, and that is enough to topple a keyframe-posed G1 -- measured,
        # not assumed. Smoothing is a deliberate choice about what is signal: it keeps
        # slopes, kerbs and steps (wide) and removes per-cell depth jitter (narrow).
        # Record the sigma; terrain the policy trains on must be reproducible.
        grid = ndimage.gaussian_filter(grid, sigma=a.smooth, mode="nearest")
        print(f"  gaussian smooth sigma={a.smooth} cells "
              f"({a.smooth*a.cell*100:.0f} cm)")

    med = a.median if a.median is not None else (3 if a.surface == "ground" else 0)
    if med > 1:
        # Salt-and-pepper spikes survive every step above: a cell straddling the
        # base of a partition takes its height from the panel face, giving a
        # one-cell-wide 50 cm fin that no robot should have to negotiate. A median
        # deletes those while leaving genuine steps and kerbs, which are wide.
        before = float(grid.max())
        grid = ndimage.median_filter(grid, size=med, mode="nearest")
        print(f"  median filter {med}x{med}: peak {before:.2f} -> {grid.max():.2f} m")

    relief = float(grid.max())
    if relief <= 0:
        raise SystemExit("terrain is perfectly flat -- nothing to build")
    norm = grid / relief

    nrow, ncol = norm.shape
    rx, ry = ncol * a.cell / 2.0, nrow * a.cell / 2.0
    hf = f'<hfield name="{name}" nrow="{nrow}" ncol="{ncol}" ' \
         f'size="{rx:.4f} {ry:.4f} {relief:.4f} {a.base_z}"/>'

    out_dir = Path("terrain/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", norm.astype(np.float64))
    (out_dir / f"{name}.json").write_text(json.dumps({
        "name": name, "nrow": nrow, "ncol": ncol,
        "cell_m": a.cell, "relief_m": round(relief, 4), "base_z": a.base_z,
        "size": [round(rx, 4), round(ry, 4), round(relief, 4), a.base_z],
        "hfield_xml": hf,
        "origin_xy_m": [round(float(v), 4) for v in xy_min],
        "observed_frac": round(float(seen.mean()), 4),
        "source_cloud": str(src),
        "surface": a.surface, "percentile": pct,
        "max_height_m": a.max_height, "fill": mode, "median": med,
        "smooth_sigma_cells": a.smooth,
        "ground_tol_m": a.ground_tol if a.surface == "ground" else None,
    }, indent=2))
    np.save(out_dir / f"{name}_observed.npy", seen)

    print(f"\nrelief {relief:.3f} m over {2*rx:.1f} x {2*ry:.1f} m")
    print(f"wrote  {out_dir/f'{name}.npy'}")
    print(f"  {hf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
