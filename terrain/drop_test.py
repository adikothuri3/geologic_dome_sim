"""Step B2 gate: drop spheres onto the heightfield and confirm the contacts are sane.

This is the same check `.claude/skills/mjcf-terrain` mandates for real2sim terrain,
rehearsed here on synthetic terrain so the Phase 4 muscle already exists.

Runs standalone against a minimal hfield-only scene -- no robot, no meshes -- so a
terrain failure can never be confused with a robot failure.

    python terrain/drop_test.py

Drop points are deliberately asymmetric: a row/col swap or a flipped row order would
show up as a mismatch between the predicted and actual resting height.

The probe is a BOX, not a sphere. A sphere rolls indefinitely down even a 4-degree slope
(MuJoCo models sliding friction, not rolling resistance), so "still moving" would report a
terrain defect where there is none. A box tests what we actually care about: does a contact
settle without jitter, penetration, or tunnelling.
"""

import argparse
import pathlib
import sys

import mujoco
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from terrain.make_hfield import (  # noqa: E402
    fill_hfield, fill_hfield_from_asset, load_asset, sample_height,
)

# Must mirror the <hfield> line in sim/scene_g1_hfield.xml.
HFIELD_ATTRS = 'nrow="128" ncol="128" size="3 3 0.15 0.1"'
BOX_HALF = 0.08
AT_REST_SPEED = 5e-3
DROP_POINTS = [(0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (-1.2, 0.9)]

def scene_xml(hfield_attrs: str) -> str:
    return f"""
<mujoco model="hfield drop test">
  <option timestep="0.002" integrator="implicitfast"/>
  <asset>
    <hfield name="terrain" {hfield_attrs}/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 0"
          friction="1 0.005 0.0001" condim="3"/>
    <body name="probe" pos="0 0 1">
      <freejoint/>
      <geom name="probe" type="box" size="{BOX_HALF} {BOX_HALF} {BOX_HALF}" density="500"/>
    </body>
  </worldbody>
</mujoco>
"""


XML = scene_xml(HFIELD_ATTRS)


def drop_points_for(model, grid=None, observed=None, n: int = 6) -> list[tuple[float, float]]:
    """Asymmetric probe points spread over the actual field, in metres.

    On a reconstruction, points are chosen on locally *flat* cells. Real terrain
    has vertical discontinuities that synthetic terrain does not -- a cubicle
    edge is an 80 cm cliff across one 5 cm cell -- and a box straddling one
    legitimately topples off it. That is correct physics, not a terrain defect,
    so probing there would only produce false failures. What this gate is for is
    jitter, penetration and tunnelling, all of which show up fine on flat cells.
    """
    rx, ry = float(model.hfield_size[0][0]), float(model.hfield_size[0][1])
    if grid is None:
        return [(0.0, 0.0), (0.55 * rx, 0.0), (0.0, 0.55 * ry),
                (-0.45 * rx, 0.35 * ry), (0.30 * rx, -0.60 * ry)]

    elev = float(model.hfield_size[0][2])
    h = grid * elev
    nrow, ncol = h.shape
    # Roughness over a box-sized footprint; flat means the probe rests on one surface.
    # Cover the whole box footprint plus a cell of margin: a probe that merely
    # clips a neighbouring step perches on it and never reaches the surface
    # sample_height predicts.
    pad = int(np.ceil(BOX_HALF / (2 * rx / ncol))) + 1
    rough = ndimage.maximum_filter(h, 2 * pad + 1) - ndimage.minimum_filter(h, 2 * pad + 1)

    # Never probe the boundary: a box dropped at exactly +/-rx has half its
    # footprint off the field and falls into the void, which reports a terrain
    # failure where there is none.
    ok = np.zeros_like(rough, bool)
    mg = pad + 2
    ok[mg:nrow - mg, mg:ncol - mg] = True
    if observed is not None:
        # Prefer cells the camera actually saw -- filled cells are invented and
        # testing them measures our fill policy, not the reconstruction.
        seen = ndimage.binary_erosion(observed, np.ones((2 * pad + 1, 2 * pad + 1)))
        if (ok & seen).sum() > 9:
            ok &= seen
    rough = np.where(ok, rough, np.inf)

    # Spread the probes: split the field into a coarse grid and take the flattest
    # cell in each block, so they never all cluster on one patch of carpet.
    pts, nb = [], 3
    for by in range(nb):
        for bx in range(nb):
            r0, r1 = by * nrow // nb, (by + 1) * nrow // nb
            c0, c1 = bx * ncol // nb, (bx + 1) * ncol // nb
            blk = rough[r0:r1, c0:c1]
            i = int(np.argmin(blk))
            r, c = r0 + i // blk.shape[1], c0 + i % blk.shape[1]
            if not np.isfinite(rough[r, c]) or rough[r, c] > 0.02:
                continue
            x = c / (ncol - 1) * 2 * rx - rx
            y = r / (nrow - 1) * 2 * ry - ry
            pts.append((round(x, 3), round(y, 3)))
    return pts[:n] if pts else [(0.0, 0.0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default=None,
                    help="heightfield built by recon/cloud_to_hfield.py "
                         "(default: the synthetic Phase 1 terrain)")
    a = ap.parse_args()

    observed = None
    if a.asset:
        grid_src, meta = load_asset(a.asset)
        obs_path = pathlib.Path(__file__).resolve().parent / "assets" / f"{a.asset}_observed.npy"
        if obs_path.exists():
            observed = np.load(obs_path)
        attrs = (f'nrow="{meta["nrow"]}" ncol="{meta["ncol"]}" '
                 f'size="{" ".join(str(v) for v in meta["size"])}"')
        model = mujoco.MjModel.from_xml_string(scene_xml(attrs))
        grid = fill_hfield_from_asset(model, a.asset)
        print(f"asset {a.asset}: {meta['nrow']}x{meta['ncol']} at "
              f"{meta['cell_m']*100:.0f} cm, observed {100*meta['observed_frac']:.1f}%")
    else:
        model = mujoco.MjModel.from_xml_string(XML)
        grid = fill_hfield(model)

    global DROP_POINTS
    DROP_POINTS = drop_points_for(model, grid if a.asset else None, observed)
    data = mujoco.MjData(model)

    print(f"grid {grid.shape}  normalized range [{grid.min():.3f}, {grid.max():.3f}]")
    print(f"hfield size (rx, ry, elev_z, base_z) = {model.hfield_size[0]}")
    relief = (grid.max() - grid.min()) * model.hfield_size[0][2]
    print(f"terrain relief = {relief * 100:.1f} cm\n")

    failures = []
    for x, y in DROP_POINTS:
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = (x, y, 1.0)
        mujoco.mj_forward(model, data)

        min_z = np.inf
        for _ in range(1500):  # 3 s at 2 ms
            mujoco.mj_step(model, data)
            min_z = min(min_z, data.qpos[2])

        predicted = sample_height(model, grid, x, y) + BOX_HALF
        actual = float(data.qpos[2])
        speed = float(np.linalg.norm(data.qvel[0:3]))
        error = abs(actual - predicted)

        at_rest = speed < AT_REST_SPEED
        no_tunnel = min_z > -0.05
        on_surface = error < 0.03  # terrain slope under the box accounts for a little

        status = "ok" if (at_rest and no_tunnel and on_surface) else "FAIL"
        print(
            f"  [{status}] ({x:+.1f}, {y:+.1f})  rest_z={actual:.4f}  "
            f"predicted={predicted:.4f}  err={error * 100:.2f} cm  "
            f"|v|={speed:.2e}  min_z={min_z:.3f}"
        )
        if status == "FAIL":
            reasons = []
            if not at_rest:
                reasons.append("still moving (jitter)")
            if not no_tunnel:
                reasons.append("tunnelled through terrain")
            if not on_surface:
                reasons.append("resting height disagrees with sample_height -- check row/col order")
            failures.append(f"({x}, {y}): {', '.join(reasons)}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("\nAll drop points rest cleanly. Terrain is safe to put the robot on.")


if __name__ == "__main__":
    main()
