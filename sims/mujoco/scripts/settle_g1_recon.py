"""Phase 4 gate: settle the Unitree G1 on reconstruction-derived terrain.

`.claude/skills/mjcf-terrain` requires this before the terrain is handed to
training: load the G1 from Menagerie, place it above the terrain, let it settle
under gravity, and confirm no penetration and no explosion.
`sims/mujoco/terrain/drop_test.py --asset ...` covers the terrain in isolation with a rigid
box; this covers the 30-body articulated robot that actually has to stand on it.

Spawn placement is not arbitrary. The G1's `stand` keyframe is straight-legged
and Phase 4 has no policy to catch a topple, so the robot is placed on the
flattest *observed* patch of terrain -- the same reasoning behind
`make_hfield.FLAT_PAD_FRAC`, except here we find a genuinely flat real spot
rather than manufacturing one.

    python sims/mujoco/scripts/settle_g1_recon.py --asset loop_office --render
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from terrain.make_hfield import (  # noqa: E402
    fill_hfield_from_asset, load_asset, sample_height,
)

MJ = pathlib.Path(__file__).resolve().parents[1]     # sims/mujoco/
REPO = pathlib.Path(__file__).resolve().parents[3]   # repo root
SCENE = MJ / "xmls" / "scene_g1_recon.xml"

SETTLE_S = 4.0
SIM_DT = 0.002
# Just enough to guarantee no initial interpenetration. Dropping the robot from
# higher makes the landing impact -- not the terrain -- dominate both the
# penetration peak and the slide, which is the wrong thing to be gating on.
SPAWN_CLEARANCE = 0.005
STEADY_S = 1.0              # trailing window judged as settled
AT_REST_SPEED = 0.05
MIN_PELVIS = 0.45           # below this the robot has collapsed
MAX_DRIFT = 0.20            # horizontal wander while settling
# Foot compliance, not a terrain defect. Measured on this scene: the same foot
# sinks 0.3 mm on flat ground and ~5 mm where a corner loads a sloped 5 cm cell,
# because Menagerie's ankle geoms carry their own solref (0.02) that wins over the
# terrain's. 10 mm leaves headroom over that while still catching real tunnelling.
MAX_PENETRATION = 0.010


def flattest_observed(model, grid, observed, footprint_m=1.6, max_h=0.25):
    """Centre of the flattest observed, *walkable* patch big enough for both feet.

    ``max_h`` keeps the spawn on the ground. Without it the flattest observed
    patch in an office walkthrough is a desk or the top of a cubicle run -- the
    robot stands there quite happily and the gate passes, having tested nothing
    anyone cares about.

    ``footprint_m`` is deliberately wider than the robot's stance. Clearing only
    the feet puts it shoulder-to-shoulder with whatever partition remains beside
    the corridor, and it topples into that on the first settle -- a spawn-placement
    failure that reads exactly like a terrain failure.
    """
    rx, ry, elev = (float(v) for v in model.hfield_size[0][:3])
    h = grid * elev
    nrow, ncol = h.shape
    pad = max(int(np.ceil(footprint_m / 2 / (2 * rx / ncol))), 1)
    k = 2 * pad + 1
    rough = ndimage.maximum_filter(h, k) - ndimage.minimum_filter(h, k)

    base = np.zeros_like(rough, bool)
    base[pad:nrow - pad, pad:ncol - pad] = True
    base &= ndimage.maximum_filter(h, k) < max_h
    if not base.any():
        raise SystemExit(f"no flat walkable patch below {max_h} m -- terrain is all structure")

    # Standing on measured ground is the whole point; filled cells are perfectly
    # flat by construction, so a gate that lands on one proves nothing. Shrink the
    # required clearance before giving that up, and say so if we do.
    if observed is not None:
        for shrink in (1.0, 0.75, 0.5):
            kk = 2 * max(int(np.ceil(footprint_m * shrink / 2 / (2 * rx / ncol))), 1) + 1
            seen = ndimage.binary_erosion(observed, np.ones((kk, kk)))
            if (base & seen).sum() > 4:
                ok, on_observed = base & seen, True
                break
        else:
            ok, on_observed = base, False
    else:
        ok, on_observed = base, True

    i = int(np.argmin(np.where(ok, rough, np.inf)))
    r, c = i // ncol, i % ncol
    x = c / (ncol - 1) * 2 * rx - rx
    y = r / (nrow - 1) * 2 * ry - ry
    return float(x), float(y), float(rough[r, c]), on_observed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default="loop_office")
    ap.add_argument("--keyframe", default="crouch",
                    help="spawn pose. `stand` is straight-legged with nothing to catch "
                         "the floating base, so it topples on real terrain roughness; "
                         "`crouch` has bent knees and a lower CoM")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs" / "phase4_g1_recon.png")
    a = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    # Terrain must land in hfield_data before any Renderer exists, or the visuals
    # silently disagree with the physics (see terrain/make_hfield.py).
    grid = fill_hfield_from_asset(model, a.asset)
    _, meta = load_asset(a.asset)

    obs_path = MJ / "terrain" / "assets" / f"{a.asset}_observed.npy"
    observed = np.load(obs_path) if obs_path.exists() else None

    x, y, rough, on_observed = flattest_observed(model, grid, observed)
    z_terrain = sample_height(model, grid, x, y)
    print(f"asset {a.asset}: {meta['nrow']}x{meta['ncol']} @ {meta['cell_m']*100:.0f} cm, "
          f"relief {meta['relief_m']:.2f} m, observed {100*meta['observed_frac']:.1f}%")
    print(f"spawn ({x:.2f}, {y:.2f})  terrain z={z_terrain:.3f}  "
          f"roughness under footprint {rough*100:.1f} cm  "
          f"[{'measured ground' if on_observed else 'FILLED ground -- invented, not measured'}]")

    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, a.keyframe)
    if kid == -1:
        raise SystemExit(f"no keyframe {a.keyframe!r} in {SCENE.name}")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    start_z = float(data.qpos[2])
    data.qpos[0], data.qpos[1] = x, y
    data.qpos[2] = z_terrain + start_z + SPAWN_CLEARANCE
    mujoco.mj_forward(model, data)

    terrain_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    n = int(SETTLE_S / SIM_DT)
    steady_from = n - int(STEADY_S / SIM_DT)
    impact_pen = steady_pen = 0.0
    steady_who = None
    settled_xy = None
    for step in range(n):
        mujoco.mj_step(model, data)
        # Deepest interpenetration, counting *terrain* contacts only. Including the
        # robot's own self-contacts makes this gate report the crouch pose's
        # thigh-against-calf overlap as a terrain defect -- it measured 5 mm of
        # "penetration" that had nothing to do with the ground.
        pen, who = 0.0, None
        for i in range(data.ncon):
            c = data.contact[i]
            if terrain_gid not in (c.geom1, c.geom2):
                continue
            if -float(c.dist) > pen:
                pen = -float(c.dist)
                other = int(c.geom2 if c.geom1 == terrain_gid else c.geom1)
                who = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other]))
        if step < steady_from:
            impact_pen = max(impact_pen, pen)
        else:
            if pen > steady_pen:
                steady_pen, steady_who = pen, who
            if settled_xy is None:
                settled_xy = (float(data.qpos[0]), float(data.qpos[1]))

    pelvis = float(data.qpos[2]) - sample_height(model, grid, float(data.qpos[0]),
                                                 float(data.qpos[1]))
    speed = float(np.linalg.norm(data.qvel[:3]))
    # Drift that matters is motion after it settled; the initial topple-and-catch
    # is the straight-legged `stand` keyframe, not the terrain.
    drift = float(np.hypot(data.qpos[0] - settled_xy[0], data.qpos[1] - settled_xy[1]))
    total_drift = float(np.hypot(data.qpos[0] - x, data.qpos[1] - y))
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))

    print(f"\nafter {SETTLE_S:.0f}s:  pelvis {pelvis:.3f} m above terrain  |v|={speed:.3f}")
    print(f"  drift {total_drift*100:.1f} cm total, {drift*100:.1f} cm after settling")
    print(f"  penetration {impact_pen*1000:.2f} mm on impact, "
          f"{steady_pen*1000:.2f} mm steady-state"
          + (f" (deepest: {steady_who})" if steady_who else ""))

    fails = []
    if not finite:
        fails.append("simulation diverged (non-finite state)")
    if pelvis < MIN_PELVIS:
        fails.append(f"collapsed: pelvis {pelvis:.3f} < {MIN_PELVIS}")
    if speed > AT_REST_SPEED:
        fails.append(f"never settled: |v|={speed:.3f} > {AT_REST_SPEED}")
    if drift > MAX_DRIFT:
        fails.append(f"still sliding after settling: {drift*100:.1f} cm > {MAX_DRIFT*100:.0f} cm")
    if steady_pen > MAX_PENETRATION:
        fails.append(f"steady-state penetration {steady_pen*1000:.1f} mm "
                     f"> {MAX_PENETRATION*1000:.0f} mm ({steady_who})")

    if a.render:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (data.qpos[0], data.qpos[1], z_terrain + 0.7)
        cam.distance, cam.azimuth, cam.elevation = 3.6, 130, -8
        renderer = mujoco.Renderer(model, 720, 1280)
        renderer.update_scene(data, camera=cam)
        import imageio.v2 as iio
        a.out.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(a.out, renderer.render())
        print(f"rendered {a.out}")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\nG1 stands on the reconstructed terrain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
