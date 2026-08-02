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

import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from terrain.make_hfield import fill_hfield, sample_height  # noqa: E402

# Must mirror the <hfield> line in sim/scene_g1_hfield.xml.
HFIELD_ATTRS = 'nrow="128" ncol="128" size="3 3 0.15 0.1"'
BOX_HALF = 0.08
AT_REST_SPEED = 5e-3
DROP_POINTS = [(0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (-1.2, 0.9)]

XML = f"""
<mujoco model="hfield drop test">
  <option timestep="0.002" integrator="implicitfast"/>
  <asset>
    <hfield name="terrain" {HFIELD_ATTRS}/>
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


def main() -> None:
    model = mujoco.MjModel.from_xml_string(XML)
    grid = fill_hfield(model)
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
