"""Gate for the full-body-collision G1 before it is trained on.

    python scripts/check_full_collision.py

Adding collision geometry to 15 bodies is easy to get subtly wrong, and the failure mode is
quiet: a box placed a few centimetres off makes the robot rest on invisible geometry, or fight
itself while standing still, and the reward curve just looks bad for no visible reason. So
each property is asserted rather than eyeballed:

  1. the model compiles and keeps the geoms the environment contract needs
  2. standing at the nominal pose, nothing touches except the two feet
     -- this is the test that catches a misplaced or oversized box
  3. dropped and toppled, the new geometry actually catches the robot (torso/arms/hips make
     contact) and it settles instead of sinking, exploding, or tunnelling
  4. peak constraint count stays inside the njmax we train with, so constraints are never
     silently dropped
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

CONTRACT_GEOMS = [
    "left_foot", "right_foot", "left_shin", "right_shin",
    "left_thigh", "right_thigh", "left_hand_collision", "right_hand_collision",
]
CONTRACT_SITES = ["left_foot", "right_foot", "left_palm", "right_palm"]
# Penetration deeper than this at rest means a geom is in the wrong place, not just resting.
MAX_REST_PENETRATION = 0.005  # metres


def contacts(model, data):
    import mujoco
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        out.append((g1 or f"geom{c.geom1}", g2 or f"geom{c.geom2}", float(c.dist)))
    return out


def build(full: bool):
    from mujoco_playground import registry
    from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

    cfg = registry.get_default_config("G1JoystickFlatTerrain")
    if not full:
        return registry.load("G1JoystickFlatTerrain", config=cfg)
    import train_g1
    task = train_g1.sync_full_collision_scene()
    return g1_joystick.Joystick(task=task, config=cfg)


def main() -> None:
    import mujoco

    failures = []

    def check(ok, label, detail=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    print("== building both models ==")
    base = build(full=False)
    full = build(full=True)
    bm, fm = base.mj_model, full.mj_model

    n_base = sum(1 for i in range(bm.ngeom) if bm.geom_contype[i] or bm.geom_conaffinity[i])
    n_full = sum(1 for i in range(fm.ngeom) if fm.geom_contype[i] or fm.geom_conaffinity[i])
    print(f"  feet-only      collidable geoms: {n_base}")
    print(f"  full-collision collidable geoms: {n_full}")

    print("\n== 1. environment contract preserved ==")
    for g in CONTRACT_GEOMS:
        check(mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0, f"geom '{g}' present")
    for s in CONTRACT_SITES:
        check(mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_SITE, s) >= 0, f"site '{s}' present")
    check(fm.nu == 29, "29 actuators", f"nu={fm.nu}")
    check(fm.nq == 36, "nq == 36", f"nq={fm.nq}")
    check(n_full > n_base, "more collidable geoms than upstream", f"{n_base} -> {n_full}")

    print("\n== 2. no self-collision at the nominal pose ==")
    d = mujoco.MjData(fm)
    mujoco.mj_resetDataKeyframe(fm, d, fm.key("home").id)
    mujoco.mj_forward(fm, d)
    standing = contacts(fm, d)
    self_contacts = [c for c in standing if "floor" not in (c[0], c[1])]
    for g1, g2, dist in standing:
        print(f"     {g1} ~ {g2}  dist={dist:+.4f}")
    check(not self_contacts, "robot is not touching itself while standing",
          f"{len(self_contacts)} self-contacts")

    print("\n== 3. settles on its feet, like the baseline ==")
    for _ in range(500):
        mujoco.mj_step(fm, d)
    rest = contacts(fm, d)
    floor_geoms = sorted({(a if b == "floor" else b) for a, b, _ in rest if "floor" in (a, b)})
    deepest = min([c[2] for c in rest], default=0.0)
    print(f"     touching floor: {floor_geoms}")
    check(set(floor_geoms) <= {"left_foot", "right_foot"},
          "only feet touch the floor after settling", str(floor_geoms))
    check(deepest > -MAX_REST_PENETRATION, "no deep penetration at rest",
          f"deepest={deepest:.4f} m")
    pelvis_z = float(d.body("pelvis").xpos[2])
    check(pelvis_z > 0.6, "still standing (pelvis above 0.6 m)", f"pelvis_z={pelvis_z:.3f}")

    print("\n== 4. the new geometry catches a fall ==")
    d2 = mujoco.MjData(fm)
    mujoco.mj_resetDataKeyframe(fm, d2, fm.key("home").id)
    d2.qvel[:3] = [1.2, 0.0, 0.0]      # shove it over
    d2.qvel[3:6] = [0.0, 3.5, 0.0]
    peak_nefc, touched = 0, set()
    for _ in range(1500):
        mujoco.mj_step(fm, d2)
        peak_nefc = max(peak_nefc, int(d2.nefc))
        for a, b, _ in contacts(fm, d2):
            if a == "floor" or b == "floor":
                touched.add(b if a == "floor" else a)
    speed = float(np.linalg.norm(d2.qvel[:3]))
    pelvis_z2 = float(d2.body("pelvis").xpos[2])
    print(f"     floor contacts seen during the fall: {sorted(touched)}")
    print(f"     final pelvis_z={pelvis_z2:.3f}  |v|={speed:.4f}  peak_nefc={peak_nefc}")
    check(len(touched) > 2, "more than just the feet hit the ground", str(sorted(touched)))
    check(np.isfinite(d2.qpos).all(), "simulation stayed finite (no explosion)")
    check(pelvis_z2 > 0.02, "did not sink through the floor", f"pelvis_z={pelvis_z2:.3f}")
    check(speed < 0.05, "came to rest", f"|v|={speed:.4f}")

    print("\n== 5. njmax headroom ==")
    import train_g1
    njmax = train_g1.NJMAX_FULL_COLLISION
    print(f"     peak nefc observed: {peak_nefc}   configured njmax: {njmax}")
    check(peak_nefc < njmax, "peak constraints fit inside njmax",
          f"{peak_nefc} < {njmax}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        sys.exit(1)
    print("Full-collision model OK.")


if __name__ == "__main__":
    main()
