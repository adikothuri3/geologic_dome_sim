"""Step B1: the Phase 1 fluency gate -- confirm the G1's structure from the live model.

Everything the rest of Phase 1 assumes about unitree_g1 is asserted here rather than
trusted from documentation.

    python sims/mujoco/scripts/inspect_model.py
"""

import os
import pathlib
import sys

import mujoco
import numpy as np

MENAGERIE = pathlib.Path(os.environ.get("MENAGERIE_DIR", "~/src/menagerie")).expanduser()
SCENE = MENAGERIE / "unitree_g1" / "scene.xml"


def main() -> None:
    if not SCENE.exists():
        sys.exit(f"scene not found: {SCENE}\nSet MENAGERIE_DIR, or finish install step A4.")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    np.set_printoptions(precision=3, suppress=True, linewidth=140)

    print("== counts ==")
    for field in ("nq", "nv", "nu", "njnt", "nbody", "ngeom", "nkey", "nhfield"):
        print(f"  {field:8s} {getattr(model, field)}")
    print(f"  timestep {model.opt.timestep}")

    # nq = 7 (free joint: 3 pos + 4 quat) + one coordinate per hinge.
    print("\n== actuators ==")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jnt = model.actuator_trnid[i, 0]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt)
        print(
            f"  {i:2d} {name:28s} joint={jnt_name:28s} "
            f"gain={model.actuator_gainprm[i, 0]:8.1f} ctrlrange={model.actuator_ctrlrange[i]}"
        )

    print("\n== bodies ==")
    for i in range(model.nbody):
        print(f"  {i:2d} {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")

    print("\n== keyframes ==")
    for i in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        print(f"  {i}: {name}")
        print(f"     qpos {model.key_qpos[i]}")
        print(f"     ctrl {model.key_ctrl[i]}")

    print("\n== assertions ==")
    checks = [
        ("nu == 29", model.nu == 29),
        ("nq == 36", model.nq == 36),
        ("nq == nu + 7", model.nq == model.nu + 7),
        ("nkey >= 1", model.nkey >= 1),
        ("'stand' keyframe present", "stand" in
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]),
        ("'pelvis' body present", mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis") != -1),
    ]
    failed = [label for label, ok in checks if not ok]
    for label, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    if failed:
        sys.exit(f"\n{len(failed)} assertion(s) failed -- the model differs from what Phase 1 assumes.")
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
