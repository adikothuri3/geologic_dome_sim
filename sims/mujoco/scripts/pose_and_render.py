"""Step B4: the Phase 1 demo -- G1 posed through keyframes on a numpy heightfield, to MP4.

    python sims/mujoco/scripts/pose_and_render.py

No policy, no RL. Poses are held by the position actuators that Menagerie's g1.xml already
defines; `mj_resetDataKeyframe` seeds both qpos and ctrl from a keyframe, and moving
between poses is a straight interpolation of the ctrl targets.

Stepping follows the project's 50 Hz control-loop convention: set ctrl once, then advance
physics SUBSTEPS times (ctrl_dt 0.02 / sim_dt 0.002). This is the same loop Phase 2's
policy will run in, so it is written by hand here rather than hidden in a wrapper.
"""

import pathlib
import sys

import imageio.v2 as iio
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from terrain.make_hfield import fill_hfield, sample_height  # noqa: E402

MJ = pathlib.Path(__file__).resolve().parents[1]     # sims/mujoco/
REPO = pathlib.Path(__file__).resolve().parents[3]   # repo root
SCENE = MJ / "xmls" / "scene_g1_hfield.xml"
OUT = REPO / "reports" / "phase1_stand.mp4"

CTRL_DT = 0.02
SUBSTEPS = 10  # x sim_dt 0.002 = CTRL_DT
FPS = 25
CAPTURE_EVERY = int(round(1.0 / (FPS * CTRL_DT)))  # 2 control steps per frame
WIDTH, HEIGHT = 960, 544  # both divisible by 16 so the H.264 encoder never silently rescales

SETTLE_S = 1.0
CLEARANCE = 0.02

# Cap on how fast a position target may travel, rad/s. Without this, swinging both arms
# down from `arms_up` to `stand` (1.8 rad) inside one second dumps enough angular momentum
# into the floating base to topple it -- the legs are straight and there is no ankle
# strategy to absorb it. Blend times below are minimums; this rate limit can stretch them.
MAX_JOINT_RATE = 0.8

# (keyframe, min_blend_s, hold_s)
SEQUENCE = [
    ("stand", 0.0, 2.0),
    ("crouch", 1.0, 1.5),
    ("stand", 1.0, 1.5),
    ("t_pose", 1.0, 1.5),
    ("arms_up", 1.0, 1.5),
    ("stand", 1.0, 2.0),
]

COLLAPSE_HEIGHT = 0.45  # pelvis above local terrain; below this it has fallen
MAX_STAND_DRIFT = 0.05


def pelvis_clearance(model, data, grid) -> float:
    """Pelvis height above the terrain directly beneath it."""
    x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
    return float(z - sample_height(model, grid, float(x), float(y)))


def main() -> None:
    if not (MJ / "xmls" / "menagerie").exists():
        sys.exit(
            "sims/mujoco/xmls/menagerie is missing. Point it at the Menagerie clone:\n"
            "  Linux:   ln -s ~/src/menagerie sims/mujoco/xmls/menagerie\n"
            "  Windows: New-Item -ItemType Junction -Path sims\\mujoco\\xmls\\menagerie "
            "-Target C:\\Users\\Aditya\\src\\menagerie"
        )

    model = mujoco.MjModel.from_xml_path(str(SCENE))

    # Terrain must be written before the Renderer exists, otherwise mjr_uploadHField
    # would be required to keep the visuals in sync with the physics.
    grid = fill_hfield(model)

    assert abs(model.opt.timestep - CTRL_DT / SUBSTEPS) < 1e-9, (
        f"timestep {model.opt.timestep} does not divide CTRL_DT into {SUBSTEPS} substeps"
    )

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)

    # Keyframe z values assume a floor at z=0; re-seat the base on the terrain, preserving
    # each pose's own height offset. Spawn is the origin, where the level pad is.
    terrain_z = sample_height(model, grid, 0.0, 0.0)
    data.qpos[2] += terrain_z + CLEARANCE
    mujoco.mj_forward(model, data)
    print(f"terrain at origin = {terrain_z * 100:.1f} cm, pelvis seeded at {data.qpos[2]:.3f} m")

    targets = {name: model.key(name).ctrl.copy() for name, _, _ in SEQUENCE}
    current = targets["stand"].copy()

    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # Grazing angle: terrain relief reads in silhouette against the sky, not from above.
    cam.distance, cam.elevation = 3.0, -8.0

    frames: list[np.ndarray] = []
    step = 0
    min_clearance = np.inf
    drift_report: list[tuple[str, float]] = []

    def control_step() -> None:
        """One 50 Hz control tick: hold ctrl, advance physics, maybe capture a frame."""
        nonlocal step, min_clearance
        data.ctrl[:] = current
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        min_clearance = min(min_clearance, pelvis_clearance(model, data, grid))

        if step % CAPTURE_EVERY == 0:
            cam.azimuth = 120.0 + 3.0 * data.time  # slow orbit, 3 deg per simulated second
            cam.lookat = data.body("pelvis").subtree_com
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        step += 1

    try:
        for _ in range(int(SETTLE_S / CTRL_DT)):
            control_step()
        print(f"settled: pelvis {pelvis_clearance(model, data, grid):.3f} m above terrain")

        for name, min_blend_s, hold_s in SEQUENCE:
            goal = targets[name]
            start = current.copy()

            excursion = float(np.abs(goal - start).max())
            blend_s = max(min_blend_s, excursion / MAX_JOINT_RATE)
            n_blend = int(blend_s / CTRL_DT)
            if n_blend:
                print(f"  -> {name:8s} blend {blend_s:.2f}s (max joint move {excursion:.2f} rad)")

            for i in range(n_blend):
                alpha = (i + 1) / n_blend
                eased = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep: zero rate at both ends
                current = start + (goal - start) * eased
                control_step()
            current = goal.copy()

            heights = []
            for _ in range(int(hold_s / CTRL_DT)):
                control_step()
                heights.append(pelvis_clearance(model, data, grid))
            drift = max(heights) - min(heights)
            if name == "stand":
                drift_report.append((name, drift))
            print(f"  {name:8s} held {hold_s:.1f}s  clearance {np.mean(heights):.3f} m  drift {drift * 100:.1f} cm")
    finally:
        renderer.close()

    print(f"\n{len(frames)} frames, {len(frames) / FPS:.1f} s at {FPS} fps")
    iio.mimwrite(OUT, frames, fps=FPS, codec="libx264", quality=8)
    print(f"wrote {OUT}")

    print("\n== assertions ==")
    checks = [
        (f"never collapsed (min pelvis clearance {min_clearance:.3f} m > {COLLAPSE_HEIGHT})",
         min_clearance > COLLAPSE_HEIGHT),
    ]
    for name, drift in drift_report:
        checks.append((f"{name} drift {drift * 100:.1f} cm < {MAX_STAND_DRIFT * 100:.0f} cm",
                       drift < MAX_STAND_DRIFT))

    failed = [label for label, ok in checks if not ok]
    for label, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    if failed:
        sys.exit(
            "\nDemo did not hold. Two usual causes:\n"
            "  * terrain too aggressive for a straight-legged stand -> lower elevation_z in\n"
            "    sims/mujoco/xmls/scene_g1_hfield.xml (currently 0.15) or raise FLAT_PAD_FRAC in\n"
            "    sims/mujoco/terrain/make_hfield.py so the level spawn pad is wider;\n"
            "  * a pose transition swinging too much mass too fast -> lower MAX_JOINT_RATE."
        )
    print("\nPhase 1 demo criteria met.")


if __name__ == "__main__":
    main()
