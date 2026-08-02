"""Step A5 gate: prove an offscreen GL context actually renders before anything depends on it.

A broken GL context found late is what turns Phase 1 into a missed deadline, so this
runs standalone against Menagerie's own scene.xml -- no project assets required.

    MUJOCO_GL=egl python scripts/check_render.py

If this fails, try in order:
  1. create /usr/share/glvnd/egl_vendor.d/10_nvidia.json so glvnd finds the WSL driver
  2. MUJOCO_GL=osmesa (software, slower, but reliable)
"""

import os
import pathlib
import sys

import imageio.v3 as iio
import mujoco

MENAGERIE = pathlib.Path(os.environ.get("MENAGERIE_DIR", "~/src/menagerie")).expanduser()
SCENE = MENAGERIE / "unitree_g1" / "scene.xml"
OUT = pathlib.Path(__file__).resolve().parent.parent / "reports" / "check_render.png"


def main() -> None:
    print(f"MUJOCO_GL = {os.environ.get('MUJOCO_GL', '<unset>')}")
    print(f"mujoco    = {mujoco.__version__}")

    if not SCENE.exists():
        sys.exit(f"scene not found: {SCENE}\nSet MENAGERIE_DIR, or finish install step A4.")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, 240, 320)
    try:
        renderer.update_scene(data)
        frame = renderer.render()
    finally:
        renderer.close()

    if frame.max() == 0:
        sys.exit("FAIL: frame is entirely black -- the GL context is not rendering.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT, frame)
    print(f"OK  frame={frame.shape} max={frame.max()}  ->  {OUT}")


if __name__ == "__main__":
    main()
