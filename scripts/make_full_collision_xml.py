"""Generate a full-body-collision G1 from Playground's MJX G1, using Menagerie mesh extents.

    python scripts/make_full_collision_xml.py [--out sim/g1_full_collision.xml]

Why this exists
---------------
Playground's `g1_mjx_feetonly.xml` collides only the feet with the world. Everything else
inherits `contype=0 conaffinity=0` from the `g1` default class, and contact happens solely
through a handful of explicit <pair> entries. That is a sound speed/accuracy trade on a flat
plane, and the wrong one for this project: on reconstructed Everest terrain a shin, knee, hip
or torso meeting rock is a real event the policy must have felt in simulation.

Why generated rather than hand-written
--------------------------------------
Hand-placing capsules on 30 bodies is exactly the kind of task where one wrong offset gives
you a robot resting on invisible geometry, and the error is nearly impossible to see in a
render. Instead every added primitive is sized and positioned from the axis-aligned bounding
box of that body's own Menagerie collision/visual meshes, computed from the compiled model.
The geometry therefore follows the real robot by construction.

What is kept, and why
---------------------
Upstream's named geoms -- left/right_foot, left/right_shin, left/right_thigh,
left/right_hand_collision -- are preserved untouched. `g1_constants.py` and `sensor.xml`
reference them by name (FEET_GEOMS, the contact sensors, the termination test), so renaming or
regenerating them would break the environment contract.

Asset provenance: every mesh referenced here resolves to mujoco_menagerie/unitree_g1/assets.
See notes/decisions.md -- Menagerie is this project's single source of robot geometry.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent

# Upstream geoms that the environment contract depends on. Never regenerate these.
CONTRACT_GEOMS = {
    "left_foot", "right_foot",
    "left_shin", "right_shin",
    "left_thigh", "right_thigh",
    "left_hand_collision", "right_hand_collision",
}

# Bodies whose collision is already covered by a contract geom above.
COVERED_BODIES = {
    "left_ankle_roll_link", "right_ankle_roll_link",   # feet
    "left_knee_link", "right_knee_link",               # shins
    "left_hip_roll_link", "right_hip_roll_link",       # thighs
    "left_wrist_yaw_link", "right_wrist_yaw_link",     # hands
}

# Contract geoms whose upstream size is a placeholder rather than the real link, refitted to
# the mesh while keeping their name.
#
# The shin capsules are radius 0.08 -- a 16 cm-thick shin -- and extend far enough down that
# they graze the floor while the robot is merely standing. Upstream could ship that because
# these geoms never collided with anything. With broadphase on it means permanent phantom
# ground contact, which would poison feet_slip, feet_air_time and the contact sensors.
# The names must survive: sensor.xml and _get_termination() address them.
REFIT_FROM_MESH = {"left_knee_link": "left_shin", "right_knee_link": "right_shin"}

# Bodies too small or too enclosed to be worth a primitive: adding one only risks resting
# self-contact for no fidelity gain (they sit inside their parent's hull).
SKIP_BODIES = {
    "left_ankle_pitch_link", "right_ankle_pitch_link",  # thin bracket inside the ankle
    "waist_roll_link",                                  # inside the torso/waist hull
    "left_wrist_roll_link", "right_wrist_roll_link",    # inside the forearm
    "left_wrist_pitch_link", "right_wrist_pitch_link",
}

# An AABB is an over-estimate of a curved link, so shrink slightly. Without this, adjacent
# links touch at the nominal pose and the robot fights itself while standing still.
SHRINK = 0.86
MIN_HALF = 0.012  # metres; below this a box is numerically pointless


def write(root, out: pathlib.Path) -> None:
    out.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def detect_resting_overlaps(body_xml: pathlib.Path):
    """Compile the generated model and report body pairs that touch at the nominal poses.

    Checked at both keyframes the scene ships with: `home` (what training resets to) and
    `knees_bent` (a deeper crouch, which closes the hip and knee angles). A pair that
    interpenetrates in either is a modelling artefact rather than real contact.
    """
    import shutil

    import mujoco

    from mujoco_playground._src.locomotion.g1 import g1_constants as consts

    xmls = consts.task_to_xml("flat_terrain").parent
    shutil.copyfile(body_xml, xmls / body_xml.name)
    scene_src = REPO / "sim" / "scene_g1_full_collision.xml"
    scene_dst = xmls / scene_src.name
    shutil.copyfile(scene_src, scene_dst)

    from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

    original = consts.task_to_xml
    consts.task_to_xml = lambda t: scene_dst if t == "probe_full" else original(t)
    try:
        env = g1_joystick.Joystick(task="probe_full")
    finally:
        consts.task_to_xml = original

    model = env.mj_model
    data = mujoco.MjData(model)

    pairs = set()
    for key in ("home", "knees_bent"):
        mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
        mujoco.mj_forward(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0:  # world/floor contact is legitimate
                continue
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
            pairs.add(tuple(sorted((n1, n2))))
    return sorted(pairs)


def load_upstream():
    """Compile Playground's G1 so we can read real mesh vertices, and locate its xml."""
    from mujoco_playground import registry
    from mujoco_playground._src.locomotion.g1 import g1_constants as consts

    env = registry.load("G1JoystickFlatTerrain")
    return env.mj_model, consts.task_to_xml("flat_terrain").parent / "g1_mjx_feetonly.xml"


def body_mesh_aabb(model, body_id):
    """AABB of every mesh geom on this body, expressed in the body frame."""
    import mujoco

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    found = False
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id:
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        if mid < 0:
            continue
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[adr:adr + num].reshape(-1, 3)

        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
        verts = verts @ rot.reshape(3, 3).T + model.geom_pos[gid]

        lo = np.minimum(lo, verts.min(axis=0))
        hi = np.maximum(hi, verts.max(axis=0))
        found = True
    return (lo, hi) if found else None


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "sim" / "g1_full_collision.xml"))
    args = ap.parse_args()

    model, upstream_xml = load_upstream()
    print(f"upstream: {upstream_xml}")

    tree = ET.parse(upstream_xml)
    root = tree.getroot()
    root.set("model", "g1 full collision")

    # 1. Make the collision class actually collide. Upstream leaves contype/conaffinity at 0
    #    (inherited from `g1`) so only explicit <pair>s produce contact; we want broadphase to
    #    find shin-on-rock without every pair being enumerated by hand.
    collision_default = None
    for d in root.iter("default"):
        if d.get("class") == "collision":
            collision_default = d
            break
    if collision_default is None:
        sys.exit("could not find the `collision` default class in the upstream xml")
    geom_default = collision_default.find("geom")
    geom_default.set("contype", "1")
    geom_default.set("conaffinity", "1")
    geom_default.set("condim", "3")
    geom_default.set("friction", "0.6 0.6 0.001")

    # 2. Index bodies in the XML tree by name, and record which already carry a collision geom.
    bodies = {}
    has_collision = set()
    for body in root.iter("body"):
        name = body.get("name")
        if not name:
            continue
        bodies[name] = body
        for g in body.findall("geom"):
            if g.get("class") == "collision" or g.get("class") == "foot":
                has_collision.add(name)

    added, skipped, refitted = [], [], []

    # Refit placeholder contract geoms to their real mesh extent before anything else.
    for body_name, geom_name in REFIT_FROM_MESH.items():
        body = bodies.get(body_name)
        if body is None:
            continue
        target = next((g for g in body.findall("geom") if g.get("name") == geom_name), None)
        if target is None:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        aabb = body_mesh_aabb(model, bid)
        if aabb is None:
            continue
        lo, hi = aabb
        centre = (hi + lo) / 2.0
        half = np.maximum((hi - lo) / 2.0 * SHRINK, MIN_HALF)
        before = f"{target.get('type')} size={target.get('size')}"
        for attr in ("fromto", "size", "pos", "quat"):
            target.attrib.pop(attr, None)
        target.set("type", "box")
        target.set("size", " ".join(f"{v:.5f}" for v in half))
        target.set("pos", " ".join(f"{v:.5f}" for v in centre))
        refitted.append((geom_name, before, half, centre))

    for name, body in bodies.items():
        if name in COVERED_BODIES or name in has_collision:
            skipped.append((name, "already covered")); continue
        if name in SKIP_BODIES:
            skipped.append((name, "deliberately skipped")); continue

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            skipped.append((name, "not in compiled model")); continue
        aabb = body_mesh_aabb(model, bid)
        if aabb is None:
            skipped.append((name, "no mesh geoms")); continue

        lo, hi = aabb
        centre = (hi + lo) / 2.0
        half = np.maximum((hi - lo) / 2.0 * SHRINK, MIN_HALF)

        geom = ET.SubElement(body, "geom")
        geom.set("name", f"{name}_collision")
        geom.set("class", "collision")
        geom.set("type", "box")
        geom.set("size", " ".join(f"{v:.5f}" for v in half))
        geom.set("pos", " ".join(f"{v:.5f}" for v in centre))
        added.append((name, half, centre))

    out = pathlib.Path(args.out)
    write(root, out)

    # 3. Second pass: find links that overlap while simply standing, and exclude those pairs.
    #
    #    Upstream's capsules were authored with collision switched off, so some of them
    #    interpenetrate at the nominal pose -- left_thigh (r=0.06) and left_shin (r=0.08)
    #    overlap by 16 mm. Harmless when nothing collides; once broadphase is on it becomes a
    #    large separating force that shoves the robot over before it takes a step.
    #
    #    These pairs are detected, not guessed, so the list stays correct if upstream changes
    #    its geometry. Only *adjacent* links overlap this way, and their mutual contact
    #    carries no physical meaning, so excluding them costs no fidelity against terrain.
    excludes = detect_resting_overlaps(out)
    if excludes:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for b1, b2 in excludes:
            el = ET.SubElement(contact, "exclude")
            el.set("name", f"selfoverlap_{b1}__{b2}")
            el.set("body1", b1)
            el.set("body2", b2)
        write(root, out)

    print(f"\nexcluded {len(excludes)} self-overlapping body pairs:")
    for b1, b2 in excludes:
        print(f"  {b1} <-> {b2}")

    header = (
        "<!--\n"
        "  GENERATED by scripts/make_full_collision_xml.py -- do not hand-edit.\n"
        "  Regenerate after any Playground update, then re-run scripts/check_full_collision.py.\n"
        "\n"
        "  Playground's G1 collides feet only. This variant collides every link, with each added\n"
        "  box derived from that body's own Menagerie mesh bounding box (shrunk by "
        f"{SHRINK}) so the\n"
        "  collision volume tracks the real robot instead of hand-guessed capsules.\n"
        "\n"
        "  Upstream's named geoms (feet, shins, thighs, hands) are preserved verbatim because\n"
        "  g1_constants.py and sensor.xml address them by name.\n"
        "-->\n"
    )
    body_xml = ET.tostring(root, encoding="unicode")
    out.write_text(header + body_xml + "\n", encoding="utf-8")

    print(f"\nrefitted {len(refitted)} placeholder geoms to their mesh:")
    for gname, before, half, centre in refitted:
        print(f"  {gname:28s} {before}  ->  box half={np.round(half, 3)} pos={np.round(centre, 3)}")

    print(f"\nadded {len(added)} collision boxes:")
    for name, half, centre in added:
        print(f"  {name:28s} half={np.round(half, 3)}  pos={np.round(centre, 3)}")
    print(f"\nskipped {len(skipped)}:")
    for name, why in skipped:
        print(f"  {name:28s} {why}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
