"""Isaac smoke gates — the ladder from sims/isaac/README.md, as asserts.

    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims\\isaac\\scripts\\check_isaac.py --gate a
    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims\\isaac\\scripts\\check_isaac.py --gate b [--num_envs 8]
    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims\\isaac\\scripts\\check_isaac.py --gate c [--num_envs 32]

Gate A: SimulationApp opens headless and closes clean; report versions.
        First run is SLOW (shader cache compile) — that is expected, not a hang.
Gate B: the full-collision G1 flat task (Dome-G1FullCollision-Flat-v0) builds,
        resets, and survives zero-action steps with finite observations.
        First run downloads USD assets from NVIDIA's S3 — also slow once.
Gate C: the domain randomization in Dome-G1FullCollision-Flat-DR-v0 actually reaches
        PhysX. Every DR term is checked by reading the per-environment spread back out
        of the simulation, not by inspecting the config. This gate exists because the
        failure it catches is silent: upstream's own G1 config disables most of Isaac's
        randomization by setting terms to None and collapsing ranges to point values,
        and a task that randomizes nothing trains and logs exactly like one that does.

This box is below Isaac's minimum spec (8 GB VRAM / 16 GB RAM vs 16/32), which is
exactly why these gates exist: they answer "does it load and step at all" before any
cloud money is spent. Always headless here — the RTX viewport alone eats ~7 GB.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# AppLauncher must configure and start the SimulationApp BEFORE any isaaclab/omni
# import. Argparse therefore runs first, app second, imports third.
from isaaclab.app import AppLauncher  # noqa: E402  (safe: pure-python module)

REPO = pathlib.Path(__file__).resolve().parents[3]
GATELOG = REPO / "runs" / "isaac" / "gates.log"


def report(msg: str) -> None:
    """Kit hijacks python stdout on Windows -- write verdicts where they survive:
    stderr for the console, runs/isaac/gates.log for the record."""
    print(msg, file=sys.stderr, flush=True)
    GATELOG.parent.mkdir(parents=True, exist_ok=True)
    with GATELOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gate", choices=["a", "b", "c"], required=True)
parser.add_argument("--num_envs", type=int, default=8,
                    help="gates b/c; 8 is deliberately tiny for the 8 GB card. Gate c "
                         "needs enough samples for a spread to be meaningful — use 32.")
parser.add_argument("--steps", type=int, default=50, help="gate b zero-action steps")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True  # never open a viewport on this box

t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
report(f"[gate {args.gate}] SimulationApp up in {time.time() - t0:.1f}s (first run compiles shaders — slow is normal)")


def gate_a() -> None:
    import isaacsim  # noqa: F401
    from importlib.metadata import version
    for pkg in ("isaacsim", "isaaclab", "isaaclab-tasks", "isaaclab-assets", "rsl-rl-lib"):
        try:
            report(f"  {pkg:>16} {version(pkg)}")
        except Exception:
            report(f"  {pkg:>16} NOT INSTALLED")
    report("[gate a] SimulationApp opened headless and will close clean.")


def gate_b() -> None:
    import gymnasium as gym
    import torch

    sys.path.insert(0, str(REPO / "sims" / "isaac" / "tasks"))
    import dome_g1  # noqa: F401  (registers Dome-G1FullCollision-Flat-v0)
    from dome_g1.flat_env_cfg import G1FullCollisionFlatEnvCfg

    env_cfg = G1FullCollisionFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    t = time.time()
    env = gym.make("Dome-G1FullCollision-Flat-v0", cfg=env_cfg)
    report(f"  env built in {time.time() - t:.1f}s (first run downloads G1 USD assets)")

    obs, _ = env.reset()
    action_dim = env.unwrapped.action_manager.total_action_dim
    device = env.unwrapped.device
    report(f"  num_envs={env.unwrapped.num_envs}  action_dim={action_dim}  device={device}")
    assert action_dim > 0, "action manager reports zero actions — env cfg is broken"
    zeros = torch.zeros(env.unwrapped.num_envs, action_dim, device=device)

    finite = True
    for i in range(args.steps):
        obs, rew, terminated, truncated, info = env.step(zeros)
        pol = obs["policy"] if isinstance(obs, dict) else obs
        if not torch.isfinite(pol).all():
            finite = False
            report(f"  [FAIL] non-finite observation at step {i}")
            break
    env.close()
    if not finite:
        sys.exit(1)
    report(f"[gate b] {args.steps} zero-action steps, observations finite. "
          "The full-collision G1 loads and steps on this box.")


def gate_c() -> None:
    """Assert every DR term in DomeG1DREventCfg produces per-environment spread.

    The measurement is deliberately made against the physics buffers PhysX is actually
    integrating (`root_physx_view`, `robot.data.*` written by the event terms), because
    the bug being guarded against — a term that is configured but no-ops — leaves the
    config looking perfect. Two ways that happens here, both real: a `scale` operation on
    a property whose default is 0.0 multiplies nothing by something, and a body-name
    pattern that matches no body randomizes an empty selection.
    """
    import gymnasium as gym
    import torch

    sys.path.insert(0, str(REPO / "sims" / "isaac" / "tasks"))
    import dome_g1  # noqa: F401  (registers the Dome tasks)
    from dome_g1.flat_env_cfg import G1FullCollisionFlatDREnvCfg

    env_cfg = G1FullCollisionFlatDREnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    t = time.time()
    env = gym.make("Dome-G1FullCollision-Flat-DR-v0", cfg=env_cfg)
    env.reset()  # startup events fire on the first reset; reset events fire every time
    report(f"  DR env built + reset in {time.time() - t:.1f}s, num_envs={env.unwrapped.num_envs}")

    robot = env.unwrapped.scene["robot"]
    view = robot.root_physx_view
    failures: list[str] = []

    def check(label: str, values: torch.Tensor, expect: str) -> None:
        """Fail unless the property differs BETWEEN environments.

        Dimension 0 is the environment, so the spread is taken along it with every other
        axis (bodies, joints, shapes) held fixed. Flattening first would instead measure
        the difference between a knee and a finger — large, constant, and no evidence of
        anything. The property under test is Playground's: that all N envs integrate
        *different physics simultaneously* rather than sharing one randomized draw.
        """
        per_element = values.max(dim=0).values - values.min(dim=0).values
        spread = float(per_element.max())
        lo, hi = float(values.min()), float(values.max())
        ok = spread > 1e-9
        report(f"  {'ok ' if ok else 'FAIL'} {label:<24} max across-env spread {spread:>9.5g}"
               f"   values {lo:.5g}…{hi:.5g}   (expected {expect})")
        if not ok:
            failures.append(f"{label}: identical in all {args.num_envs} envs — the term is a no-op")

    # physics_material: (num_envs, num_shapes, 3) = static friction, dynamic friction, restitution
    materials = view.get_material_properties()
    check("link static friction", materials[..., 0], "0.4-1.0")
    check("link dynamic friction", materials[..., 1], "0.3-0.9")

    # link_mass / torso_mass
    masses = view.get_masses()
    check("per-link mass (kg)", masses, "x0.9-1.1 of default")
    check("total robot mass (kg)", masses.sum(dim=-1, keepdim=True), "about +-5% of nominal")
    torso_idx = robot.find_bodies("torso_link")[0]
    check("torso mass (kg)", masses[:, torso_idx], "+-1.0 kg on top of the scale")

    # joint_parameters: written through write_joint_*_to_sim, so robot.data is current
    check("joint armature", robot.data.joint_armature, "x1.0-1.05 of default")

    # Joint friction is deliberately NOT randomized, and the gate says so out loud rather
    # than leaving a silently-inert term in the config: the G1 USD ships 0.0 friction on
    # every joint, and a `scale` operation cannot lift zero. Asserted rather than assumed,
    # so a future asset that does carry joint friction surfaces here as a prompt to add it.
    friction = getattr(robot.data, "joint_friction_coeff", None)
    if friction is None:  # older isaaclab spelled it joint_friction
        friction = robot.data.joint_friction
    if float(friction.abs().max()) == 0.0:
        report("  ok  joint friction             0.0 on every joint in the G1 USD — deliberately not "
               "randomized (MuJoCo's frictionloss has no counterpart). See flat_env_cfg.py.")
    else:
        report(f"  note the G1 asset now HAS joint friction (max {float(friction.abs().max()):.4g}); "
               "MuJoCo randomizes it x U(0.5, 2.0). Consider friction_distribution_params.")

    # reset_robot_joints: +-0.05 rad offsets, applied fresh at every reset. Measured as a
    # deviation from the nominal pose, which itself spans ~2 rad across the 37 joints.
    check("initial joint pos (rad)", robot.data.joint_pos - robot.data.default_joint_pos,
          "+-0.05 around default")

    # The command distribution is part of the task definition too — a heading controller
    # silently overwriting the yaw-rate command is the same class of silent bug.
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd = cmd_term.command
    report(f"  ok  command vx range in sample   [{float(cmd[:, 0].min()):+.2f}, {float(cmd[:, 0].max()):+.2f}]  "
           f"heading_command={cmd_term.cfg.heading_command}")
    if cmd_term.cfg.heading_command:
        failures.append("heading_command is True — sampled ang_vel_z never reaches the robot")
    if float(cmd[:, 0].min()) >= 0.0 and args.num_envs >= 32:
        failures.append("no negative vx sampled — the policy would never learn to walk backwards")

    # push_robot is an interval term; confirm the manager scheduled it rather than
    # inspecting the config, since `None`-ing a term is exactly how upstream disables it.
    interval_terms = env.unwrapped.event_manager.active_terms.get("interval", [])
    if "push_robot" in interval_terms:
        report(f"  ok  interval events active     {interval_terms}")
    else:
        failures.append(f"push_robot not in active interval events: {interval_terms}")

    env.close()
    for f in failures:
        report(f"  [FAIL] {f}")
    if failures:
        sys.exit(1)
    report("[gate c] every DR term produces per-env spread in PhysX; commands are direct "
           "3-channel joystick. The randomization is real, not just configured.")


try:
    {"a": gate_a, "b": gate_b, "c": gate_c}[args.gate]()
except BaseException:
    # Report BEFORE the finally. `simulation_app.close()` terminates the process itself
    # (Kit's own exit path), so an exception propagating out of a gate is swallowed
    # whole: no traceback, exit code 0, a gate that looks like it passed silently.
    import traceback
    report("[gate %s] FAILED\n%s" % (args.gate, traceback.format_exc()))
    simulation_app.close()
    sys.exit(1)
else:
    report(f"[gate {args.gate}] PASS  (total {time.time() - t0:.1f}s)")
finally:
    simulation_app.close()
