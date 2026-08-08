"""Trail-scene gate: the Eiger Trail USD terrain loads and a G1 stands on it.

    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims\\isaac\\scripts\\check_trail.py [--num_envs 8]

What it asserts, in order:

  1. Dome-G1FullCollision-EigerTrail-v0 builds (USD terrain imports, PhysX cooks
     the ~1M-triangle collision mesh, the height-scan RayCaster finds the mesh).
  2. Env origins are the trail's spawn points, not the stock z=0 grid: their z
     values must span the strip's real elevation range.
  3. Robots don't fall through the mesh: after settle steps, every root height
     stays near its origin (within a fall-and-crouch tolerance, since zero
     actions on a slope mean the robot may sit down or topple — but a robot
     that fell *through* the terrain is hundreds of metres down).
  4. Observations (which include the height scan) stay finite.

Same conventions as check_isaac.py: headless always, verdicts to stderr +
runs/isaac/gates.log because Kit hijacks stdout (notes/setup.md).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

from isaaclab.app import AppLauncher  # noqa: E402  (safe: pure-python module)

REPO = pathlib.Path(__file__).resolve().parents[3]
GATELOG = REPO / "runs" / "isaac" / "gates.log"


def report(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    GATELOG.parent.mkdir(parents=True, exist_ok=True)
    with GATELOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8,
                    help="deliberately tiny for the 8 GB card")
parser.add_argument("--steps", type=int, default=100, help="zero-action settle steps")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True  # never open a viewport on this box

t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
report(f"[gate trail] SimulationApp up in {time.time() - t0:.1f}s")


def gate_trail() -> None:
    import gymnasium as gym
    import torch

    sys.path.insert(0, str(REPO / "sims" / "isaac" / "tasks"))
    import dome_g1.trail_env_cfg as trail  # registers the EigerTrail tasks

    env_cfg = trail.G1FullCollisionEigerTrailEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    t = time.time()
    env = gym.make("Dome-G1FullCollision-EigerTrail-v0", cfg=env_cfg)
    report(f"  env built in {time.time() - t:.1f}s (includes PhysX cooking ~1M collision tris)")

    origins = env.unwrapped.scene.env_origins
    z_span = float(origins[:, 2].max() - origins[:, 2].min())
    x_span = float(origins[:, 0].max() - origins[:, 0].min())
    report(f"  origins: x span {x_span:.0f} m, z span {z_span:.0f} m over {len(origins)} envs")
    assert z_span > 1.0, (
        "env-origin z span is flat — origins are the stock z=0 grid, "
        "not the trail spawn points; EigerTrailImporter did not take effect"
    )

    obs, _ = env.reset()
    action_dim = env.unwrapped.action_manager.total_action_dim
    zeros = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)

    root_z = env.unwrapped.scene["robot"].data.root_pos_w[:, 2]
    drop0 = origins[:, 2] - root_z
    report(f"  after reset: root sits {float(-drop0.max()):.2f}..{float(-drop0.min()):.2f} m above origin z")

    finite = True
    for i in range(args.steps):
        obs, rew, terminated, truncated, info = env.step(zeros)
        pol = obs["policy"] if isinstance(obs, dict) else obs
        if not torch.isfinite(pol).all():
            finite = False
            report(f"  [FAIL] non-finite observation at step {i}")
            break

    root_z = env.unwrapped.scene["robot"].data.root_pos_w[:, 2]
    drop = origins[:, 2] - root_z
    worst = float(drop.max())
    report(f"  after {args.steps} zero-action steps: worst drop below origin {worst:.2f} m")
    # Env resets teleport fallen robots back to origins, so a persistent large drop
    # means robots are falling THROUGH the mesh, not off it.
    through = worst > 5.0
    if through:
        report("  [FAIL] a robot is far below its origin — falling through the collision mesh?")

    env.close()
    if not finite or through:
        sys.exit(1)
    report(f"[gate trail] terrain loads, origins are on the trail (z span {z_span:.0f} m), "
           "robots stand on the mesh, observations finite.")


try:
    gate_trail()
except BaseException:
    # Report BEFORE the finally: simulation_app.close() terminates the process itself,
    # so an exception propagating out would otherwise vanish with exit code 0.
    import traceback
    report("[gate trail] FAILED\n%s" % traceback.format_exc())
    simulation_app.close()
    sys.exit(1)
else:
    report(f"[gate trail] PASS  (total {time.time() - t0:.1f}s)")
finally:
    simulation_app.close()
