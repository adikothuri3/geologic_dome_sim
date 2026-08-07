"""Isaac smoke gates — the ladder from sims/isaac/README.md, as asserts.

    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims\\isaac\\scripts\\check_isaac.py --gate a
    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims\\isaac\\scripts\\check_isaac.py --gate b [--num_envs 8]

Gate A: SimulationApp opens headless and closes clean; report versions.
        First run is SLOW (shader cache compile) — that is expected, not a hang.
Gate B: the full-collision G1 flat task (Dome-G1FullCollision-Flat-v0) builds,
        resets, and survives zero-action steps with finite observations.
        First run downloads USD assets from NVIDIA's S3 — also slow once.

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
parser.add_argument("--gate", choices=["a", "b"], required=True)
parser.add_argument("--num_envs", type=int, default=8,
                    help="gate b only; 8 is deliberately tiny for the 8 GB card")
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


try:
    gate_a() if args.gate == "a" else gate_b()
finally:
    simulation_app.close()
report(f"[gate {args.gate}] PASS  (total {time.time() - t0:.1f}s)")
