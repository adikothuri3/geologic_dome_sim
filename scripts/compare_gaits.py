"""Compare the gait of two trained policies under an identical command.

    python scripts/compare_gaits.py runs/<run_a> runs/<run_b> [--command 1 0 0]

Written for the Phase 2 stretch experiment (change one reward weight, show how the gait
changes). Total reward cannot answer that question: changing a reward *weight* changes the
scale the run is scored on, so the two runs' reward numbers are not comparable. What is
comparable is the behaviour -- stride timing, duty factor, how well the commanded velocity is
actually tracked -- so that is what this measures.

Metrics per policy, over one fixed-command rollout:
  * tracked velocity vs commanded (the thing the policy is actually for)
  * steps/second and swing fraction per foot (the gait itself)
  * pelvis height and its variation (is it walking or crouch-shuffling?)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

ENV_NAME = "G1JoystickFlatTerrain"


# One env per collision model, reused across runs. Each MJX/warp env holds a few hundred MB of
# device buffers, so building one per run OOMs on this 8 GB card at three runs -- and the runs
# being compared share a collision model anyway, since that is what makes them comparable.
_ENV_CACHE: dict = {}


def get_env(collision: str):
    if collision not in _ENV_CACHE:
        import render_policy as rp
        _ENV_CACHE[collision] = rp.build_env(collision)
    return _ENV_CACHE[collision]


def rollout(run_dir: pathlib.Path, command, seconds: float, seed: int, use_best: bool):
    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import networks as ppo_networks
    from etils import epath
    from mujoco_playground.config import locomotion_params
    from orbax import checkpoint as ocp

    blob = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    env = get_env(blob.get("collision", "feet-only"))

    params_path = (run_dir / "params").resolve()
    if use_best and (run_dir / "best.json").exists():
        meta = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
        cand = REPO / meta["checkpoint"]
        if cand.exists():
            params_path = cand.resolve()

    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
    network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **dict(ppo_params.get("network_factory", {})),
    )
    make_policy = ppo_networks.make_inference_fn(network)
    raw = ocp.PyTreeCheckpointer().restore(epath.Path(params_path))
    norm = raw[0]
    if isinstance(norm, dict):
        template = running_statistics.init_state(
            {k: specs.Array(tuple(v), jp.dtype("float32"))
             for k, v in env.observation_size.items()})
        norm = template.replace(mean=norm["mean"], std=norm["std"],
                                summed_variance=norm["summed_variance"])
    inference_fn = jax.jit(make_policy((norm, raw[1]), deterministic=True))

    cmd = jp.array(command, dtype=jp.float32)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    state.info["command"] = cmd

    vels, heights, contacts = [], [], []
    n = int(seconds / env.dt)
    for _ in range(n):
        act_rng, rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step(state, action)
        state.info["command"] = cmd
        vels.append(np.asarray(env.get_local_linvel(state.data, "pelvis")))
        heights.append(float(state.data.qpos[2]))
        contacts.append(np.asarray(state.info["last_contact"], dtype=bool))
        if bool(state.done):
            break

    vels = np.array(vels)
    heights = np.array(heights)
    contacts = np.array(contacts)
    duration = len(vels) * env.dt

    # A step is a rising edge of foot contact: swing -> stance.
    touchdowns = [int(((~contacts[:-1, f]) & contacts[1:, f]).sum())
                  for f in range(contacts.shape[1])]
    swing_fraction = [float((~contacts[:, f]).mean()) for f in range(contacts.shape[1])]

    return {
        "survived_s": duration,
        "fell": len(vels) < n,
        "vx_mean": float(vels[:, 0].mean()),
        "vx_err": float(abs(vels[:, 0].mean() - command[0])),
        "pelvis_h_mean": float(heights.mean()),
        "pelvis_h_std": float(heights.std()),
        "steps_per_s": [t / duration for t in touchdowns],
        "swing_fraction": swing_fraction,
        "params": str(params_path.relative_to(REPO) if params_path.is_relative_to(REPO)
                      else params_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="two or more run directories to compare")
    ap.add_argument("--command", nargs=3, type=float, default=[1.0, 0.0, 0.0])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true",
                    help="use each run's final weights instead of its best checkpoint")
    args = ap.parse_args()

    print(f"command vx={args.command[0]} vy={args.command[1]} wz={args.command[2]}, "
          f"{args.seconds}s, seed {args.seed}\n")

    results = {}
    for r in args.runs:
        d = pathlib.Path(r)
        if not d.is_absolute():
            d = REPO / d
        blob = json.loads((d / "config.json").read_text(encoding="utf-8"))
        overrides = blob.get("reward_overrides") or {}
        label = d.name
        results[label] = rollout(d, args.command, args.seconds, args.seed, not args.final)
        results[label]["overrides"] = (
            ", ".join(f"{t} {v['from']:g}->{v['to']:g}" for t, v in overrides.items()) or "none")

    rows = [
        ("reward overrides", lambda m: m["overrides"]),
        ("weights used", lambda m: m["params"]),
        ("survived (s)", lambda m: f"{m['survived_s']:.1f}" + ("  FELL" if m["fell"] else "")),
        ("vx tracked (m/s)", lambda m: f"{m['vx_mean']:.3f}"),
        ("vx error (m/s)", lambda m: f"{m['vx_err']:.3f}"),
        ("steps/s per foot", lambda m: " / ".join(f"{s:.2f}" for s in m["steps_per_s"])),
        ("swing fraction", lambda m: " / ".join(f"{s:.2f}" for s in m["swing_fraction"])),
        ("pelvis height (m)", lambda m: f"{m['pelvis_h_mean']:.3f} +/- {m['pelvis_h_std']:.3f}"),
    ]
    # Printed one column per run rather than one row: with three or more runs the rows get
    # unreadably wide, and the interesting comparison is down a metric, not across.
    names = list(results)
    for label, fn in rows:
        print(f"{label}")
        for n in names:
            print(f"    {n[-46:]:<48s} {fn(results[n])}")
        print()

    print("\nSwing fraction is the share of time a foot is off the ground; higher means longer\n"
          "swing phases. That is the quantity feet_air_time pays for, so it is where a change\n"
          "to that weight should show up first.")


if __name__ == "__main__":
    main()
