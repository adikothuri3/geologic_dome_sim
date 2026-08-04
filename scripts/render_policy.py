"""Roll out a trained G1 policy and write an MP4.

    python scripts/render_policy.py runs/<run_id>
    python scripts/render_policy.py runs/<run_id> --command 0 0 1.0 --seconds 8

A reward curve says a policy improved. It does not say whether the gait looks like walking or
like a controlled fall that happens to track velocity well. This renders the thing itself.

The joystick command is held fixed for the whole clip (upstream resamples it mid-episode,
which makes for a confusing video), so `--command 1 0 0` is "walk forward at 1 m/s".

Video is written with imageio + imageio-ffmpeg rather than mediapy, so one code path covers
Windows and Linux -- see notes/decisions.md.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import sys

# Both must be set before JAX/MuJoCo initialise their backends.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

ENV_NAME = "G1JoystickFlatTerrain"
FPS = 25


def build_env(collision: str):
    """Rebuild the environment the run was trained in -- collision model included."""
    from mujoco_playground import registry
    from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

    cfg = registry.get_default_config(ENV_NAME)
    if collision == "full-collision":
        import train_g1
        cfg.njmax = train_g1.NJMAX_FULL_COLLISION
        task = train_g1.sync_full_collision_scene()
        return g1_joystick.Joystick(task=task, config=cfg)
    import train_g1
    cfg.njmax = train_g1.NJMAX_FEET_ONLY
    return registry.load(ENV_NAME, config=cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a directory under runs/ holding params/ and config.json")
    ap.add_argument("--command", nargs=3, type=float, default=[1.0, 0.0, 0.0],
                    metavar=("VX", "VY", "WZ"),
                    help="joystick command held for the whole clip (m/s, m/s, rad/s)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--camera", default="track")
    ap.add_argument("--out", default=None)
    ap.add_argument("--best", action="store_true",
                    help="render the run's best-scoring checkpoint instead of its final weights")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir

    params_path = (run_dir / "params").resolve()
    if args.best:
        # Runs so far peak near 80M steps and drift down afterwards, so the final weights are
        # not the best weights -- see notes/experiments.md.
        best_meta = run_dir / "best.json"
        if not best_meta.exists():
            sys.exit(f"{best_meta} not found; this run predates per-eval checkpointing")
        meta = json.loads(best_meta.read_text(encoding="utf-8"))
        candidate = REPO / meta["checkpoint"]
        if not candidate.exists():
            sys.exit(f"best checkpoint missing: {candidate}")
        params_path = candidate.resolve()
        print(f"using BEST checkpoint: step {meta['best_step']:,} "
              f"(reward {meta['best_reward']:.2f} vs final {meta['final_reward']:.2f})")
    if not params_path.exists():
        sys.exit(f"no params at {params_path}")

    cfg_path = run_dir / "config.json"
    blob = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    collision = blob.get("collision", "feet-only")
    print(f"run        {run_dir.name}")
    print(f"commit     {blob.get('commit', '?')}")
    print(f"collision  {collision}")
    print(f"command    vx={args.command[0]} vy={args.command[1]} wz={args.command[2]}")

    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from etils import epath
    from mujoco_playground.config import locomotion_params
    from orbax import checkpoint as ocp

    env = build_env(collision)

    # Rebuild the exact network the run trained, then load its weights.
    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
    factory_kwargs = dict(ppo_params.get("network_factory", {}))
    network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **factory_kwargs,
    )
    make_policy = ppo_networks.make_inference_fn(network)

    params = ocp.PyTreeCheckpointer().restore(epath.Path(params_path))
    if not isinstance(params, (list, tuple)) or len(params) < 2:
        sys.exit(f"unexpected params structure: {type(params)}")

    # Orbax restores plain dicts, but the policy expects brax's RunningStatisticsState. Rather
    # than hand-rebuild that dataclass -- its saved `count` is serialised as {hi, lo} and it
    # has a `mode` field the checkpoint predates -- start from a fresh state of the right type
    # and graft on the statistics. Only mean/std/summed_variance affect inference; `count`
    # matters solely for further online updates, which a replay does not do.
    from brax.training.acme import specs

    normalizer = params[0]
    if isinstance(normalizer, dict):
        template = running_statistics.init_state(
            {k: specs.Array(tuple(v), jp.dtype("float32"))
             for k, v in env.observation_size.items()}
        )
        normalizer = template.replace(
            mean=normalizer["mean"],
            std=normalizer["std"],
            summed_variance=normalizer["summed_variance"],
        )

    inference_fn = jax.jit(make_policy((normalizer, params[1]), deterministic=True))

    command = jp.array(args.command, dtype=jp.float32)
    n_steps = int(args.seconds / env.dt)
    print(f"rolling out {n_steps} steps ({args.seconds}s at dt={env.dt})")

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    state = reset(rng)
    state.info["command"] = command

    trajectory, rewards = [], []
    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step(state, action)
        # Upstream resamples the command mid-episode; pin it so the clip shows one behaviour.
        state.info["command"] = command
        trajectory.append(state)
        rewards.append(float(state.reward))
        if bool(state.done):
            print(f"  episode terminated at step {i} ({i * env.dt:.1f}s) -- robot fell")
            break

    print(f"  mean reward/step {np.mean(rewards):+.4f}   steps rendered {len(trajectory)}")

    frames = env.render(trajectory, height=args.height, width=args.width, camera=args.camera)

    out = pathlib.Path(args.out) if args.out else REPO / f"{run_dir.name}.mp4"
    import imageio.v3 as iio
    iio.imwrite(out, np.asarray(frames), fps=FPS, codec="libx264", quality=8)
    print(f"wrote {out}  ({len(frames)} frames at {FPS} fps)")


if __name__ == "__main__":
    main()
