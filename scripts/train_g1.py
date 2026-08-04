"""Phase 2: train the G1 joystick locomotion policy on this 8 GB box.

    python scripts/train_g1.py --smoke      # ~minutes, proves the loop runs end to end
    python scripts/train_g1.py              # the real baseline run

Implements the `training-run` skill's non-negotiables:
  * num_envs capped at 1024-2048 (upstream default 8192 OOMs on a 4060 Ti)
  * the code that runs must be committed; the short hash is recorded
  * config captured alongside checkpoints
  * one row appended to notes/experiments.md afterwards -- success OR failure

Scaling rule: brax PPO wants `batch_size * num_minibatches == num_envs`
(upstream: 256 * 32 == 8192). We hold num_minibatches at 32 and derive batch_size,
so the gradient maths stays identical -- only the parallelism shrinks.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

# Set before JAX initialises the GPU backend (it reads these at first device use, and this
# module imports jax lazily inside main() so the ordering holds). The 4060 Ti also drives the
# Windows desktop, so JAX's default 75% preallocation asks for 6.0 GiB of 8 GiB and fails.
# Allocating on demand leaves the display its share; num_envs is the lever for the rest.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
EXPERIMENTS = REPO / "notes" / "experiments.md"

ENV_NAME = "G1JoystickFlatTerrain"
NUM_MINIBATCHES = 32

# Our full-collision scene. Playground resolves <include> and mesh paths relative to the
# top-level scene file, so this has to be executed from inside its own xmls/ directory --
# a copy elsewhere loads without meshes. sim/ holds the source of truth; sync_full_collision_scene()
# copies it into place.
FULL_COLLISION_SCENE = REPO / "sim" / "scene_g1_full_collision.xml"
FULL_COLLISION_BODY = REPO / "sim" / "g1_full_collision.xml"
FULL_COLLISION_TASK = "full_collision"

# Constraint rows per env. Upstream sets njmax = 29*2 + 8*4 = 90 and the feet-only baseline
# already logged `nefc overflow - please increase njmax to 93`, meaning constraints were being
# silently dropped. Adding six ground-contact pairs makes that far worse, so both modes get a
# raised value; it is a per-env buffer, so this costs VRAM linearly in num_envs.
NJMAX_FEET_ONLY = 128
NJMAX_FULL_COLLISION = 384


def sync_full_collision_scene():
    """Copy our scene next to Playground's, and teach the env how to find it.

    Returns the task name to construct Joystick with.
    """
    import shutil

    from mujoco_playground._src.locomotion.g1 import g1_constants as consts

    xmls_dir = consts.task_to_xml("flat_terrain").parent
    if not FULL_COLLISION_BODY.exists():
        sys.exit(f"{rel(FULL_COLLISION_BODY)} is missing. "
                 "Generate it with: python scripts/make_full_collision_xml.py")
    shutil.copyfile(FULL_COLLISION_BODY, xmls_dir / FULL_COLLISION_BODY.name)
    dest = xmls_dir / FULL_COLLISION_SCENE.name
    shutil.copyfile(FULL_COLLISION_SCENE, dest)

    original = consts.task_to_xml

    def task_to_xml(task_name):
        if task_name == FULL_COLLISION_TASK:
            return dest
        return original(task_name)

    consts.task_to_xml = task_to_xml
    print(f"full-collision scene -> {dest}")
    return FULL_COLLISION_TASK


def rel(p: pathlib.Path) -> str:
    """Repo-relative path for display, falling back to absolute rather than raising.

    This is called from the `finally` block after training; an exception here would mask
    the actual run outcome.
    """
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def git_commit(allow_dirty: bool) -> str:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty and not allow_dirty:
        sys.exit(
            "Working tree is dirty. The `training-run` skill requires the code that runs to be\n"
            "committed so the experiments.md row points at reproducible code.\n"
            "Commit or stash first, or pass --allow-dirty for a throwaway run.\n\n"
            f"{dirty}"
        )
    return head + ("-dirty" if dirty else "")


def append_experiment_row(row: dict) -> None:
    """Append one row to the table in notes/experiments.md. Never edits existing rows."""
    text = EXPERIMENTS.read_text(encoding="utf-8")
    lines = text.splitlines()

    sep = next((i for i, l in enumerate(lines) if l.strip().startswith("| ---")), None)
    if sep is None:
        print("WARN: could not find the table in experiments.md; row not written")
        return

    last = sep
    for i in range(sep + 1, len(lines)):
        if lines[i].lstrip().startswith("|"):
            last = i
        elif lines[i].strip():
            break

    new = "| {run_id} | {commit} | {config} | {n_envs} | {metrics} | {takeaway} |".format(**row)
    lines.insert(last + 1, new)

    # The placeholder note stops being true the moment a real row exists.
    lines = [l for l in lines if not l.startswith("*No runs yet")]

    out, done = [], False
    for l in lines:
        if not done and l.startswith("updated:"):
            l, done = f"updated: {datetime.now().astimezone().strftime('%Y-%m-%d')}", True
        out.append(l)

    EXPERIMENTS.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"appended row to {rel(EXPERIMENTS)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to prove the loop works (few envs, few steps)")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--full-collision", action="store_true",
                    help="collide shins, thighs and hands with the ground, not just feet")
    ap.add_argument("--njmax", type=int, default=None,
                    help="constraint rows per env; overrides the mode default")
    args = ap.parse_args()

    commit = git_commit(args.allow_dirty or args.smoke)

    if args.smoke:
        num_envs = args.num_envs or 256
        num_timesteps = args.num_timesteps or 2_000_000
        num_evals = 2
    else:
        # Skill cap for this box. 2048 is a quarter of upstream's 8192.
        num_envs = args.num_envs or 2048
        # Upstream's default. The 100M feet-only baseline peaked at 78M (reward 4.53) and then
        # drifted back to 2.96, so the extra budget is about stability, not just more steps.
        num_timesteps = args.num_timesteps or 200_000_000
        num_evals = 20

    if num_envs % NUM_MINIBATCHES != 0:
        sys.exit(f"num_envs ({num_envs}) must be divisible by num_minibatches ({NUM_MINIBATCHES})")
    batch_size = num_envs // NUM_MINIBATCHES

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params

    devices = jax.devices()
    if not any(d.platform == "gpu" for d in devices):
        sys.exit("No GPU visible to JAX. Run scripts/check_phase2.py first.")

    env_cfg = registry.get_default_config(ENV_NAME)
    env_cfg.njmax = args.njmax or (
        NJMAX_FULL_COLLISION if args.full_collision else NJMAX_FEET_ONLY
    )

    if args.full_collision:
        # Built directly rather than through registry.load, which is hardwired to the
        # feet-only task. Domain randomisation and the PPO config still come from the
        # registry entry, so only the collision model differs from the baseline.
        from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

        task = sync_full_collision_scene()
        env = g1_joystick.Joystick(task=task, config=env_cfg)
        eval_env = g1_joystick.Joystick(task=task, config=env_cfg)
    else:
        env = registry.load(ENV_NAME, config=env_cfg)
        eval_env = registry.load(ENV_NAME, config=env_cfg)

    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)

    training_params = dict(ppo_params)
    training_params.update(
        num_envs=num_envs,
        batch_size=batch_size,
        num_minibatches=NUM_MINIBATCHES,
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        seed=args.seed,
    )

    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )

    # Collision mode and step budget both go in the run_id. A full-collision run is a different
    # physics model, not a rerun, and a 100M run is not a shorter 200M one -- neither may land
    # in the other's directory or read as the same experiment. Date alone is not unique enough:
    # a same-day rerun previously collided, and force=True on the save makes that silent
    # clobbering rather than a loud error.
    collision = "full-collision" if args.full_collision else "feet-only"
    budget = f"{num_timesteps // 1_000_000}M"
    run_id = (f"{datetime.now().astimezone().strftime('%Y-%m-%d')}-g1-joystick-{collision}"
              f"-{budget}" + ("-smoke" if args.smoke else ""))
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config_blob = {
        "run_id": run_id,
        "commit": commit,
        "env": ENV_NAME,
        "collision": collision,
        "started": datetime.now(timezone.utc).isoformat(),
        "device": str(devices[0]),
        "ppo": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                for k, v in training_params.items()},
        "env_cfg": {"ctrl_dt": env_cfg.ctrl_dt, "sim_dt": env_cfg.sim_dt,
                    "episode_length": env_cfg.episode_length,
                    "njmax": int(env_cfg.njmax)},
    }
    (out_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    print(f"run_id     {run_id}")
    print(f"commit     {commit}")
    print(f"device     {devices[0]}")
    print(f"collision  {collision}  (njmax {int(env_cfg.njmax)})")
    print(f"num_envs   {num_envs}  (batch_size {batch_size} x num_minibatches {NUM_MINIBATCHES})")
    print(f"timesteps  {num_timesteps:,}")
    print(f"output     {rel(out_dir)}\n")

    history: list[dict] = []
    t0 = time.time()

    # Keep every evaluation's weights, not just the last.
    #
    # brax returns the FINAL params, and both runs so far peaked around 80M steps and then
    # drifted down -- the feet-only baseline ended 1.6 below its peak, the full-collision run
    # 4.6 below. Training longer therefore made the *saved* policy worse while the run looked
    # like it had simply plateaued. Checkpoints are a few MB each and there are only num_evals
    # of them, which is a trivial price for not throwing away the best policy of the run.
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_policy(current_step, make_policy, params):
        del make_policy
        from etils import epath
        from orbax import checkpoint as ocp
        path = epath.Path((ckpt_dir / f"step_{int(current_step):012d}").resolve())
        ocp.PyTreeCheckpointer().save(path, params, force=True)

    def progress(num_steps, metrics):
        reward = float(metrics.get("eval/episode_reward", float("nan")))
        std = float(metrics.get("eval/episode_reward_std", float("nan")))
        elapsed = time.time() - t0
        history.append({"steps": int(num_steps), "reward": reward, "std": std,
                        "elapsed_s": round(elapsed, 1)})
        print(f"  steps {int(num_steps):>12,}  reward {reward:8.3f} +/- {std:6.3f}  "
              f"[{elapsed / 60:.1f} min]", flush=True)
        (out_dir / "progress.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        randomization_fn=registry.get_domain_randomizer(ENV_NAME),
        progress_fn=progress,
        policy_params_fn=save_policy,
    )

    status, takeaway = "ok", ""
    try:
        make_inference_fn, params, _ = train_fn(
            environment=env,
            eval_env=eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        from etils import epath
        from orbax import checkpoint as ocp
        # force=True: run_id is date-based, so a second run on the same day reuses this path.
        # Without it orbax raises *after* training finishes and the whole run is lost to a
        # save error -- which is exactly what happened on the first checkpointing smoke test.
        ocp.PyTreeCheckpointer().save(
            epath.Path(out_dir / "params").resolve(), params, force=True)
        print(f"\nsaved params to {out_dir / 'params'}")
    except Exception as exc:  # a failed run is still a logged run
        status = "FAILED"
        takeaway = f"{type(exc).__name__}: {exc}"[:200]
        print(f"\nTRAINING FAILED: {takeaway}", file=sys.stderr)
    finally:
        wall_min = (time.time() - t0) / 60
        best = max((h["reward"] for h in history), default=float("nan"))
        final = history[-1]["reward"] if history else float("nan")

        # Record which checkpoint was actually the good one, so nobody has to re-derive it
        # from progress.json to render or fine-tune the best policy.
        if history:
            best_entry = max(history, key=lambda h: h["reward"])
            best_ckpt = ckpt_dir / f"step_{int(best_entry['steps']):012d}"
            (out_dir / "best.json").write_text(json.dumps({
                "best_step": best_entry["steps"],
                "best_reward": best_entry["reward"],
                "final_reward": final,
                "checkpoint": rel(best_ckpt),
                "exists": best_ckpt.exists(),
            }, indent=2), encoding="utf-8")
            print(f"best eval was {best_entry['reward']:.2f} at step {best_entry['steps']:,} "
                  f"-> {rel(best_ckpt)}")
        if status == "ok":
            takeaway = ("smoke test only, not a usable policy" if args.smoke
                        else f"{collision} at {num_envs} envs; reward {final:.1f}")
        append_experiment_row({
            "run_id": run_id,
            "commit": commit,
            "config": f"{ENV_NAME}, **{collision}**, PPO, bs={batch_size}, "
                      f"nmb={NUM_MINIBATCHES}, njmax={int(env_cfg.njmax)}, "
                      f"steps={num_timesteps:,}, seed={args.seed}",
            "n_envs": num_envs,
            "metrics": f"final eval reward {final:.2f}, best {best:.2f}, {wall_min:.0f} min"
                       + ("" if status == "ok" else f" — {status}"),
            "takeaway": takeaway,
        })
        (out_dir / "progress.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
