"""Phase 4a: train the full-body-collision G1 to follow velocity commands, in Isaac Lab.

    python sims/isaac/scripts/train_g1_flat.py --smoke        # 10 iterations, proves the loop
    python sims/isaac/scripts/train_g1_flat.py                # local run (small; real runs -> cloud)
    python sims/isaac/scripts/train_g1_flat.py --variant baseline   # the no-DR A/B control

    # the real run -- a rented GPU with >=24 GB, see "Cloud" below
    python sims/isaac/scripts/train_g1_flat.py --num_envs 4096 --max_iterations 1500

    # continue an interrupted run to the same absolute iteration target
    python sims/isaac/scripts/train_g1_flat.py --resume runs/isaac/<run_id> --max_iterations 1500

Implements the `training-run` skill's non-negotiables:
  * the code that runs must be committed; the short hash is recorded
  * config captured alongside checkpoints (config.json in the run dir)
  * one row appended to notes/experiments.md afterwards -- success OR failure
  * hardware limits respected: this box is below Isaac's minimum spec, so the local
    default is num_envs=256 headless; anything bigger belongs on a cloud GPU.

Tasks (sims/isaac/tasks/dome_g1) -- both are upstream's flat velocity task with G1_CFG
(full collision) swapped in for G1_MINIMAL_CFG:

  dr        Dome-G1FullCollision-Flat-DR-v0   default. Velocity-command tracking under
            the Phase-2 domain randomization set (friction, link/torso mass, armature,
            initial pose, pushes) and a direct 3-channel joystick command. The Isaac
            counterpart of the MuJoCo policy in notes/locomotion-policy.md.
  baseline  Dome-G1FullCollision-Flat-v0      upstream's config as-shipped, which
            randomizes nothing but observations. Kept so the cost of DR is measurable
            rather than assumed.

Run `check_isaac.py --gate c` before spending cloud money on the DR variant: it asserts
the randomization actually reaches PhysX.

Logs + checkpoints land under runs/isaac/<run_id>/ (RSL-RL writes model_*.pt there).

Cloud
-----
This box (8 GB VRAM) caps out at 256 envs, which is 9.2M samples over 1500 iterations
against upstream's 147M at 4096 — 16x less, and measured to be the wrong side of the
line: the 2026-08-08 local run reached 100% survival with velocity tracking still at
zero (see notes/experiments.md). The real run wants >=24 GB. On a fresh cloud box:

    git clone <repo> && cd GeologicDome
    pwsh sims/isaac/setup_isaac.ps1                # or the Linux equivalent install
    export OMNI_KIT_ACCEPT_EULA=YES
    python sims/isaac/scripts/check_isaac.py --gate c --num_envs 32   # DR is live?
    python sims/isaac/scripts/train_g1_flat.py --num_envs 4096 --max_iterations 1500
    python sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id> --video

Nothing here is local-only: paths are repo-relative, the app is always headless, and
the experiments row is written before teardown either way. Budget from the local rate
(1.85 s/iteration at 256 envs) knowing a 24 GB card runs more envs per second, not
fewer — 1500 iterations at 4096 envs is hours, not tens of minutes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

from isaaclab.app import AppLauncher  # pure-python; safe before app start

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNS = REPO / "runs" / "isaac"
EXPERIMENTS = REPO / "notes" / "experiments.md"
VARIANTS = {
    "dr": ("Dome-G1FullCollision-Flat-DR-v0", "g1fc-flat-dr"),
    "baseline": ("Dome-G1FullCollision-Flat-v0", "g1fc-flat"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=sorted(VARIANTS), default="dr",
                    help="dr: velocity tracking under domain randomization (default). "
                         "baseline: upstream's config, no dynamics randomization.")
parser.add_argument("--num_envs", type=int, default=256,
                    help="local ceiling on 8 GB; real training on cloud uses 4096")
parser.add_argument("--max_iterations", type=int, default=300,
                    help="ABSOLUTE iteration target, not a delta — with --resume, training "
                         "stops at this iteration rather than running this many more")
parser.add_argument("--resume", default=None, metavar="RUN_DIR_OR_CKPT",
                    help="continue from a run directory (its highest model_*.pt) or an "
                         "explicit checkpoint. Weights, optimizer state and iteration "
                         "count are all restored.")
parser.add_argument("--smoke", action="store_true", help="64 envs, 10 iterations")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run_name", default=None)
parser.add_argument("--allow-dirty", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True  # non-negotiable on this box
if args.smoke:
    args.num_envs, args.max_iterations = 64, 10

TASK, VARIANT_SLUG = VARIANTS[args.variant]


def git_commit(allow_dirty: bool) -> str:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty and not allow_dirty:
        sys.exit("Working tree is dirty. The `training-run` skill requires committed code "
                 "so the experiments.md row points at something reproducible.\n"
                 "Commit or stash, or pass --allow-dirty for a throwaway run.\n\n" + dirty)
    return head + ("-dirty" if dirty else "")


def append_experiment_row(row: dict) -> None:
    """Append one row to notes/experiments.md. Never edits existing rows."""
    line = ("| {run_id} | {commit} | {config} | {n_envs} | {metrics} | {takeaway} |"
            .format(**row))
    text = EXPERIMENTS.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    EXPERIMENTS.write_text(text + line + "\n", encoding="utf-8", newline="\n")
    print(f"appended row to {EXPERIMENTS.relative_to(REPO)}")


def resolve_resume(spec: str) -> pathlib.Path:
    """Accept either a run directory or an explicit .pt, return the checkpoint."""
    p = pathlib.Path(spec)
    if not p.is_absolute():
        p = REPO / p
    if p.is_file():
        return p
    if not p.is_dir():
        sys.exit(f"--resume: no such run directory or checkpoint: {p}")
    # Sort numerically. model_9 sorts after model_1000 as a string, so a lexical max
    # would silently resume a 1500-iteration run from iteration 9.
    ckpts = sorted(p.glob("model_*.pt"), key=lambda q: int(q.stem.split("_")[1]))
    if not ckpts:
        sys.exit(f"--resume: no model_*.pt checkpoints in {p}")
    return ckpts[-1]


resume_ckpt = resolve_resume(args.resume) if args.resume else None

commit = git_commit(args.allow_dirty)
slug = args.run_name or (f"{VARIANT_SLUG}-smoke" if args.smoke else VARIANT_SLUG)
if resume_ckpt and not args.run_name:
    # A resumed run gets its own directory and its own experiments row: the row IDs are
    # the join key for the whole log and rows are never deleted, so reusing the original
    # run's ID would leave two different results wearing one name.
    slug += "-resumed"
run_id = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-isaac-{slug}"
run_dir = RUNS / run_id
run_dir.mkdir(parents=True, exist_ok=True)

# ---- start the app, then import the sim-side world ------------------------------
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym            # noqa: E402
import torch                       # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, str(REPO / "sims" / "isaac" / "tasks"))
import dome_g1  # noqa: F401, E402  (registers the tasks)

# The gym registry is the single source of truth for which cfg pairs with which task —
# looking the entry points up here means adding a variant is one dict entry above and
# one gym.register(), with no third place to keep in sync.
spec = gym.spec(TASK)
env_cfg = spec.kwargs["env_cfg_entry_point"]()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed

agent_cfg = spec.kwargs["rsl_rl_cfg_entry_point"]()
agent_cfg.max_iterations = args.max_iterations
agent_cfg.seed = args.seed

config = {
    "task": TASK, "variant": args.variant, "commit": commit, "seed": args.seed,
    "num_envs": args.num_envs, "max_iterations": args.max_iterations,
    "robot": "G1_CFG (full collision)", "smoke": args.smoke,
    "domain_randomization": sorted(env_cfg.events.__dict__) if args.variant == "dr" else "none (upstream defaults)",
    "resumed_from": str(resume_ckpt.relative_to(REPO)) if resume_ckpt else None,
    "command_ranges": {
        "lin_vel_x": list(env_cfg.commands.base_velocity.ranges.lin_vel_x),
        "lin_vel_y": list(env_cfg.commands.base_velocity.ranges.lin_vel_y),
        "ang_vel_z": list(env_cfg.commands.base_velocity.ranges.ang_vel_z),
        "heading_command": env_cfg.commands.base_velocity.heading_command,
    },
    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
(run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

def write_row(status: str, metrics_txt: str) -> None:
    if args.smoke:
        took = "smoke test only, not a usable policy"
    elif args.num_envs <= 256:
        # 256 envs x 1500 iters is 9.2M samples against upstream's 147M at 4096. The
        # 2026-08-08 run at this size reached 100% survival with tracking still at zero,
        # so the caveat is measured on this task, not inherited boilerplate.
        took = (f"{args.num_envs}-env run — this is the local ceiling on 8 GB and ~16x under "
                f"upstream's sample count; score it with play_g1_flat.py before believing it")
    else:
        took = f"full-size run at {args.num_envs} envs — score with play_g1_flat.py"
    dr = ("**DR on** (friction, link+torso mass, CoM, armature, ±0.05 rad pose, pushes 5–10 s), "
          "direct 3-channel command vx±1.0/vy±0.5/wz±1.0"
          if args.variant == "dr" else "**no dynamics DR** (upstream G1 defaults), vx 0…1 + heading control")
    append_experiment_row({
        "run_id": run_id, "commit": commit,
        "config": f"{TASK}, **full-collision G1_CFG**, {dr}, RSL-RL PPO, "
                  f"iters={args.max_iterations}, seed={args.seed}",
        "n_envs": args.num_envs,
        "metrics": metrics_txt,
        "takeaway": took if status == "ok" else metrics_txt,
    })


t_train = time.time()
try:
    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir),
                            device=agent_cfg.device)

    start_iter = 0
    if resume_ckpt:
        runner.load(str(resume_ckpt))          # weights + optimizer + iteration count
        start_iter = runner.current_learning_iteration
        if start_iter >= args.max_iterations:
            # `raise SystemExit(msg)`, NOT `sys.exit(msg)`. Once the SimulationApp is up,
            # Kit has rebound `sys.exit` to a pybind11 binding that takes an int, so a
            # string argument raises TypeError — which the `except Exception` below then
            # dutifully records as a FAILED training run. Raising the exception directly
            # keeps a user error a user error. (Above the app start, sys.exit is fine.)
            raise SystemExit(
                f"--resume: checkpoint is already at iteration {start_iter}, which is "
                f"at or past --max_iterations {args.max_iterations}. Nothing to do; "
                f"raise the target to train further.")
        print(f"resumed from {resume_ckpt.name} at iteration {start_iter}", file=sys.stderr)

    # RSL-RL's `learn` takes a COUNT and runs [start_iter, start_iter + count), so on a
    # resume the count has to be the remainder. Treating --max_iterations as an absolute
    # target keeps "train to 1500" meaning the same thing whether or not it took two goes.
    n_iters = args.max_iterations - start_iter
    runner.learn(num_learning_iterations=n_iters, init_at_random_ep_len=True)

    minutes = (time.time() - t_train) / 60
    resumed = f"resumed from iteration {start_iter}, " if resume_ckpt else ""
    metrics_txt = (f"{resumed}{n_iters} iters to {args.max_iterations}, {minutes:.0f} min, "
                   f"final ckpt model_{args.max_iterations - 1}.pt in runs/isaac/{run_id} "
                   f"(curves: tensorboard --logdir runs/isaac/{run_id})")
except SystemExit:
    raise                                       # an explicit sys.exit is not a run failure
except Exception as e:  # noqa: BLE001 — the row must be written no matter what
    write_row("FAILED", f"FAILED — {type(e).__name__}: {e}")
    raise

# The row is written BEFORE teardown: on Windows, Kit shutdown (env.close /
# simulation_app.close) can die with a native access violation that no finally
# block survives. The run's result must already be on disk by then.
write_row("ok", metrics_txt)
print(f"done: ok  ({(time.time() - t0) / 60:.1f} min total)", file=sys.stderr, flush=True)

try:
    env.close()
finally:
    simulation_app.close()
