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

Tasks (sims/isaac/tasks/dome_g1) -- all are upstream's flat velocity task with G1_CFG
(full collision) swapped in for G1_MINIMAL_CFG:

  dr        Dome-G1FullCollision-Flat-DR-v0   default. Velocity-command tracking under
            the Phase-2 domain randomization set (friction, link/torso mass, armature,
            initial pose, pushes) and a direct 3-channel joystick command. The Isaac
            counterpart of the MuJoCo policy in notes/locomotion-policy.md.
  heading   Dome-G1FullCollision-Flat-Heading-v0  the positive control: the same DR
            physics under UPSTREAM's task definition (heading-derived yaw, forward-only,
            upstream's feet_air_time gate). Not a candidate policy -- an instrument for
            telling "our task is hard" apart from "our harness is broken".
  baseline  Dome-G1FullCollision-Flat-v0      upstream's config as-shipped, which
            randomizes nothing but observations. Kept so the cost of DR is measurable
            rather than assumed.

Run `check_isaac.py --gate c` before spending cloud money on the DR variant: it asserts
the randomization actually reaches PhysX.

Logs + checkpoints land under runs/isaac/<run_id>/ (RSL-RL writes model_*.pt there).

Guards (sims/isaac/scripts/train_guards.py) -- what the legacy MJX trainer had and this
one did not:

  progress.jsonl   one JSON line per iteration, appended. Survives a SIGKILL, which is
                   what an external kill and a Colab disconnect both are -- and which
                   previously cost a 1,792-iteration run its experiments row entirely.
  best.json +      the best checkpoint by `walk_score`, not by mean reward. Mean reward
  model_best.pt    rose across the whole of the failed 4096-env run, on the yaw term,
                   while the robot never stepped; `notes/locomotion-policy.md` states
                   plainly that it is not a progress signal for this task.
  --abort-if-flat  kills a run whose `feet_air_time` stays flat past iteration 500. This
                   automates the criterion in sims/isaac/README.md and turns a provably
                   dead 3-hour run into a 25-minute one.

Cloud
-----
The full walkthrough, including the Colab notebook, is in sims/isaac/README.md. In brief:

    bash sims/isaac/setup_isaac_cloud.sh
    export OMNI_KIT_ACCEPT_EULA=YES
    PY=~/venvs/isaac/bin/python
    $PY sims/isaac/scripts/check_isaac.py  --gate c --num_envs 32      # DR is live?
    $PY sims/isaac/scripts/train_g1_flat.py --num_envs 4096 --max_iterations 3000
    $PY sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id> --checkpoint best --video

Nothing here is local-only: paths are repo-relative, the app is always headless, and the
experiments row is written before teardown either way.

Sizing. 4096 envs fits in 5.05 GB and runs at ~3.7 s/iteration on the 8 GB dev box
(notes/setup.md) — the "256 envs is the local ceiling" guess was wrong by 16x, and the
first Phase-4a run paid for it with 9.2M samples against upstream's 147M. Cloud is now
about wall-clock and the renderer, not about whether the run fits.
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
# variant -> (training task, eval task for play_g1_flat.py, run-id slug)
VARIANTS = {
    "dr": ("Dome-G1FullCollision-Flat-DR-v0",
           "Dome-G1FullCollision-Flat-DR-Play-v0", "g1fc-flat-dr"),
    "heading": ("Dome-G1FullCollision-Flat-Heading-v0",
                "Dome-G1FullCollision-Flat-Heading-Play-v0", "g1fc-flat-heading"),
    "baseline": ("Dome-G1FullCollision-Flat-v0",
                 "Dome-G1FullCollision-Flat-DR-Play-v0", "g1fc-flat"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=sorted(VARIANTS), default="dr",
                    help="dr: velocity tracking under domain randomization (default). "
                         "heading: the same physics under upstream's task definition — "
                         "the positive control. baseline: upstream's config, no dynamics "
                         "randomization.")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="4096 is upstream's own count and fits in 5.05 GB (notes/setup.md)")
parser.add_argument("--max_iterations", type=int, default=3000,
                    help="ABSOLUTE iteration target, not a delta — with --resume, training "
                         "stops at this iteration rather than running this many more")
parser.add_argument("--resume", default=None, metavar="RUN_DIR_OR_CKPT",
                    help="continue from a run directory (its highest model_*.pt) or an "
                         "explicit checkpoint. Weights, optimizer state and iteration "
                         "count are all restored. Pair with --run_name to keep resuming "
                         "into the SAME directory, which an auto-retry loop needs.")
parser.add_argument("--save_interval", type=int, default=25,
                    help="iterations between checkpoints. 25 rather than upstream's 50 "
                         "because a Colab disconnect costs at most one interval")
parser.add_argument("--abort-if-flat", dest="abort_if_flat", type=int, default=500,
                    metavar="ITER",
                    help="abort if feet_air_time is still flat at this iteration (0 to "
                         "disable). Automates the criterion in sims/isaac/README.md")
parser.add_argument("--reward-scale", action="append", default=[], metavar="TERM=VALUE",
                    help="override one reward weight, e.g. action_rate_l2=-0.001 "
                         "(repeatable). The term must already exist — a typo would "
                         "otherwise create a dead entry that silently does nothing.")
parser.add_argument("--smoke", action="store_true", help="64 envs, 10 iterations")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run_name", default=None)
parser.add_argument("--allow-dirty", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True  # the RTX renderer is the thing that does not fit; never a viewport
if args.smoke:
    args.num_envs, args.max_iterations, args.save_interval = 64, 10, 5

TASK, PLAY_TASK, VARIANT_SLUG = VARIANTS[args.variant]

# Parsed before the app starts so a malformed spec costs nothing and can use plain
# sys.exit. The term NAMES cannot be checked until env_cfg exists; that happens below.
reward_overrides: dict[str, float] = {}
for spec in args.reward_scale:
    term, sep, value = spec.partition("=")
    if not sep:
        sys.exit(f"--reward-scale expects TERM=VALUE, got {spec!r}")
    try:
        reward_overrides[term.strip()] = float(value)
    except ValueError:
        sys.exit(f"--reward-scale expects a number, got {value!r} in {spec!r}")


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


def cell(value) -> str:
    """Make a value safe to put in a markdown table cell.

    Newlines become ' / ' and pipes are escaped. Both matter, and not hypothetically: on
    2026-08-08 a multi-line `TypeError` traceback went straight into a metrics cell and
    split the whole table across seven lines, and an unescaped `|v|=0.001` in an earlier
    row silently turned six cells into eight. Since the row is written from an exception
    message on the failure path, the one moment it must not corrupt the log is exactly the
    moment its content is least predictable.
    """
    text = " / ".join(str(value).splitlines()).strip()
    return text.replace("|", r"\|")


def append_experiment_row(row: dict) -> None:
    """Append one row to notes/experiments.md. Never edits existing rows."""
    line = "| " + " | ".join(cell(row[k]) for k in
                             ("run_id", "commit", "config", "n_envs", "metrics",
                              "takeaway")) + " |"
    text = EXPERIMENTS.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    EXPERIMENTS.write_text(text + line + "\n", encoding="utf-8", newline="\n")
    print(f"appended row to {EXPERIMENTS.relative_to(REPO)}")


def iteration_checkpoints(run_dir: pathlib.Path) -> list[pathlib.Path]:
    """`model_<iteration>.pt` in the directory, oldest first.

    Two filters, both load-bearing. **`model_best.pt` is excluded**: the guards write it
    into the same directory under the same glob, and `int("best")` raises — an unfiltered
    sort crashes every resume and every default eval. And the sort is **numeric**, because
    `model_9` sorts after `model_1000` as a string, so a lexical max would silently resume
    a 3000-iteration run from iteration 9.

    Duplicated verbatim in play_g1_flat.py rather than shared, because both call sites run
    *before* AppLauncher and the only natural home (train_guards.py) imports rsl_rl, which
    needs a running Kit.
    """
    numbered = []
    for p in run_dir.glob("model_*.pt"):
        tag = p.stem.split("_", 1)[1]
        if tag.isdigit():
            numbered.append((int(tag), p))
    return [p for _, p in sorted(numbered)]


def resolve_resume(spec: str) -> pathlib.Path:
    """Accept either a run directory or an explicit .pt, return the checkpoint."""
    p = pathlib.Path(spec)
    if not p.is_absolute():
        p = REPO / p
    if p.is_file():
        return p
    if not p.is_dir():
        sys.exit(f"--resume: no such run directory or checkpoint: {p}")
    ckpts = iteration_checkpoints(p)
    if not ckpts:
        sys.exit(f"--resume: no model_<iteration>.pt checkpoints in {p}")
    return ckpts[-1]


resume_ckpt = resolve_resume(args.resume) if args.resume else None

commit = git_commit(args.allow_dirty)
# A reward override is a different experiment, not a rerun — it gets its own run_id so the
# comparison against the unmodified task stays legible in experiments.md. Same convention
# as the legacy MJX trainer (sims/mujoco/scripts/train_g1.py).
tweak = "".join(f"-{t}{v:g}" for t, v in sorted(reward_overrides.items()))
slug = args.run_name or ((f"{VARIANT_SLUG}-smoke" if args.smoke else VARIANT_SLUG) + tweak)
if resume_ckpt and not args.run_name:
    # A resumed run gets its own directory and its own experiments row: the row IDs are
    # the join key for the whole log and rows are never deleted, so reusing the original
    # run's ID would leave two different results wearing one name.
    #
    # WITH --run_name, the directory is reused instead. That is what an auto-retry loop
    # needs — every attempt must land back in the same runs/isaac/<id>/ so `--resume` can
    # find the highest checkpoint. Each attempt still appends its own row, distinguished
    # by the "resumed from iteration N" it carries.
    slug += "-resumed"
run_id = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-isaac-{slug}"
run_dir = RUNS / run_id
run_dir.mkdir(parents=True, exist_ok=True)
# A resumed attempt lands in the same directory, so last attempt's verdict has to go before
# this attempt starts — otherwise a driver polling for it reads a stale "ok" and stops.
(run_dir / "outcome.json").unlink(missing_ok=True)

# ---- start the app, then import the sim-side world ------------------------------
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym            # noqa: E402
import torch                       # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_guards import CollapseAbort, DomeOnPolicyRunner  # noqa: E402

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
agent_cfg.save_interval = args.save_interval

# -- reward-weight overrides --------------------------------------------------------
# Applied here, after the whole __post_init__ chain, so an override lands on the final
# weight rather than one a subclass is about to rewrite. The term must already exist:
# a typo would otherwise sit in the config looking like an experiment while changing
# nothing, which is precisely the failure mode notes/decisions.md keeps running into.
applied_overrides = {}
for term, value in reward_overrides.items():
    rew_term = getattr(env_cfg.rewards, term, None)
    if rew_term is None or not hasattr(rew_term, "weight"):
        available = sorted(k for k, v in env_cfg.rewards.__dict__.items()
                           if hasattr(v, "weight"))
        # SystemExit, not sys.exit(str) — Kit has rebound sys.exit to a pybind11 binding
        # that takes an int, and a string argument raises TypeError (see below).
        raise SystemExit(f"--reward-scale: unknown reward term {term!r} on {TASK}.\n"
                         "Available:\n  " + "\n  ".join(available))
    applied_overrides[term] = {"from": float(rew_term.weight), "to": value}
    print(f"reward override: {term} {rew_term.weight} -> {value}", file=sys.stderr)
    rew_term.weight = value

config = {
    "task": TASK, "play_task": PLAY_TASK, "variant": args.variant, "commit": commit,
    "seed": args.seed,
    "num_envs": args.num_envs, "max_iterations": args.max_iterations,
    "save_interval": args.save_interval, "abort_if_flat": args.abort_if_flat or None,
    "robot": "G1_CFG (full collision)", "smoke": args.smoke,
    "domain_randomization": (sorted(env_cfg.events.__dict__)
                             if args.variant in ("dr", "heading")
                             else "none (upstream defaults)"),
    "reward_overrides": applied_overrides or None,
    "feet_air_time_func": getattr(env_cfg.rewards.feet_air_time.func, "__name__", "?"),
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

DR_BLURB = ("**DR on** (friction, link+torso mass, CoM, armature, ±0.05 rad pose, "
            "pushes 5–10 s)")
VARIANT_BLURB = {
    "dr": DR_BLURB + ", direct 3-channel command vx±1.0/vy±0.5/wz±1.0, "
                     "**`feet_air_time_joystick` gate**",
    "heading": DR_BLURB + ", **upstream task definition** (heading-derived yaw, vx 0…1, "
                          "upstream `feet_air_time` gate) — the positive control",
    "baseline": "**no dynamics DR** (upstream G1 defaults), vx 0…1 + heading control",
}


def guard_numbers(summary: dict) -> str:
    """The reward-term evidence, in the order that actually discriminates.

    Deliberately leads with feet_air_time and track_lin, not mean reward: in the failed
    4096-env run mean reward rose the entire time on the yaw term while the robot never
    stepped, so a row that leads with it misreports the run (notes/decisions.md).
    """
    if not summary:
        return ""

    def num(v, fmt="{:.4f}"):
        return fmt.format(v) if isinstance(v, (int, float)) else "?"

    walked = "**walks**" if summary.get("walked") else "**no stepping**"
    best = (f"best walk_score {num(summary.get('best_walk_score'), '{:.3f}')} at iteration "
            f"{summary.get('best_iteration')}" if summary.get("best_iteration") is not None
            else "no best checkpoint (run too short)")
    return (f" — {walked}: final feet_air_time {num(summary.get('final_feet_air_time'))}, "
            f"track_lin {num(summary.get('final_track_lin'))} "
            f"(stand-still scores 0.37), track_ang {num(summary.get('final_track_ang'))}, "
            f"mean reward {num(summary.get('final_mean_reward'), '{:.2f}')}, "
            f"lr {num(summary.get('final_learning_rate'), '{:.2e}')}. {best}")


def write_outcome(status: str, metrics_txt: str, summary: dict) -> None:
    """The machine-readable verdict, next to the checkpoints.

    Why a file and not an exit code: **the exit code of this script is not under its
    control.** `raise SystemExit(3)` after `simulation_app.close()` was measured on
    2026-08-08 to produce an exit code of **0** — Kit owns process shutdown and terminates
    the process itself, the same way it rebinds `sys.exit` to a pybind11 `post_quit()` that
    only takes an int (see the --resume guard below, and the `2026-08-08-isaac-guard-selftest`
    row). An automated driver therefore cannot distinguish "the watchdog stopped this run"
    from "training finished" by exit status, and a driver that guesses wrong resumes a
    policy that provably will not walk.

    Absent, this file means the process died without reaching its own teardown — a crash or
    a disconnect — which is exactly the case a retry loop should retry.
    """
    (run_dir / "outcome.json").write_text(json.dumps({
        "status": status,                    # ok | collapsed | FAILED
        "run_id": run_id, "commit": commit, "task": TASK,
        "play_task": PLAY_TASK, "variant": args.variant,
        "max_iterations": args.max_iterations, "num_envs": args.num_envs,
        "metrics": metrics_txt, **summary,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8", newline="\n")


def write_row(status: str, metrics_txt: str, summary: dict | None = None) -> None:
    summary = summary or {}
    write_outcome(status, metrics_txt, summary)
    if args.smoke:
        took = "smoke test only, not a usable policy"
    elif status == "collapsed":
        took = ("**negative result, not a crash** — the stepping reward never lifted, so "
                "this reward configuration does not produce locomotion at any sample count. "
                "Next lever: `--reward-scale action_rate_l2=-0.001`, paying gait smoothness "
                "for locomotion (notes/decisions.md, 2026-08-08)")
    elif status != "ok":
        took = metrics_txt
    elif summary.get("walked"):
        took = (f"full-size run at {args.num_envs} envs, **stepping reward lifted** — "
                f"score it with `play_g1_flat.py --checkpoint best` before believing it "
                f"walks; MAE equal to the command magnitude is the signature of zero motion")
    else:
        took = (f"ran to {args.max_iterations} without the stepping reward lifting — "
                f"score with `play_g1_flat.py --checkpoint best`, but expect MAE ≈ command "
                f"magnitude, which is what a non-walking policy scores")
    append_experiment_row({
        "run_id": run_id, "commit": commit,
        "config": f"{TASK}, **full-collision G1_CFG**, {VARIANT_BLURB[args.variant]}"
                  + (f", **reward overrides** {applied_overrides}" if applied_overrides else "")
                  + f", RSL-RL PPO, iters={args.max_iterations}, seed={args.seed}",
        "n_envs": args.num_envs,
        "metrics": metrics_txt + guard_numbers(summary),
        "takeaway": took,
    })


t_train = time.time()
runner = None
try:
    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = DomeOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir),
                                device=agent_cfg.device)
    # Both windows scale down on a short run. Not cosmetic: at their full sizes a 10-iteration
    # smoke can neither trip the watchdog nor write a best.json, so the two guards that most
    # need proving would be exactly the two a smoke test cannot exercise.
    runner.configure_guards(
        abort_if_flat=args.abort_if_flat or None,
        flat_window=min(200, max(2, args.max_iterations // 4)),
        best_warmup=min(100, max(1, args.max_iterations // 4)),
        run_dir=run_dir,
    )

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
    summary = runner.guard_summary()
    metrics_txt = (f"{resumed}{n_iters} iters to {args.max_iterations}, {minutes:.0f} min, "
                   f"final ckpt model_{args.max_iterations - 1}.pt in runs/isaac/{run_id} "
                   f"(curves: tensorboard --logdir runs/isaac/{run_id})")
    status = "ok"
except SystemExit:
    raise                                       # an explicit sys.exit is not a run failure
except CollapseAbort as e:
    # NOT a failure. The run answered its question — this reward configuration does not
    # produce locomotion — and that answer is exactly as loggable as a policy would be.
    # Recording it as FAILED would bury a result among crashes.
    minutes = (time.time() - t_train) / 60
    summary = runner.guard_summary() if runner is not None else {}
    metrics_txt = (f"**aborted by the {e.reason} watchdog** at iteration {e.iteration} "
                   f"({minutes:.0f} min): {e.detail}")
    status = "collapsed"
    write_row(status, metrics_txt, summary)
    print(f"\nABORTED: {e}", file=sys.stderr, flush=True)
except Exception as e:  # noqa: BLE001 — the row must be written no matter what
    summary = runner.guard_summary() if runner is not None else {}
    write_row("FAILED", f"FAILED — {type(e).__name__}: {e}", summary)
    raise
else:
    # The row is written BEFORE teardown: on Windows, Kit shutdown (env.close /
    # simulation_app.close) can die with a native access violation that no finally
    # block survives. The run's result must already be on disk by then.
    write_row(status, metrics_txt, summary)
    print(f"done: ok  ({(time.time() - t0) / 60:.1f} min total)", file=sys.stderr, flush=True)

try:
    env.close()
finally:
    simulation_app.close()
