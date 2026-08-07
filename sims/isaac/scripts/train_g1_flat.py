"""Phase 4a: train the full-body-collision G1 on a flat plane in Isaac Lab (RSL-RL PPO).

    python sims/isaac/scripts/train_g1_flat.py --smoke     # 10 iterations, proves the loop
    python sims/isaac/scripts/train_g1_flat.py             # local run (small; real runs -> cloud)

Implements the `training-run` skill's non-negotiables:
  * the code that runs must be committed; the short hash is recorded
  * config captured alongside checkpoints (config.json in the run dir)
  * one row appended to notes/experiments.md afterwards -- success OR failure
  * hardware limits respected: this box is below Isaac's minimum spec, so the local
    default is num_envs=256 headless; anything bigger belongs on a cloud GPU.

Task: Dome-G1FullCollision-Flat-v0 (sims/isaac/tasks/dome_g1) -- upstream's flat
velocity task with G1_CFG (full collision) swapped in for G1_MINIMAL_CFG.
Logs + checkpoints land under runs/isaac/<run_id>/ (RSL-RL writes model_*.pt there).
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
TASK = "Dome-G1FullCollision-Flat-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=256,
                    help="local ceiling on 8 GB; real training on cloud uses 4096")
parser.add_argument("--max_iterations", type=int, default=300)
parser.add_argument("--smoke", action="store_true", help="64 envs, 10 iterations")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run_name", default=None)
parser.add_argument("--allow-dirty", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True  # non-negotiable on this box
if args.smoke:
    args.num_envs, args.max_iterations = 64, 10


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


commit = git_commit(args.allow_dirty)
slug = args.run_name or ("g1fc-flat-smoke" if args.smoke else "g1fc-flat")
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
import dome_g1  # noqa: F401, E402  (registers the task)
from dome_g1.flat_env_cfg import DomeG1FlatPPORunnerCfg, G1FullCollisionFlatEnvCfg  # noqa: E402

env_cfg = G1FullCollisionFlatEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed

agent_cfg = DomeG1FlatPPORunnerCfg()
agent_cfg.max_iterations = args.max_iterations
agent_cfg.seed = args.seed

config = {
    "task": TASK, "commit": commit, "seed": args.seed,
    "num_envs": args.num_envs, "max_iterations": args.max_iterations,
    "robot": "G1_CFG (full collision)", "smoke": args.smoke,
    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
(run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

def write_row(status: str, metrics_txt: str) -> None:
    took = ("smoke test only, not a usable policy" if args.smoke else
            f"local Isaac run at {args.num_envs} envs; real training belongs on cloud")
    append_experiment_row({
        "run_id": run_id, "commit": commit,
        "config": f"{TASK}, **full-collision G1_CFG**, RSL-RL PPO, iters={args.max_iterations}, seed={args.seed}",
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
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    minutes = (time.time() - t_train) / 60
    metrics_txt = (f"{args.max_iterations} iters, {minutes:.0f} min, "
                   f"final ckpt model_{args.max_iterations - 1}.pt in runs/isaac/{run_id} "
                   f"(curves: tensorboard --logdir runs/isaac/{run_id})")
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
