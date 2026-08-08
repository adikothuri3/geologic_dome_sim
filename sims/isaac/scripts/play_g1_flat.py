"""Roll out a trained Isaac G1 policy under a held velocity command and score the tracking.

    python sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id>
    python sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id> --command 0 0 1.0 --seconds 8
    python sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id> --sweep
    python sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id> --video

A reward curve says a policy improved. It does not say whether the robot realises the
velocity it was asked for, or whether it is falling forward in a way that scores well.
This holds one command for the whole clip and reports the error directly, which is the
number that transfers: `sims/mujoco/scripts/compare_gaits.py` produces the same
quantity for the MuJoCo policy, so an Isaac policy on flat ground can be compared
against the Phase-2 baseline rather than only against itself.

Reported per command, over the whole rollout after a settling window:

  vx / vy / wz MAE    mean |commanded - achieved|, in the robot's own frame. This is
                      what "follows velocity commands" means numerically.
  survival            fraction of envs still upright at the end. A policy with a low
                      MAE and a low survival rate is tracking well right up until it
                      falls, which the MAE alone will not show.
  act rate            mean |a_t - a_{t-1}| over joints and steps, in action units. This
                      is "jittery" as a number. The MuJoCo Phase-2 policy was visibly
                      jittery because Playground ships `action_rate` and `energy` scaled
                      to 0.0; Isaac's flat G1 keeps action_rate_l2 at -0.005, so the
                      claim that this gait is smoother is one a metric should settle
                      rather than the eye.
  joint vel           RMS joint velocity, rad/s. Buzzing at a joint that is not going
                      anywhere shows up here even when the action rate looks tame.

A per-step trace of commanded vs achieved velocity is written to `play_timeseries.json`
for plotting (see sims/isaac/scripts/plot_play.py).

The command is pinned every step. Upstream resamples mid-episode (and the PLAY cfg
already stretches the resample interval), but the command manager also zeroes standing
envs and can run a heading controller, so overwriting the buffer after each step is the
only way to guarantee the clip shows one commanded behaviour.

Video is opt-in (`--video`): it forces the offscreen renderer up, which costs several GB
on this 8 GB card. The numbers above need no renderer at all.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from isaaclab.app import AppLauncher  # pure-python; safe before app start

REPO = pathlib.Path(__file__).resolve().parents[3]
TASK = "Dome-G1FullCollision-Flat-DR-Play-v0"

# The rollout the MuJoCo baseline is scored on, so the two are comparable.
SWEEP = [
    ("forward 1.0 m/s", (1.0, 0.0, 0.0)),
    ("backward 0.5 m/s", (-0.5, 0.0, 0.0)),
    ("strafe 0.5 m/s", (0.0, 0.5, 0.0)),
    ("turn 1.0 rad/s", (0.0, 0.0, 1.0)),
    ("stand still", (0.0, 0.0, 0.0)),
]

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run_dir", help="a runs/isaac/<run_id> directory holding model_*.pt")
parser.add_argument("--checkpoint", default=None,
                    help="explicit .pt to load; default is the highest-numbered model_*.pt")
parser.add_argument("--command", nargs=3, type=float, default=None,
                    metavar=("VX", "VY", "WZ"),
                    help="command held for the whole clip (m/s, m/s, rad/s). Default: --sweep")
parser.add_argument("--sweep", action="store_true",
                    help="score the five-command sweep instead of a single command")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--settle", type=float, default=1.0,
                    help="seconds excluded from the error average while the gait starts up")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true",
                    help="also write an mp4 (forces the renderer up; several GB of VRAM)")
parser.add_argument("--task", default=TASK)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True                      # never open a viewport on this box
if args.video:
    args.enable_cameras = True            # offscreen render only

run_dir = pathlib.Path(args.run_dir)
if not run_dir.is_absolute():
    run_dir = REPO / run_dir
if not run_dir.is_dir():
    sys.exit(f"no such run directory: {run_dir}")

if args.checkpoint:
    ckpt = pathlib.Path(args.checkpoint)
else:
    # RSL-RL writes model_<iteration>.pt; sort numerically, because model_9 sorts after
    # model_1000 as a string and silently evaluating iteration 9 of a 1500-iteration run
    # would look like a policy that never learned.
    candidates = sorted(run_dir.glob("model_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not candidates:
        sys.exit(f"no model_*.pt checkpoints in {run_dir}")
    ckpt = candidates[-1]

cfg_blob = {}
cfg_path = run_dir / "config.json"
if cfg_path.exists():
    cfg_blob = json.loads(cfg_path.read_text(encoding="utf-8"))

# ---- start the app, then import the sim-side world ------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym            # noqa: E402
import torch                       # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, str(REPO / "sims" / "isaac" / "tasks"))
import dome_g1  # noqa: F401, E402  (registers the tasks)


def report(msg: str) -> None:
    """Kit hijacks python stdout on Windows; stderr survives."""
    print(msg, file=sys.stderr, flush=True)


spec = gym.spec(args.task)
env_cfg = spec.kwargs["env_cfg_entry_point"]()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
agent_cfg = spec.kwargs["rsl_rl_cfg_entry_point"]()
agent_cfg.seed = args.seed

report(f"run        {run_dir.name}")
report(f"commit     {cfg_blob.get('commit', '?')}")
report(f"trained on {cfg_blob.get('task', '?')}  ({cfg_blob.get('num_envs', '?')} envs, "
       f"{cfg_blob.get('max_iterations', '?')} iters)")
report(f"checkpoint {ckpt.name}")
report(f"eval task  {args.task}  ({args.num_envs} envs)")

env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
if args.video:
    video_dir = run_dir / "videos"
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=int(args.seconds / env_cfg.sim.dt / env_cfg.decimation),
        disable_logger=True,
    )
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(str(ckpt))
policy = runner.get_inference_policy(device=env.unwrapped.device)

dt = env.unwrapped.step_dt
n_steps = int(args.seconds / dt)
n_settle = int(args.settle / dt)
cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
robot = env.unwrapped.scene["robot"]


def roll_out(label: str, command: tuple[float, float, float]) -> dict:
    """Hold `command` for the whole clip; return per-channel MAE and survival."""
    target = torch.tensor(command, device=env.unwrapped.device).repeat(args.num_envs, 1)

    # The reset has to happen inside inference mode, not just the stepping. Once a
    # rollout has run under `torch.inference_mode()`, the sim's buffers are inference
    # tensors, and `reset_root_state_uniform` writing to them from normal mode raises
    # "Inplace update to inference tensor outside InferenceMode". Upstream's play script
    # never resets mid-loop, so it never meets this; a command sweep resets four times.
    with torch.inference_mode():
        env.reset()
    obs = env.get_observations()
    if isinstance(obs, tuple):  # older wrappers return (obs, extras)
        obs = obs[0]

    err_lin = torch.zeros(2, device=env.unwrapped.device)
    err_yaw = torch.zeros((), device=env.unwrapped.device)
    act_rate = torch.zeros((), device=env.unwrapped.device)
    joint_vel_sq = torch.zeros((), device=env.unwrapped.device)
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    scored = 0
    prev_actions = None
    trace = []

    for step in range(n_steps):
        # Pin before acting so the observation the policy sees carries the held command.
        cmd_term.vel_command_b[:] = target
        cmd_term.is_standing_env[:] = False
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        alive &= ~dones.bool()

        if step >= n_settle:
            # Compare in the base frame, which is the frame the command is given in.
            lin = robot.data.root_lin_vel_b[:, :2]
            yaw = robot.data.root_ang_vel_b[:, 2]
            m = alive
            if m.any():
                err_lin += (target[m, :2] - lin[m]).abs().mean(dim=0)
                err_yaw += (target[m, 2] - yaw[m]).abs().mean()
                if prev_actions is not None:
                    act_rate += (actions[m] - prev_actions[m]).abs().mean()
                joint_vel_sq += robot.data.joint_vel[m].pow(2).mean()
                scored += 1
                trace.append({
                    "t": round(step * dt, 3),
                    "vx": round(float(lin[m, 0].mean()), 4),
                    "vy": round(float(lin[m, 1].mean()), 4),
                    "wz": round(float(yaw[m].mean()), 4),
                })
        prev_actions = actions.clone()

    scored = max(scored, 1)
    return {
        "command": label,
        "target": list(command),
        "vx_mae": float(err_lin[0] / scored),
        "vy_mae": float(err_lin[1] / scored),
        "wz_mae": float(err_yaw / scored),
        "survival": float(alive.float().mean()),
        "action_rate": float(act_rate / scored),
        "joint_vel_rms": float((joint_vel_sq / scored).sqrt()),
        "trace": trace,
    }


commands = SWEEP if (args.sweep or args.command is None) else [
    (f"vx={args.command[0]} vy={args.command[1]} wz={args.command[2]}", tuple(args.command))
]

report("")
report(f"{'command':<20} {'vx MAE':>8} {'vy MAE':>8} {'wz MAE':>8} {'survival':>9} "
       f"{'act rate':>9} {'joint vel':>10}")
report("-" * 79)
results = []
for label, command in commands:
    r = roll_out(label, command)  # resets internally, under inference mode
    results.append(r)
    report(f"{r['command']:<20} {r['vx_mae']:>8.3f} {r['vy_mae']:>8.3f} "
           f"{r['wz_mae']:>8.3f} {r['survival']:>8.0%} {r['action_rate']:>9.4f} "
           f"{r['joint_vel_rms']:>10.3f}")

meta = {
    "checkpoint": ckpt.name,
    "run_id": run_dir.name,
    "eval_task": args.task,
    "num_envs": args.num_envs,
    "seconds": args.seconds,
    "settle_s": args.settle,
    "trained_iterations": cfg_blob.get("max_iterations"),
    "trained_num_envs": cfg_blob.get("num_envs"),
}
# Two files on purpose: the metrics summary stays small enough to read in a terminal
# and paste into notes/experiments.md, while the per-step traces (thousands of rows)
# go next door where only the plotter looks at them.
(run_dir / "play_metrics.json").write_text(json.dumps(
    {**meta, "results": [{k: v for k, v in r.items() if k != "trace"} for r in results]},
    indent=2), encoding="utf-8")
(run_dir / "play_timeseries.json").write_text(json.dumps(
    {**meta, "dt": dt, "results": results}, indent=2), encoding="utf-8")
report("")
report(f"wrote {(run_dir / 'play_metrics.json').relative_to(REPO)}"
       f" and play_timeseries.json")
if args.video:
    report(f"video under {(run_dir / 'videos').relative_to(REPO)}")

# Written before teardown: Kit shutdown on Windows can die with a native access
# violation that no finally block survives (notes/setup.md).
try:
    env.close()
finally:
    simulation_app.close()
