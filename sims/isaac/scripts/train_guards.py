"""Collapse guards and best-checkpoint tracking for the Isaac RSL-RL trainer.

`DomeOnPolicyRunner` is `rsl_rl.runners.OnPolicyRunner` with one method overridden --
`log()`, which RSL-RL already calls once per iteration with `learn()`'s `locals()`.
Nothing about PPO, the rollout, or the checkpoint format changes.

Why this exists
---------------
The legacy MJX trainer (`sims/mujoco/scripts/train_g1.py`) checkpoints every evaluation and
writes a `best.json` naming the best one, because brax returns the FINAL parameters and both
2026-08-04 runs peaked near 80M steps and then drifted *down* -- the full-collision run ended
4.6 below its own peak. The Isaac trainer had no equivalent: RSL-RL saves every
`save_interval` iterations and the last one written is whatever the run happened to stop on.

> The metric is NOT mean reward, and that is the entire point.

`notes/locomotion-policy.md` says it outright: "Mean reward is not a progress signal for this
task." In run `2026-08-08-isaac-g1fc-flat-dr-4096` mean reward climbed -30.4 -> +4.11 over
1,792 iterations *entirely on the yaw term*, while `feet_air_time` sat at 0.005-0.010 from
iteration 199 to 1599 and the robot never took a step. Anyone tracking the headline number
would have called that run a success. So every guard here reads the individual reward terms
Isaac Lab publishes as `Episode_Reward/<term>` (see
`isaaclab/managers/reward_manager.py:120`), never the total.

    walk_score = Episode_Reward/track_lin_vel_xy_exp + Episode_Reward/track_ang_vel_z_exp
    gate       = Episode_Reward/feet_air_time >= FEET_AIR_TIME_MIN

`track_lin_vel_xy_exp` is the quantity that was pinned at **0.37 -- what standing still
scores** -- for 1,400 iterations of the failed run, and `feet_air_time` is the term that
distinguishes stepping from pivoting on the spot. Both are in the score for a reason: yaw
alone is collectable without a gait, which is exactly the trap that was fallen into.

Four behaviours
---------------
`progress.jsonl`   one JSON object appended per iteration. **Append-only on purpose**: a
                   Colab disconnect and an external `kill` are both SIGKILL, which no
                   `finally` block survives -- the lab notebook records a 4-hour sweep and a
                   whole 1,792-iteration run lost exactly that way, the latter leaving no
                   `experiments.md` row at all. A line already on disk cannot be lost.
`best.json` +      the best *gated* iteration's weights, MuJoCo parity. Ungated bests are
`model_best.pt`    tracked too, and superseded the moment a genuinely stepping iteration
                   appears; `best.json` carries `"gated": false` when the run never stepped,
                   so a non-walking policy cannot quietly present itself as the run's best.
collapse watchdog  automates `sims/isaac/README.md` -- "if you see that shape by iteration
                   ~500, kill it; more compute provably does not fix it". Turns a 3-hour dead
                   run into a ~25-minute one.
NaN guard          aborts on a NaN/inf mean reward or loss. The MJX track only ever
                   *recorded* NaN (`notes/experiments.md` row 1, "final eval reward nan").

The learning rate is recorded too but is not a guard. RSL-RL's PPO runs `schedule="adaptive"`
with `desired_kl=0.01`, halving or raising the LR to hold the KL in band, bounded to
[1e-5, 1e-2]. An LR pinned at the 1e-5 floor means the policy is fighting its own objective --
useful evidence in the log, but too indirect to abort on.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import statistics
import sys
import time

import torch
from rsl_rl.runners import OnPolicyRunner

# -- the reward terms the guards read -------------------------------------------------
# Isaac Lab's RewardManager publishes every active term as `Episode_Reward/<term>`, so
# these are the term names from `G1Rewards` / the base `RewardsCfg`, not our own labels.
TRACK_LIN = "Episode_Reward/track_lin_vel_xy_exp"
TRACK_ANG = "Episode_Reward/track_ang_vel_z_exp"
FEET_AIR_TIME = "Episode_Reward/feet_air_time"

# What standing perfectly still scores on `track_lin_vel_xy_exp`, measured in the failed
# 4096-env run: it plateaued at exactly this from iteration 200 onward. Recorded in
# best.json purely as context, so a 0.37 never reads as progress.
STAND_STILL_FLOOR = 0.37

# The stepping threshold. The failed run sat at 0.005-0.010 for 1,400 iterations; a healthy
# run is past ~0.05 and climbing by iteration 300-500 (sims/isaac/README.md). 0.02 sits
# clearly above the dead band and clearly below the healthy one.
FEET_AIR_TIME_MIN = 0.02

# Ignore the first N iterations for best-checkpoint purposes. Early scores are noise, and
# without this the run writes model_best.pt on nearly every one of the first iterations.
BEST_WARMUP_ITERS = 100


class CollapseAbort(RuntimeError):
    """Training converged to a non-walking behaviour, or went numerically bad.

    Deliberately NOT a subclass of the errors `train_g1_flat.py` reports as FAILED: a run
    that aborts here produced a *result* -- "this reward configuration does not produce
    locomotion" -- and gets a normal experiments row saying so.
    """

    def __init__(self, reason: str, iteration: int, detail: str) -> None:
        super().__init__(f"{reason} at iteration {iteration}: {detail}")
        self.reason = reason
        self.iteration = iteration
        self.detail = detail


def _finite(x: float) -> bool:
    return x is not None and math.isfinite(x)


class DomeOnPolicyRunner(OnPolicyRunner):
    """`OnPolicyRunner` plus progress logging, best-checkpoint tracking and abort criteria.

    Constructed exactly like the base class; the guards are configured afterwards through
    `configure_guards()` so the signature stays compatible with anything upstream does to
    `OnPolicyRunner.__init__`.
    """

    # -- configuration -----------------------------------------------------------------

    def configure_guards(
        self,
        *,
        abort_if_flat: int | None = 500,
        flat_window: int = 200,
        feet_air_time_min: float = FEET_AIR_TIME_MIN,
        best_warmup: int = BEST_WARMUP_ITERS,
        run_dir: str | os.PathLike | None = None,
    ) -> None:
        """
        abort_if_flat     first iteration at which the collapse watchdog may fire, or None
                          to disable it. The default of 500 is the README's own criterion.
        flat_window       how many trailing iterations must ALL be below the threshold. A
                          window, not an instant, because the term is noisy per iteration --
                          it is the *flatness over hundreds of iterations* that discriminates
                          "converged to the wrong thing" from "still learning".
        feet_air_time_min the stepping threshold.
        best_warmup       iterations to ignore for best-checkpoint purposes.
        run_dir           where progress.jsonl / best.json / model_best.pt go. Defaults to
                          the runner's own log_dir.
        """
        self._abort_if_flat = abort_if_flat
        self._flat_window = flat_window
        self._air_min = feet_air_time_min
        self._best_warmup = best_warmup

        self._dir = pathlib.Path(run_dir) if run_dir is not None else pathlib.Path(self.log_dir)
        self._progress_path = self._dir / "progress.jsonl"
        self._best_path = self._dir / "best.json"
        self._best_ckpt = self._dir / "model_best.pt"

        self._air_history: list[float] = []
        self._best_score = -math.inf
        self._best_gated = False       # True once the best came from a stepping iteration
        self._best_record: dict | None = None
        self._last_record: dict | None = None
        self._t0 = time.time()

        # -- carry the previous attempt's best across a resume ----------------------
        # Without this the best is forgotten on every restart, and the first improvement
        # after the warmup overwrites model_best.pt with whatever the resumed run happens
        # to find -- which is typically WORSE, since PPO dips on restart. On Colab that is
        # not an edge case: every disconnect resumes, so best-checkpoint tracking would be
        # defeated in exactly the environment it was built for. `_best_gated` is restored
        # too, so a run that has already learned to step cannot have its best replaced by
        # a non-stepping iteration.
        if self._best_path.exists() and self._best_ckpt.exists():
            try:
                prior = json.loads(self._best_path.read_text(encoding="utf-8"))
                self._best_score = float(prior["best_walk_score"])
                self._best_gated = bool(prior.get("gated", False))
                self._best_record = prior
                self.report(f"guards: carrying best walk_score {self._best_score:.4f} from "
                            f"iteration {prior.get('best_iteration')} "
                            f"(gated={self._best_gated})")
            except (ValueError, KeyError, TypeError) as exc:
                # A truncated best.json is a reason to start tracking afresh, not to refuse
                # to train -- the checkpoints themselves are untouched either way.
                self.report(f"guards: ignoring unreadable best.json ({exc})")

    # -- the one overridden method -----------------------------------------------------

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # Upstream first: it owns the TensorBoard writes and the terminal table, and it
        # normalises `ep_infos` entries to tensors, which the extraction below relies on.
        super().log(locs, width, pad)
        if not hasattr(self, "_air_history"):
            return                      # configure_guards() was never called; behave as base

        it = int(locs["it"])
        terms = self._episode_means(locs["ep_infos"])
        rew_buf, len_buf = locs.get("rewbuffer"), locs.get("lenbuffer")
        mean_reward = statistics.mean(rew_buf) if rew_buf else None
        mean_ep_len = statistics.mean(len_buf) if len_buf else None

        air = terms.get(FEET_AIR_TIME)
        lin = terms.get(TRACK_LIN)
        ang = terms.get(TRACK_ANG)
        walk_score = (lin + ang) if (_finite(lin) and _finite(ang)) else None
        gated = _finite(air) and air >= self._air_min

        record = {
            "iteration": it,
            "walk_score": walk_score,
            "track_lin_vel_xy_exp": lin,
            "track_ang_vel_z_exp": ang,
            "feet_air_time": air,
            "gated": gated,
            "mean_reward": mean_reward,
            "mean_episode_length": mean_ep_len,
            # The adaptive-KL schedule's current LR. Pinned at its 1e-5 floor is a sign the
            # policy is fighting its own objective -- evidence, not a trigger.
            "learning_rate": float(self.alg.learning_rate),
            "losses": {k: float(v) for k, v in locs.get("loss_dict", {}).items()},
            "reward_terms": terms,
            "elapsed_s": round(time.time() - self._t0, 1),
        }
        self._last_record = record
        self._append_progress(record)

        self._check_numerics(it, mean_reward, record["losses"])
        if walk_score is not None:
            self._track_best(it, walk_score, gated, record)
        if _finite(air):
            self._air_history.append(float(air))
            self._check_collapse(it)

    # -- internals ---------------------------------------------------------------------

    @staticmethod
    def _episode_means(ep_infos: list) -> dict[str, float]:
        """Mean of every episode-info key over the iteration, as plain floats.

        The same reduction upstream's `log()` does for TensorBoard, repeated here rather
        than scraped back out of the writer: reading our own numbers keeps the guards
        independent of which logger backend is configured.
        """
        if not ep_infos:
            return {}
        out: dict[str, float] = {}
        for key in ep_infos[0]:
            vals = []
            for info in ep_infos:
                if key not in info:
                    continue
                v = info[key]
                if isinstance(v, torch.Tensor):
                    vals.append(v.float().mean().item() if v.numel() else float("nan"))
                else:
                    vals.append(float(v))
            if vals:
                out[key] = float(sum(vals) / len(vals))
        return out

    def _append_progress(self, record: dict) -> None:
        # One line, opened and closed per iteration. Slower than holding a handle, and the
        # point: a SIGKILL between iterations leaves a complete file.
        with self._progress_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record) + "\n")

    def _check_numerics(self, it: int, mean_reward, losses: dict) -> None:
        bad = [k for k, v in losses.items() if not _finite(v)]
        if mean_reward is not None and not _finite(mean_reward):
            bad.append("mean_reward")
        if bad:
            raise CollapseAbort("nan", it, f"non-finite: {', '.join(sorted(bad))}")

    def _track_best(self, it: int, score: float, gated: bool, record: dict) -> None:
        if it < self._best_warmup:
            return
        if gated and not self._best_gated:
            # First stepping iteration. Every ungated best so far described a policy that
            # was not walking, so they stop counting rather than competing on score.
            self._best_score, self._best_gated = -math.inf, True
        if self._best_gated and not gated:
            return                      # once stepping, only stepping iterations qualify
        if score <= self._best_score:
            return

        self._best_score = score
        self.save(str(self._best_ckpt))
        self._best_record = {
            "best_iteration": it,
            "best_walk_score": round(score, 5),
            "gated": gated,
            "feet_air_time": record["feet_air_time"],
            "track_lin_vel_xy_exp": record["track_lin_vel_xy_exp"],
            "track_ang_vel_z_exp": record["track_ang_vel_z_exp"],
            "mean_reward_at_best": record["mean_reward"],
            "checkpoint": self._best_ckpt.name,
            # Copied from the MJX trainer's best.json: a self-verifying field, so a
            # best.json pointing at a checkpoint that never landed is detectable.
            "exists": self._best_ckpt.exists(),
            "stand_still_floor": STAND_STILL_FLOOR,
            "feet_air_time_gate": self._air_min,
            "note": (
                "walk_score = track_lin_vel_xy_exp + track_ang_vel_z_exp, chosen over mean "
                "reward because mean reward rose the whole of the failed 4096-env run on the "
                "yaw term alone. gated=false means feet_air_time never reached the stepping "
                "threshold -- this is the best checkpoint of a run that did not walk."
            ),
        }
        self._write_best()

    def _write_best(self) -> None:
        if self._best_record is None:
            return
        self._best_record["exists"] = self._best_ckpt.exists()
        self._best_path.write_text(
            json.dumps(self._best_record, indent=2), encoding="utf-8", newline="\n")

    def _check_collapse(self, it: int) -> None:
        if self._abort_if_flat is None or it < self._abort_if_flat:
            return
        window = self._air_history[-self._flat_window:]
        if len(window) < self._flat_window:
            # Not enough history yet. Reached on a --resume, whose in-memory history starts
            # empty: the watchdog then needs a full window of NEW iterations before it can
            # fire, which is the conservative direction.
            return
        peak = max(window)
        if peak < self._air_min:
            raise CollapseAbort(
                "feet_air_time flat", it,
                f"peak {peak:.4f} over the last {self._flat_window} iterations, below the "
                f"{self._air_min} stepping threshold. This is the shape of run "
                f"2026-08-08-isaac-g1fc-flat-dr-4096, which stayed flat for 1,400 further "
                f"iterations and never walked -- more compute provably does not fix it "
                f"(notes/decisions.md, 2026-08-08). Next lever: action_rate_l2, via "
                f"--reward-scale action_rate_l2=-0.001.")

    # -- what the trainer reports ------------------------------------------------------

    def guard_summary(self) -> dict:
        """Everything the experiments.md row needs. Safe to call after an abort.

        Returns `{}` rather than raising if `configure_guards()` never ran. The trainer
        calls this from its `except` handlers, so an AttributeError here would replace the
        real exception with a confusing one and cost the run its honest experiments row.
        """
        if not hasattr(self, "_best_path"):
            return {}
        self._write_best()
        last = self._last_record or {}
        best = self._best_record or {}
        return {
            "best_iteration": best.get("best_iteration"),
            "best_walk_score": best.get("best_walk_score"),
            "best_gated": best.get("gated", False),
            "best_checkpoint": best.get("checkpoint") if best else None,
            "final_iteration": last.get("iteration"),
            "final_feet_air_time": last.get("feet_air_time"),
            "final_track_lin": last.get("track_lin_vel_xy_exp"),
            "final_track_ang": last.get("track_ang_vel_z_exp"),
            "final_mean_reward": last.get("mean_reward"),
            "final_learning_rate": last.get("learning_rate"),
            "walked": bool(best.get("gated", False)),
            "progress": self._progress_path.name,
        }

    def report(self, msg: str) -> None:
        """Kit hijacks python stdout; stderr survives (notes/setup.md)."""
        print(msg, file=sys.stderr, flush=True)
