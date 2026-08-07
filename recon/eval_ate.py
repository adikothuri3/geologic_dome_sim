"""Score a reconstruction's camera trajectory against GrandTour's CPT7 ground truth.

Phase 3. Until now every reconstruction in this project has been judged by
`traj_length_over_extent` -- a *self-consistency* proxy that catches a collapsed
trajectory but cannot tell a good reconstruction from a smoothly-wrong one. It also
scored one run 2.87 while the render looked terrible (Aug 5). This script replaces the
proxy with an external reference wherever GrandTour footage is the input.

Three numbers come out, and they answer different questions:

**ATE after Sim(3) alignment** -- the standard monocular metric. Umeyama-fits scale,
rotation and translation, then reports RMSE of the residual. This is the headline
number and the one comparable to the SLAM baselines in the GrandTour paper. Note the
floor: the CPT7 tightly-coupled solution itself carries ~0.132 m mean ATE, so a result
near that is at the reference's own noise level, not necessarily better than it.

**The recovered scale** -- and this is the part that matters most to the pipeline.
Umeyama's scale factor *is* metres-per-reconstruction-unit, measured against a
survey-grade reference. `calibrate_scale.py`'s camera-height anchor was found on Aug 6
to be repeatable only to 14% on identical footage, with no ground truth to say whether
either value was right. This gives one. If a `scale.json` sits beside the run, the two
are printed together, because that comparison is the first real check on every step
height a Phase 5 policy would train on.

**RPE over a fixed distance** -- ATE is dominated by wherever the trajectory diverges
worst, so a single late failure and a uniform wobble can score alike. Relative pose
error over a sliding metric window localises drift, which is what tells a window-count
sweep whether the damage happens at window boundaries or continuously.

The per-segment breakdown exists for the same reason: in windowed (VO) mode the paper
warns that Sim(3) fusion "incurs extra alignment error that compounds with the number
of windows", so error concentrated at boundaries and error spread evenly imply
different fixes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """gt_tum.txt -> (frame_index, xyz). Written by fetch_grandtour.py."""
    idx, xyz = [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        idx.append(int(f[0]))
        xyz.append([float(f[2]), float(f[3]), float(f[4])])
    return np.array(idx), np.array(xyz, dtype=np.float64)


def load_est(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """trajectory.npz -> (frame_index, xyz), indices parsed from the frame filenames.

    fetch_grandtour.py names every frame with its source index precisely so this
    association is exact rather than a timestamp nearest-neighbour search.
    """
    d = np.load(run / "trajectory.npz", allow_pickle=True)
    centers = np.asarray(d["cam_centers"], dtype=np.float64)
    paths = [str(p) for p in d["frame_paths"]]
    idx = []
    for p in paths:
        stem = Path(p).stem
        if not stem.isdigit():
            raise SystemExit(
                f"frame {p!r} is not named by source index -- this run did not come "
                f"from fetch_grandtour.py, so it cannot be associated to GT")
        idx.append(int(stem))
    return np.array(idx), centers


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform mapping src onto dst (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / src.shape[0]
    U, sig, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var_s = (S ** 2).sum() / src.shape[0]
    s = float((sig * np.diag(W)).sum() / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def rpe(est: np.ndarray, gt: np.ndarray, delta_m: float) -> np.ndarray:
    """Translational drift over sliding `delta_m` stretches of the GT path."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    errs = []
    j = 0
    for i in range(len(d)):
        while j < len(d) and d[j] - d[i] < delta_m:
            j += 1
        if j >= len(d):
            break
        errs.append(abs(np.linalg.norm(est[j] - est[i]) - np.linalg.norm(gt[j] - gt[i])))
    return np.array(errs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ATE / scale / RPE of a reconstruction against CPT7 ground truth")
    ap.add_argument("--run", type=Path, required=True,
                    help="reconstruction output folder (holds trajectory.npz)")
    ap.add_argument("--gt", type=Path, required=True, help="gt_tum.txt")
    ap.add_argument("--rpe_delta", type=float, default=10.0,
                    help="RPE window in metres of GT path")
    ap.add_argument("--segments", type=int, default=8,
                    help="equal-length segments for the drift breakdown")
    ap.add_argument("--plot", action="store_true",
                    help="write ate_plot.png -- top-down overlay + error vs distance")
    a = ap.parse_args()

    gt_idx, gt_xyz = load_gt(a.gt)
    est_idx, est_xyz = load_est(a.run)

    common, gi, ei = np.intersect1d(gt_idx, est_idx, return_indices=True)
    if common.size < 10:
        raise SystemExit(
            f"only {common.size} frames common to GT and run -- association failed")
    G, E = gt_xyz[gi], est_xyz[ei]
    print(f"{common.size} associated poses "
          f"(GT {gt_idx.size}, estimate {est_idx.size})")

    gt_len = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
    print(f"GT path length {gt_len:.1f} m\n")

    s, R, t = umeyama(E, G, with_scale=True)
    aligned = (s * (R @ E.T)).T + t
    err = np.linalg.norm(aligned - G, axis=1)
    ate_rmse = float(np.sqrt((err ** 2).mean()))

    print("=" * 62)
    print(f"ATE (Sim(3) aligned)   RMSE {ate_rmse:8.3f} m")
    print(f"                       mean {err.mean():8.3f}   "
          f"median {np.median(err):.3f}   max {err.max():.3f}")
    print(f"                       as % of path length: "
          f"{100*ate_rmse/max(gt_len,1e-9):.2f}%")
    if ate_rmse < 0.132:
        print("  NOTE: below the CPT7 reference's own 0.132 m mean ATE -- at the")
        print("        noise floor of the ground truth, do not read it as better")
    print()
    print(f"recovered scale        {s:.4f} m per reconstruction unit")

    scale_json = a.run / "scale.json"
    if scale_json.exists():
        try:
            anchor = json.loads(scale_json.read_text())
            m_per_unit = anchor.get("m_per_unit") or anchor.get("scale")
            if m_per_unit:
                dev = 100 * (float(m_per_unit) - s) / s
                print(f"  calibrate_scale.py camera anchor: {float(m_per_unit):.4f} "
                      f"({dev:+.1f}% vs ground truth)")
                print("  This is the first GT check on the monocular scale anchor.")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    print()

    r = rpe(aligned, G, a.rpe_delta)
    if r.size:
        print(f"RPE over {a.rpe_delta:.0f} m    RMSE {np.sqrt((r**2).mean()):8.3f} m   "
              f"median {np.median(r):.3f}   n={r.size}")
        print(f"                       drift rate "
              f"{100*np.median(r)/a.rpe_delta:.2f}% of distance travelled")
    print()

    print(f"per-segment ATE ({a.segments} equal slices of the association):")
    bounds = np.linspace(0, common.size, a.segments + 1).astype(int)
    seg_rows = []
    for k in range(a.segments):
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < 2:
            continue
        e = err[lo:hi]
        seg_rows.append({"segment": k, "frames": [int(common[lo]), int(common[hi - 1])],
                         "rmse_m": round(float(np.sqrt((e ** 2).mean())), 3)})
        print(f"  {k}: frames {common[lo]:6d}-{common[hi-1]:6d}  "
              f"RMSE {np.sqrt((e**2).mean()):7.3f} m")
    if seg_rows:
        worst = max(seg_rows, key=lambda x: x["rmse_m"])
        best = min(seg_rows, key=lambda x: x["rmse_m"])
        ratio = worst["rmse_m"] / max(best["rmse_m"], 1e-9)
        print(f"\n  worst/best segment ratio {ratio:.1f}x -- "
              + ("error is concentrated, look for a window boundary there"
                 if ratio > 3 else "error is spread evenly, not a stitching failure"))
    print("=" * 62)

    rec = {
        "run": str(a.run), "gt": str(a.gt),
        "n_associated": int(common.size),
        "gt_path_length_m": round(gt_len, 3),
        "ate_sim3_rmse_m": round(ate_rmse, 4),
        "ate_mean_m": round(float(err.mean()), 4),
        "ate_median_m": round(float(np.median(err)), 4),
        "ate_max_m": round(float(err.max()), 4),
        "ate_pct_of_path": round(100 * ate_rmse / max(gt_len, 1e-9), 3),
        "scale_m_per_unit": round(s, 5),
        "rpe_delta_m": a.rpe_delta,
        "rpe_rmse_m": (round(float(np.sqrt((r ** 2).mean())), 4) if r.size else None),
        "rpe_median_m": (round(float(np.median(r)), 4) if r.size else None),
        "segments": seg_rows,
        "cpt7_reference_ate_m": 0.132,
    }
    out = a.run / "ate.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"-> {out}")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping --plot")
            return
        dist = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))])
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        ax[0].plot(G[:, 0], G[:, 1], lw=2, label="CPT7 ground truth")
        ax[0].plot(aligned[:, 0], aligned[:, 1], lw=1.2, label="LingBot-Map (Sim(3))")
        ax[0].set_aspect("equal")
        ax[0].set_xlabel("east (m)")
        ax[0].set_ylabel("north (m)")
        ax[0].legend()
        ax[0].set_title(f"top-down · ATE {ate_rmse:.2f} m")
        ax[1].plot(dist, err, lw=1)
        ax[1].axhline(ate_rmse, ls="--", c="k", lw=0.8, label=f"RMSE {ate_rmse:.2f} m")
        ax[1].set_xlabel("distance along GT path (m)")
        ax[1].set_ylabel("position error (m)")
        ax[1].legend()
        ax[1].set_title("error vs distance · spikes mark window boundaries")
        fig.tight_layout()
        fig.savefig(a.run / "ate_plot.png", dpi=130)
        print(f"-> {a.run/'ate_plot.png'}")


if __name__ == "__main__":
    main()
