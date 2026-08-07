"""How far apart are these frames, in the units LingBot-Map's keyframe selector uses?

Phase 3 preflight. This is the diagnostic that explained `example/courthouse` on
Aug 6, promoted out of an ad-hoc measurement into a script, because it turned out to
be the single number that decides whether a reconstruction can work at all -- and we
got it wrong twice by reasoning instead of measuring.

Upstream's adaptive keyframe selector (paper sec 4.4, `_compute_flow_magnitude` in
`gct_stream_window_v2.py`) reprojects the current frame's pixels into the last
keyframe's camera using predicted depth and poses, and promotes the frame once the
**mean L2 pixel displacement** clears a threshold. `process_videos.sh` sets that
threshold to **25.0 px** at 518 px width. So 25 px is not a rule of thumb, it is the
spacing upstream's own published demos were tuned around, with every intermediate
frame densely tracked in between.

We cannot compute their metric without running the model (it needs predicted depth and
pose). Dense Farneback flow on the raw frames measures the same physical quantity --
mean per-pixel displacement, parallax included -- and needs no GPU, so it can gate a
run before any VRAM is spent. Phase correlation is reported alongside it purely to stay
comparable with the numbers already in notes/experiments.md (courthouse ~47 px, loop
~2 px), where it was the method used.

The output that matters is the **interval table**: flow measured between frames i and
i+k, for each candidate k. Read off the k whose median lands near 25 px and that is
`--keyframe_interval`, chosen from the footage instead of swept blind. If even k=1 is
already past 25 px, no keyframe_interval helps -- the frames are the problem, and that
is exactly the state courthouse was in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Upstream's own target, from demo_render/process_videos.sh (FLOW_THRESHOLD=25.0).
UPSTREAM_FLOW_TARGET = 25.0
DEFAULT_INTERVALS = (1, 2, 3, 4, 6, 8, 10)


def _load_gray(path: Path, width: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"could not read {path}")
    if img.shape[1] != width:
        h = int(round(img.shape[0] * width / img.shape[1]))
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    return img


def _dense_flow_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean L2 displacement, the quantity upstream thresholds at 25 px."""
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None,
        pyr_scale=0.5, levels=4, winsize=21, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    return float(np.linalg.norm(flow, axis=2).mean())


def _phase_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Global translational shift -- the Aug 6 courthouse method, kept for continuity."""
    (dx, dy), _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    return float(np.hypot(dx, dy))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure inter-frame flow against upstream's 25 px keyframe target")
    ap.add_argument("--frames", type=Path, required=True,
                    help="folder of JPEG/PNG frames, in filename order")
    ap.add_argument("--width", type=int, default=518,
                    help="measure at the model's input width; flow in pixels is "
                         "meaningless without it")
    ap.add_argument("--intervals", type=int, nargs="+", default=list(DEFAULT_INTERVALS),
                    help="candidate --keyframe_interval values to evaluate")
    ap.add_argument("--sample", type=int, default=300,
                    help="pairs sampled per interval, spread evenly over the clip. "
                         "Dense flow is the cost here; 300 pairs is far past what the "
                         "median needs and keeps this a preflight, not a job")
    ap.add_argument("--fps", type=float, default=None,
                    help="source rate, for reporting seconds-per-keyframe")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the table as JSON (default: <frames>/../flow.json)")
    a = ap.parse_args()

    paths = sorted(
        p for p in a.frames.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if len(paths) < 2:
        raise SystemExit(f"need >=2 frames in {a.frames}, found {len(paths)}")
    print(f"{len(paths)} frames in {a.frames}")
    print(f"measuring at {a.width} px width, "
          f"upstream keyframe target {UPSTREAM_FLOW_TARGET:.0f} px\n")

    cache: dict[int, np.ndarray] = {}

    def gray(i: int) -> np.ndarray:
        if i not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[i] = _load_gray(paths[i], a.width)
        return cache[i]

    h = gray(0).shape[0]
    print(f"working resolution {a.width}x{h}\n")

    rows = []
    print(f"{'kfi':>4} {'pairs':>6} {'flow med':>9} {'p90':>8} "
          f"{'phase med':>10} {'s/kf':>7}  verdict")
    print("-" * 68)
    for k in a.intervals:
        n_pairs = len(paths) - k
        if n_pairs < 1:
            continue
        idx = np.unique(np.linspace(0, n_pairs - 1, min(a.sample, n_pairs)).astype(int))
        flows, phases = [], []
        for i in idx:
            g0, g1 = gray(int(i)), gray(int(i) + k)
            flows.append(_dense_flow_mean(g0, g1))
            phases.append(_phase_shift(g0, g1))
        flows = np.array(flows)
        phases = np.array(phases)
        med = float(np.median(flows))
        p90 = float(np.percentile(flows, 90))
        spk = k / a.fps if a.fps else float("nan")
        # Under target: the selector has denser input than it needs, which is the
        # regime the mechanism was designed for. Over: keyframes are further apart
        # than upstream ever tested, and nothing downstream recovers the gap.
        if med <= UPSTREAM_FLOW_TARGET:
            verdict = "OK"
        elif med <= 1.5 * UPSTREAM_FLOW_TARGET:
            verdict = "marginal"
        else:
            verdict = "TOO SPARSE"
        rows.append({
            "keyframe_interval": k, "pairs": int(idx.size),
            "flow_median_px": round(med, 2),
            "flow_p90_px": round(p90, 2),
            "flow_mean_px": round(float(flows.mean()), 2),
            "phase_shift_median_px": round(float(np.median(phases)), 2),
            "seconds_per_keyframe": (None if a.fps is None else round(spk, 3)),
            "verdict": verdict,
        })
        print(f"{k:>4} {idx.size:>6} {med:>9.2f} {p90:>8.2f} "
              f"{np.median(phases):>10.2f} {spk:>7.3f}  {verdict}")

    print()
    ok = [r for r in rows if r["verdict"] == "OK"]
    if not ok:
        best = min(rows, key=lambda r: r["flow_median_px"])
        print(f"NO interval meets the 25 px target -- even kfi=1 measures "
              f"{best['flow_median_px']:.1f} px.")
        print("These frames are sampled more sparsely than upstream's keyframe")
        print("spacing, so there is nothing densely tracked in between and no")
        print("keyframe_interval can recover it. Source denser footage or accept")
        print("that this is the courthouse regime.")
        recommended = 1
    else:
        recommended = max(r["keyframe_interval"] for r in ok)
        print(f"RECOMMENDED --keyframe_interval {recommended} "
              f"(median flow {next(r['flow_median_px'] for r in ok if r['keyframe_interval']==recommended):.1f} px "
              f"<= {UPSTREAM_FLOW_TARGET:.0f} px)")
        print("Larger intervals cover more frames per window, but every value above")
        print("this one buys window count with keyframe spacing the model was never")
        print("tuned for -- that is the trade the sweep is meant to price, not assume.")

    out = a.out or (a.frames.parent / "flow.json")
    out.write_text(json.dumps({
        "frames": str(a.frames),
        "n_frames": len(paths),
        "width": a.width,
        "height": h,
        "upstream_flow_target_px": UPSTREAM_FLOW_TARGET,
        "fps": a.fps,
        "recommended_keyframe_interval": recommended,
        "intervals": rows,
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
