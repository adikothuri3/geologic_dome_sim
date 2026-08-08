"""Plot commanded vs achieved velocity from a play_g1_flat.py rollout.

    python sims/isaac/scripts/plot_play.py runs/isaac/<run_id>

Reads `play_timeseries.json` and writes `reports/<run_id>-tracking.png`: one panel per
command in the sweep, each showing the three body-frame velocity channels against the
dashed line the policy was asked to hold.

Why this and not the summary table: a MAE is a single number and cannot distinguish a
policy that locks onto the command and holds it from one that oscillates either side of
it with the same mean error. That difference is the whole question when the complaint
about the MuJoCo policy was jitter, and it is obvious on a trace and invisible in a mean.

No Isaac import — this is plain matplotlib over a JSON file, so it runs after the fact
on any checkout, including one with no GPU.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # no display on this box; write files only
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]

# Validated categorical slots 1–3 (dataviz skill, references/palette.md). These three
# clear the all-pairs CVD and normal-vision floors in both modes, which the full eight
# do not; a fourth channel would have to fold into "Other" or a facet rather than take
# slot 4. Aqua sits at 2.74:1 on the light surface — below the 3:1 contrast gate — so
# the relief rule applies and every series carries a visible direct label.
SERIES = [
    ("vx", "#2a78d6", "forward  vx"),
    ("vy", "#eb6834", "lateral  vy"),
    ("wz", "#1baf7a", "yaw  ωz"),
]
INK, MUTED, GRID, BASELINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="runs/isaac/<run_id> holding play_timeseries.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    src = run_dir / "play_timeseries.json"
    if not src.exists():
        sys.exit(f"no play_timeseries.json in {run_dir} — run play_g1_flat.py first")

    blob = json.loads(src.read_text(encoding="utf-8"))
    results = blob["results"]

    fig, axes = plt.subplots(1, len(results), figsize=(2.7 * len(results), 4.3),
                             sharey=True, facecolor=SURFACE)
    if len(results) == 1:
        axes = [axes]

    # A shared y-range across panels, so a flat line in one panel and a tracked command
    # in another are read at the same scale. Per-panel autoscaling would make a policy
    # that does nothing look identical to one that works.
    lo, hi = -1.25, 1.25

    for ax, r in zip(axes, results):
        ax.set_facecolor(SURFACE)
        t = [p["t"] for p in r["trace"]]
        commanded = max(abs(v) for v in r["target"]) > 1e-9
        # The channel this panel is actually about — the one with a nonzero command.
        active = max(range(3), key=lambda i: abs(r["target"][i]))

        ax.axhline(0.0, color=GRID, lw=1.0, zorder=1)
        # The commanded value, as a reference rather than a series: it is not measured
        # data and must not compete with the three channels for attention.
        for value in {v for v in r["target"] if abs(v) > 1e-9}:
            ax.axhline(value, color=MUTED, lw=1.5, ls=(0, (5, 4)), zorder=2)
            # Sit the label on the side of the reference line the trace approaches from,
            # so it lands in empty space rather than on top of the series it annotates.
            ax.annotate(f"commanded {value:+g}", xy=(0.03, value),
                        xycoords=("axes fraction", "data"),
                        xytext=(0, 5 if value > 0 else -13), textcoords="offset points",
                        color=MUTED, fontsize=7.5, zorder=6)

        for i, (key, color, label) in enumerate(SERIES):
            y = [p[key] for p in r["trace"]]
            ax.plot(t, y, color=color, lw=2.0, zorder=4, solid_capstyle="round")
            # Direct-label each series in the panel where it is the *commanded* channel.
            # Labelling all three in one panel collides by construction: the two idle
            # channels both sit on zero, so their labels land on the same pixels. Here
            # every label lands on a line held away from the others.
            if commanded and i == active:
                ax.annotate(label, xy=(t[-1], y[-1]), xytext=(-4, 10),
                            textcoords="offset points", color=color, fontsize=8.5,
                            ha="right", va="center", fontweight="bold", zorder=6)

        ax.set_title(r["command"], color=INK, fontsize=9.5, pad=20)
        # The number the panel is evidence for, directly under its title — below the
        # axes it overprints the x-axis label.
        mae = [r["vx_mae"], r["vy_mae"], r["wz_mae"]][active]
        stat = f"MAE {mae:.2f}" if commanded else "commanded zero"
        ax.annotate(f"{stat} · {r['survival']:.0%} upright", xy=(0.5, 1.015),
                    xycoords="axes fraction", ha="center", color=MUTED, fontsize=8)

        ax.set_xlabel("seconds", color=MUTED, fontsize=8)
        ax.set_ylim(lo, hi)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)

    axes[0].set_ylabel("body-frame velocity   m/s  ·  rad/s", color=MUTED, fontsize=8)

    # A legend as well as the direct labels: the two idle channels in each panel are
    # never direct-labelled anywhere, so without this their identity would rest on
    # colour alone.
    handles = [plt.Line2D([], [], color=c, lw=2.5, label=lbl) for _, c, lbl in SERIES]
    leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.005),
                     ncol=3, frameon=False, fontsize=8.5, handlelength=1.4,
                     columnspacing=1.6, handletextpad=0.5)
    for text, (_, color, _) in zip(leg.get_texts(), SERIES):
        text.set_color(color)

    meta = (f"{blob.get('run_id', run_dir.name)} · {blob['checkpoint']} · "
            f"trained {blob.get('trained_iterations', '?')} iterations at "
            f"{blob.get('trained_num_envs', '?')} envs · "
            f"{blob['num_envs']} eval envs, mean across those still upright")
    fig.suptitle("Does the policy realise the velocity it was commanded?",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.011, ha="left", y=0.985)
    fig.text(0.011, 0.928, meta, color=MUTED, fontsize=8, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))

    out = pathlib.Path(args.out) if args.out else (
        REPO / "reports" / f"{run_dir.name}-tracking.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    try:
        shown = out.relative_to(REPO)
    except ValueError:      # --out may point anywhere, including outside the repo
        shown = out
    print(f"wrote {shown}")


if __name__ == "__main__":
    main()
