"""Render the Eiger Trail terrain assets into reports/ evidence images.

Four renders, regenerable any time from data/eiger_trail/:

    eiger_trail_map.png        regional DEM hillshade + the trail route (LV95)
    eiger_trail_strip.png      the straightened training strip, hillshaded, with
                               the walkable spawn origins overlaid
    eiger_trail_oblique.png    3D oblique of a rock-step section of the strip
    eiger_trail_profile.png    centerline elevation profile, absolute metres

Run in the isaac venv:
    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims/isaac/terrain/render_trail.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

REPO = Path(__file__).resolve().parents[3]
TERRAIN = REPO / "data" / "eiger_trail" / "terrain"
OUT = REPO / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_trail_terrain import load_mosaic  # noqa: E402

ACCENT = "#C8442C"   # the one identity color: the trail / spawn marks
HILL_CMAP = "gray"
INK = "#333639"
MUTED = "#73777B"


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def render_map(trail: dict) -> None:
    grid, e_min, n_max = load_mosaic(REPO / "data" / "swissalti3d")
    ds = 4  # 0.5 m -> 2 m for the figure
    g = grid[::ds, ::ds]
    shade = LightSource(azdeg=315, altdeg=45).hillshade(
        np.nan_to_num(g, nan=np.nanmean(g)), dx=0.5 * ds, dy=0.5 * ds, vert_exag=1.0
    )
    shade = np.ma.masked_where(np.isnan(g), shade)

    fig, ax = plt.subplots(figsize=(11, 9))
    e1 = e_min + grid.shape[1] * 0.5
    n0 = n_max - grid.shape[0] * 0.5
    cmap = plt.get_cmap(HILL_CMAP).copy()
    cmap.set_bad("#ECECEC")
    ax.imshow(shade, cmap=cmap, extent=(e_min / 1000, e1 / 1000, n0 / 1000, n_max / 1000))

    e = np.asarray(trail["e"]) / 1000
    n = np.asarray(trail["n"]) / 1000
    ax.plot(e, n, color=ACCENT, lw=2.2, solid_capstyle="round", label="Eiger Trail (route 353)")
    ax.plot(e[0], n[0], "o", color=ACCENT, ms=7, mec="white", mew=1.2)
    ax.plot(e[-1], n[-1], "s", color=ACCENT, ms=7, mec="white", mew=1.2)
    ax.annotate("Eigergletscher  2320 m", (e[0], n[0]), textcoords="offset points",
                xytext=(10, -12), fontsize=9, color=INK)
    ax.annotate("Alpiglen  1615 m", (e[-1], n[-1]), textcoords="offset points",
                xytext=(10, 8), fontsize=9, color=INK)
    ax.set_xlabel("LV95 easting (km)")
    ax.set_ylabel("LV95 northing (km)")
    ax.set_title("Eiger Trail over swissALTI3D 0.5 m — the terrain now in Isaac Sim",
                 fontsize=12, color=INK, loc="left")
    ax.set_aspect("equal")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "eiger_trail_map.png", dpi=160)
    plt.close(fig)


def render_strip(H: np.ndarray, s: np.ndarray, y: np.ndarray, origins: np.ndarray) -> None:
    rows = 6
    n = H.shape[0]
    chunk = int(np.ceil(n / rows))
    ls = LightSource(azdeg=315, altdeg=45)
    fig, axes = plt.subplots(rows, 1, figsize=(13, 6.5))
    for r, ax in enumerate(axes):
        i0, i1 = r * chunk, min((r + 1) * chunk, n)
        shade = ls.hillshade(H[i0:i1].T, dx=0.5, dy=0.5, vert_exag=1.0)
        ax.imshow(shade, cmap=HILL_CMAP, origin="lower",
                  extent=(s[i0], s[i1 - 1], y[0], y[-1]), aspect="auto")
        m = (origins[:, 0] >= s[i0]) & (origins[:, 0] <= s[i1 - 1])
        ax.plot(origins[m, 0], origins[m, 1], ".", color=ACCENT, ms=2.2)
        ax.set_ylabel(f"{s[i0]/1000:.1f}–{s[min(i1, n-1) - 1]/1000:.1f} km",
                      fontsize=8, color=MUTED, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
    axes[0].set_title(
        "The straightened training strip (5.36 km × 24 m, 0.5 m cells) — "
        "dots are the 1,089 walkable spawn origins", fontsize=11, color=INK, loc="left")
    axes[-1].set_xlabel("distance along trail (m)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "eiger_trail_strip.png", dpi=160)
    plt.close(fig)


def render_oblique(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> None:
    i0, i1 = int(2395 / 0.5), int(2455 / 0.5)  # the rock-step cluster (arc 2415–2420 m)
    Hs = H[i0:i1]
    S, Y = np.meshgrid(s[i0:i1], y, indexing="ij")
    ls = LightSource(azdeg=315, altdeg=50)
    rgb = ls.shade(Hs, cmap=plt.get_cmap("gray"), vert_exag=1.2, blend_mode="soft")

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(S, Y, Hs, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    zr = float(Hs.max() - Hs.min())
    ax.set_box_aspect((60, 24, zr * 1.5))  # 1.5x vertical exaggeration, stated in title
    ax.view_init(elev=38, azim=-65)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
    ax.grid(False)
    ax.set_xlabel("along trail (m)", fontsize=8, color=INK)
    ax.set_ylabel("cross (m)", fontsize=8, color=INK)
    ax.set_zlabel("z (m)", fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title("Oblique: the rock-step section at 2.40–2.46 km (vertical ×1.5)",
                 fontsize=11, color=INK, loc="left")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
    fig.savefig(OUT / "eiger_trail_oblique.png", dpi=160)
    plt.close(fig)


def render_profile(H: np.ndarray, s: np.ndarray, z_offset: float) -> None:
    center = H[:, H.shape[1] // 2] + z_offset
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(s / 1000, center, center.min() - 20, color="#B9C3CC", alpha=0.45, lw=0)
    ax.plot(s / 1000, center, color="#2C5FA8", lw=1.8)
    ax.set_xlabel("distance along trail (km)")
    ax.set_ylabel("elevation (m a.s.l.)")
    ax.set_title("Centerline elevation profile — 713 m descent, mean grade 13.3%",
                 fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", color="#E4E6E8", lw=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.01)
    ax.set_ylim(center.min() - 20, center.max() + 30)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "eiger_trail_profile.png", dpi=160)
    plt.close(fig)


def main() -> int:
    trail = json.loads((REPO / "data" / "eiger_trail" / "trail_lv95.json").read_text())
    d = np.load(TERRAIN / "heightfield.npz")
    H, s, y = d["height"], d["arc"], d["cross"]
    origins = np.load(TERRAIN / "origins.npz")["origins"]
    OUT.mkdir(exist_ok=True)

    render_map(trail)
    render_strip(H, s, y, origins)
    render_oblique(H, s, y)
    render_profile(H, s, float(d["z_offset"]))
    for name in ("map", "strip", "oblique", "profile"):
        p = OUT / f"eiger_trail_{name}.png"
        print(f"{p}  ({p.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
