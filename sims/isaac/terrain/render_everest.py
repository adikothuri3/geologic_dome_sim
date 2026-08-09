"""Render the Everest summit terrain assets into reports/ evidence images.

Four renders, regenerable any time from data/everest/ + data/hma_dem/:

    everest_map.png       ~12 km context hillshade around the massif with the
                          training-patch outline (HMA Albers grid)
    everest_patch.png     the training patch, hillshaded, spawn origins overlaid
    everest_oblique.png   3D oblique of the summit pyramid, true vertical scale
    everest_profile.png   E-W elevation profile through the summit, absolute metres

Run in the isaac venv:
    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims/isaac/terrain/render_everest.py
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
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[3]
TERRAIN = REPO / "data" / "everest" / "terrain"
OUT = REPO / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_everest_terrain import read_geotiff, wgs84_to_hma_albers  # noqa: E402

ACCENT = "#C8442C"   # the one identity color: patch outline / spawn marks
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


def render_map(meta: dict, dem_path: Path) -> None:
    grid, px, e0, n0, _ = read_geotiff(dem_path)
    e_c, n_c = meta["center_albers"]
    ctx = 6000.0  # half-width of the context window (m)
    r0 = max(0, int((n0 - (n_c + ctx)) / px))
    r1 = int((n0 - (n_c - ctx)) / px)
    c0 = max(0, int((e_c - ctx - e0) / px))
    c1 = int((e_c + ctx - e0) / px)
    g = grid[r0:r1, c0:c1][::2, ::2]  # 16 m for the figure
    del grid
    shade = LightSource(azdeg=315, altdeg=45).hillshade(
        np.nan_to_num(g, nan=np.nanmean(g)), dx=px * 2, dy=px * 2, vert_exag=1.0
    )
    shade = np.ma.masked_where(np.isnan(g), shade)

    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = plt.get_cmap(HILL_CMAP).copy()
    cmap.set_bad("#ECECEC")
    ext = ((e_c - ctx) / 1000, (e_c + ctx) / 1000,
           (n_c - ctx) / 1000, (n_c + ctx) / 1000)
    ax.imshow(shade, cmap=cmap, extent=ext)

    half = meta["size_m"] / 2 / 1000
    ax.add_patch(Rectangle((e_c / 1000 - half, n_c / 1000 - half), 2 * half, 2 * half,
                           fill=False, edgecolor=ACCENT, lw=2.2))
    s_e, s_n = wgs84_to_hma_albers(27.9881, 86.9250)
    ax.plot(s_e / 1000, s_n / 1000, "^", color=ACCENT, ms=9, mec="white", mew=1.2)
    ax.annotate("Everest  8849 m", (s_e / 1000, s_n / 1000), textcoords="offset points",
                xytext=(10, 8), fontsize=9, color=INK)
    ax.set_xlabel("HMA Albers easting (km)")
    ax.set_ylabel("HMA Albers northing (km)")
    ax.set_title("Everest massif, HMA 8 m DEM (tile-677) — the training patch outlined",
                 fontsize=12, color=INK, loc="left")
    ax.set_aspect("equal")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "everest_map.png", dpi=160)
    plt.close(fig)


def render_patch(H: np.ndarray, x: np.ndarray, y: np.ndarray,
                 origins: np.ndarray, meta: dict) -> None:
    shade = LightSource(azdeg=315, altdeg=45).hillshade(
        H, dx=meta["resolution_m"], dy=meta["resolution_m"], vert_exag=1.0
    )
    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.imshow(shade, cmap=HILL_CMAP, origin="lower",
              extent=(x[0], x[-1], y[0], y[-1]))
    ax.plot(origins[:, 0], origins[:, 1], ".", color=ACCENT, ms=4)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    ax.plot(x[j], y[i], "^", color=ACCENT, ms=10, mec="white", mew=1.2)
    ax.annotate("summit", (x[j], y[i]), textcoords="offset points",
                xytext=(10, 6), fontsize=9, color=INK)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(
        f"The training patch ({meta['size_m']:.0f} m sq, 8 m cells) — dots are the "
        f"{meta['num_origins']} spawn origins (slope <= {meta['origin_max_slope_deg']:.0f} deg)",
        fontsize=11, color=INK, loc="left")
    ax.set_aspect("equal")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "everest_patch.png", dpi=160)
    plt.close(fig)


def render_oblique(H: np.ndarray, x: np.ndarray, y: np.ndarray, meta: dict) -> None:
    X, Y = np.meshgrid(x, y, indexing="xy")
    ls = LightSource(azdeg=315, altdeg=50)
    rgb = ls.shade(H, cmap=plt.get_cmap("gray"), vert_exag=1.0, blend_mode="soft")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, H, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    zr = float(H.max() - H.min())
    ax.set_box_aspect((x[-1] - x[0], y[-1] - y[0], zr))  # true scale
    ax.view_init(elev=32, azim=-120)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
    ax.grid(False)
    ax.set_xlabel("east (m)", fontsize=8, color=INK)
    ax.set_ylabel("north (m)", fontsize=8, color=INK)
    ax.set_zlabel("z (m)", fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title(f"Oblique from the southwest — {zr:.0f} m of relief, true vertical scale",
                 fontsize=11, color=INK, loc="left")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
    fig.savefig(OUT / "everest_oblique.png", dpi=160)
    plt.close(fig)


def render_profile(H: np.ndarray, x: np.ndarray, z_offset: float) -> None:
    i, _ = np.unravel_index(np.argmax(H), H.shape)
    line = H[i] + z_offset
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(x, line, line.min() - 40, color="#B9C3CC", alpha=0.45, lw=0)
    ax.plot(x, line, color="#2C5FA8", lw=1.8)
    j = int(np.argmax(line))
    ax.annotate(f"{line[j]:.0f} m", (x[j], line[j]), textcoords="offset points",
                xytext=(6, 6), fontsize=9, color=INK)
    ax.set_xlabel("east (m, through the summit)")
    ax.set_ylabel("elevation (m a.s.l.)")
    ax.set_title("E-W elevation profile through the summit", fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", color="#E4E6E8", lw=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.01)
    ax.set_ylim(line.min() - 40, line.max() + 60)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "everest_profile.png", dpi=160)
    plt.close(fig)


def main() -> int:
    meta = json.loads((TERRAIN / "meta.json").read_text())
    d = np.load(TERRAIN / "heightfield.npz")
    H, x, y = d["height"], d["x"], d["y"]
    origins = np.load(TERRAIN / "origins.npz")["origins"]
    OUT.mkdir(exist_ok=True)

    dem = REPO / "data" / "hma_dem" / "HMA_DEM8m_MOS_20170716_tile-677.tif"
    render_map(meta, dem)
    render_patch(H, x, y, origins, meta)
    render_oblique(H, x, y, meta)
    render_profile(H, x, float(d["z_offset"]))
    for name in ("map", "patch", "oblique", "profile"):
        p = OUT / f"everest_{name}.png"
        print(f"{p}  ({p.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
