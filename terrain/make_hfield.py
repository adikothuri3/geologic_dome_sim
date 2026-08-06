"""Step B2: generate a numpy heightfield and write it into a MuJoCo hfield asset.

The four mechanics that cause most hfield bugs, all handled here:

  * ``<hfield size="radius_x radius_y elevation_z base_z"/>`` -- ``elevation_z`` is a
    SCALE FACTOR, not a maximum. The data must be normalized to [0, 1].
  * ``base_z`` must be non-zero or the terrain has no thickness where elevation is 0.
  * ``model.hfield_data`` is 1-D, row-major, length ``nrow * ncol``, stored at offset
    ``model.hfield_adr[hid]``.
  * Row order in ``mjModel`` is flipped relative to data given in XML. We bypass XML and
    write ``mjModel`` directly, so our row 0 is the -y edge. ``drop_test.py`` validates
    this empirically rather than taking it on faith -- Phase 4 orientation depends on it.

Fill ``hfield_data`` BEFORE constructing a Renderer, otherwise ``mujoco.mjr_uploadHField``
must be called or the visuals will not match the physics.
"""

import pathlib

import mujoco
import numpy as np

# Radius (in normalized [-1, 1] grid coords) of a level pad at the origin, blended out
# with a smoothstep. The G1's `stand` keyframe is straight-legged and there is no policy
# to catch it in Phase 1, so the spawn point is deliberately flat. Set to 0.0 to spawn on
# raw terrain. Disclosed in the lab notebook.
FLAT_PAD_FRAC = 0.06


def elevation_grid(nrow: int, ncol: int, flat_pad_frac: float = FLAT_PAD_FRAC) -> np.ndarray:
    """Smooth, low-frequency terrain, normalized to [0, 1]. Indexed [row, col] = [y, x]."""
    xs = np.linspace(-1.0, 1.0, ncol)
    ys = np.linspace(-1.0, 1.0, nrow)
    x, y = np.meshgrid(xs, ys)

    # Wavelengths of roughly 2-3 m over the 6 m field: visible undulation on camera while
    # keeping slopes well under what a straight-legged stand can tolerate.
    def surface(u, v):
        return np.sin(6.3 * u) * np.cos(5.1 * v) + 0.5 * np.sin(9.9 * u + 1.1) * np.cos(8.7 * v - 0.4)

    z = surface(x, y)

    if flat_pad_frac > 0.0:
        r = np.hypot(x, y)
        t = np.clip((r - flat_pad_frac) / flat_pad_frac, 0.0, 1.0)
        weight = t * t * (3.0 - 2.0 * t)  # smoothstep, 0 at the pad -> 1 outside the blend
        z = z * weight + float(surface(0.0, 0.0)) * (1.0 - weight)

    z = z - z.min()
    span = z.max()
    return z / span if span > 0 else z


def fill_hfield(model: mujoco.MjModel, name: str = "terrain", **kwargs) -> np.ndarray:
    """Generate terrain and write it into ``model``. Returns the normalized [0, 1] grid."""
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
    if hid == -1:
        raise ValueError(f"no hfield named {name!r} in this model")

    nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    grid = elevation_grid(nrow, ncol, **kwargs)

    adr = int(model.hfield_adr[hid])
    model.hfield_data[adr : adr + nrow * ncol] = grid.ravel()
    return grid


def load_asset(name: str):
    """Load a reconstruction-derived heightfield built by recon/cloud_to_hfield.py.

    Returns ``(grid, meta)`` where grid is the normalized [0, 1] array and meta
    carries the real-world ``size`` the <hfield> must declare for metres to survive.
    """
    import json

    d = pathlib.Path(__file__).resolve().parent / "assets"
    grid = np.load(d / f"{name}.npy")
    meta = json.loads((d / f"{name}.json").read_text())
    return grid, meta


def fill_hfield_from_asset(model: mujoco.MjModel, name: str,
                           hfield_name: str = "terrain") -> np.ndarray:
    """Write a reconstruction heightfield into ``model``. Returns the grid.

    Same contract as ``fill_hfield``: writes ``model.hfield_data`` directly, so
    row 0 is the -y edge and ``sample_height`` stays valid.
    """
    grid, meta = load_asset(name)
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
    if hid == -1:
        raise ValueError(f"no hfield named {hfield_name!r} in this model")

    nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    if (nrow, ncol) != grid.shape:
        raise ValueError(
            f"<hfield> declares {nrow}x{ncol} but asset {name!r} is "
            f"{grid.shape[0]}x{grid.shape[1]}; copy meta['hfield_xml'] into the scene")

    # The grid is normalized to [0, 1]; every real metre lives in `size`. A scene
    # still carrying a previous build's size silently rescales the whole terrain,
    # and nothing downstream would notice.
    want, have = np.asarray(meta["size"], float), np.asarray(model.hfield_size[hid], float)
    if not np.allclose(want, have, atol=1e-4):
        raise ValueError(
            f"<hfield> size {have.tolist()} does not match asset {name!r} "
            f"({want.tolist()}); copy meta['hfield_xml'] into the scene")

    adr = int(model.hfield_adr[hid])
    model.hfield_data[adr: adr + nrow * ncol] = grid.ravel()
    return grid


def sample_height(model: mujoco.MjModel, grid: np.ndarray, x: float, y: float,
                  name: str = "terrain") -> float:
    """World-frame terrain height in metres at (x, y), bilinearly interpolated."""
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
    radius_x, radius_y, elevation_z, _base_z = model.hfield_size[hid]
    nrow, ncol = grid.shape

    col = (x + radius_x) / (2.0 * radius_x) * (ncol - 1)
    row = (y + radius_y) / (2.0 * radius_y) * (nrow - 1)
    col = float(np.clip(col, 0, ncol - 1))
    row = float(np.clip(row, 0, nrow - 1))

    c0, r0 = int(np.floor(col)), int(np.floor(row))
    c1, r1 = min(c0 + 1, ncol - 1), min(r0 + 1, nrow - 1)
    fc, fr = col - c0, row - r0

    top = grid[r0, c0] * (1 - fc) + grid[r0, c1] * fc
    bot = grid[r1, c0] * (1 - fc) + grid[r1, c1] * fc
    return float((top * (1 - fr) + bot * fr) * elevation_z)
