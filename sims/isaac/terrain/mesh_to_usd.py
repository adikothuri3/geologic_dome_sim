"""Convert a terrain OBJ into a static-collision USD via Isaac Lab's MeshConverter.

Collision is the exact triangle mesh (TriangleMeshPropertiesCfg, PhysX
"none"-approximation) — valid only for static geometry, which terrain is.
No mass/rigid props: the terrain never moves.

Run in the isaac venv (opens a headless SimulationApp — first run pays the
shader-cache cost; gate verdicts go to stderr because Kit hijacks stdout,
see notes/setup.md):

    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims/isaac/terrain/mesh_to_usd.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--obj", type=Path, default=REPO / "data/eiger_trail/terrain/trail_terrain.obj")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/eiger_trail/terrain/usd")
    ap.add_argument("--name", default="eiger_trail.usd")
    args = ap.parse_args()

    # Kit's CWD is not the repo — relative paths break USD layer re-reads inside
    # MeshConverter ("Accessed invalid null prim"), so resolve before launch.
    args.obj = args.obj.resolve()
    args.out_dir = args.out_dir.resolve()
    if not args.obj.exists():
        log(f"FAIL: OBJ not found: {args.obj} — run build_trail_terrain.py first")
        return 1

    t0 = time.time()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    log(f"SimulationApp up in {time.time() - t0:.0f} s")

    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg

    cfg = MeshConverterCfg(
        asset_path=str(args.obj),
        usd_dir=str(args.out_dir),
        usd_file_name=args.name,
        force_usd_conversion=True,
        make_instanceable=False,
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=schemas_cfg.TriangleMeshPropertiesCfg(),
        mass_props=None,
        rigid_props=None,
    )
    t1 = time.time()
    usd_path = MeshConverter(cfg).usd_path
    size_mb = Path(usd_path).stat().st_size / 1e6
    log(f"PASS: {usd_path} ({size_mb:.0f} MB) in {time.time() - t1:.0f} s")

    # Kit teardown can access-violate on this box — everything is already on disk.
    try:
        app.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
