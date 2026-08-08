"""Fetch the Eiger Trail route polyline + the swissALTI3D 0.5 m DEM tiles that cover it.

The trail: OSM relation 19465820 ("Eiger Trail", SchweizMobil route 353,
Eigergletscher -> Alpiglen, ~6 km of rocky alpine descent under the Eiger north
face). Same terrain family as the GrandTour EIG-1 benchmark footage the recon
toolchain is scored on.

The DEM: swisstopo swissALTI3D, 0.5 m grid, LV95 (EPSG:2056), free anonymous
download via the data.geo.admin.ch STAC API. Tiles are 1 km x 1 km GeoTIFFs.

Outputs (all under data/, gitignored — this script is the provenance record):
    data/eiger_trail/trail_lv95.json    ordered centerline [E, N] + arc length
    data/eiger_trail/relation.json      raw Overpass response (cached)
    data/swissalti3d/*.tif              0.5 m GeoTIFF tiles covering the corridor

Stdlib only; downloads go through curl.exe because Python TLS on this box needs
the Norton CA bundle while curl uses schannel and just works (notes/setup.md).

Usage:
    python sims/isaac/terrain/fetch_eiger_trail.py [--corridor 200]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAIL_DIR = REPO / "data" / "eiger_trail"
DEM_DIR = REPO / "data" / "swissalti3d"

RELATION_ID = 19465820
OVERPASS = "https://overpass-api.de/api/interpreter"
STAC_ITEMS = (
    "https://data.geo.admin.ch/api/stac/v0.9/"
    "collections/ch.swisstopo.swissalti3d/items"
)


def curl(url: str, out: Path | None = None, retries: int = 3) -> bytes:
    """Fetch a URL with curl.exe (schannel TLS — works under Norton MITM)."""
    for attempt in range(1, retries + 1):
        cmd = ["curl.exe", "-sS", "--fail", "--max-time", "300", url]
        if out is not None:
            cmd += ["-o", str(out)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            return b"" if out is not None else proc.stdout
        print(f"  curl attempt {attempt}/{retries} failed: {proc.stderr.decode(errors='replace').strip()}")
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


def wgs84_to_lv95(lat: float, lon: float) -> tuple[float, float]:
    """Swisstopo approximate formulas (accuracy ~1 m — fine for a 24 m strip)."""
    p = (lat * 3600.0 - 169028.66) / 10000.0
    l = (lon * 3600.0 - 26782.5) / 10000.0
    e = (
        2600072.37
        + 211455.93 * l
        - 10938.51 * l * p
        - 0.36 * l * p * p
        - 44.54 * l**3
    )
    n = (
        1200147.07
        + 308807.95 * p
        + 3745.25 * l * l
        + 76.63 * p * p
        - 194.56 * l * l * p
        + 119.79 * p**3
    )
    return e, n


def fetch_relation() -> dict:
    cached = TRAIL_DIR / "relation.json"
    if cached.exists():
        print(f"using cached relation: {cached}")
        return json.loads(cached.read_text())
    print(f"fetching OSM relation {RELATION_ID} from Overpass ...")
    query = f"[out:json][timeout:90];relation({RELATION_ID});out geom;"
    raw = subprocess.run(
        ["curl.exe", "-sS", "--fail", "--max-time", "120", "-G", OVERPASS,
         "--data-urlencode", f"data={query}"],
        capture_output=True, check=True,
    ).stdout
    data = json.loads(raw)
    cached.write_text(json.dumps(data))
    return data


def chain_ways(relation: dict) -> list[tuple[float, float]]:
    """Concatenate the relation's way geometries into one ordered polyline.

    The members come ordered head-to-tail; a way may still be reversed relative
    to the chain, and tiny gaps (missing connector nodes) get bridged by a
    straight segment. Anything larger than 100 m is treated as data corruption.
    """
    ways = [m for m in relation["elements"][0]["members"] if m["type"] == "way"]
    pts: list[tuple[float, float]] = []
    for w in ways:
        geom = [(g["lat"], g["lon"]) for g in w["geometry"]]
        if pts:
            def gap(a, b):
                return math.hypot((a[0] - b[0]) * 111320.0,
                                  (a[1] - b[1]) * 111320.0 * math.cos(math.radians(a[0])))
            if gap(pts[-1], geom[-1]) < gap(pts[-1], geom[0]):
                geom.reverse()
            g = gap(pts[-1], geom[0])
            if g > 100.0:
                raise RuntimeError(f"way {w['ref']} is {g:.0f} m from the chain — relation order broken")
            if g < 0.5:
                geom = geom[1:]  # shared joint node
        pts.extend(geom)
    return pts


def tiles_for_corridor(en: list[tuple[float, float]], margin: float) -> set[tuple[int, int]]:
    """1 km LV95 tile keys within `margin` metres of the centerline."""
    keys: set[tuple[int, int]] = set()
    for e, n in en:
        for de in (-margin, 0.0, margin):
            for dn in (-margin, 0.0, margin):
                keys.add((int((e + de) // 1000), int((n + dn) // 1000)))
    return keys


def stac_tile_assets(en: list[tuple[float, float]]) -> dict[tuple[int, int], str]:
    """Map tile key -> 0.5 m GeoTIFF asset URL, via one paginated bbox query."""
    lats = [g for g, _ in _wgs_cache]
    lons = [g for _, g in _wgs_cache]
    bbox = f"{min(lons)-0.01},{min(lats)-0.01},{max(lons)+0.01},{max(lats)+0.01}"
    url = f"{STAC_ITEMS}?bbox={bbox}&limit=100"
    assets: dict[tuple[int, int], str] = {}
    while url:
        page = json.loads(curl(url))
        for feat in page.get("features", []):
            # id: swissalti3d_<year>_<EEEE>-<NNNN>
            ee, nn = feat["id"].rsplit("_", 1)[-1].split("-")
            key = (int(ee), int(nn))
            for name, asset in feat["assets"].items():
                if asset.get("eo:gsd") == 0.5 and name.endswith(".tif"):
                    assets[key] = asset["href"]
        url = next((lk["href"] for lk in page.get("links", []) if lk["rel"] == "next"), None)
    return assets


_wgs_cache: list[tuple[float, float]] = []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corridor", type=float, default=200.0,
                    help="half-width (m) of the tile-coverage corridor around the trail")
    args = ap.parse_args()

    TRAIL_DIR.mkdir(parents=True, exist_ok=True)
    DEM_DIR.mkdir(parents=True, exist_ok=True)

    relation = fetch_relation()
    latlon = chain_ways(relation)
    _wgs_cache.extend(latlon)
    en = [wgs84_to_lv95(lat, lon) for lat, lon in latlon]

    # cumulative arc length in metres
    arc = [0.0]
    for (e0, n0), (e1, n1) in zip(en, en[1:]):
        arc.append(arc[-1] + math.hypot(e1 - e0, n1 - n0))
    print(f"trail: {len(en)} vertices, {arc[-1]:.0f} m, "
          f"E {min(e for e,_ in en):.0f}-{max(e for e,_ in en):.0f}, "
          f"N {min(n for _,n in en):.0f}-{max(n for _,n in en):.0f}")

    trail = {
        "source": f"OSM relation {RELATION_ID} (Eiger Trail, route 353, Eigergletscher->Alpiglen)",
        "crs": "EPSG:2056 (LV95), metres",
        "length_m": arc[-1],
        "e": [e for e, _ in en],
        "n": [n for _, n in en],
        "arc_m": arc,
    }
    (TRAIL_DIR / "trail_lv95.json").write_text(json.dumps(trail))
    print(f"wrote {TRAIL_DIR / 'trail_lv95.json'}")

    needed = tiles_for_corridor(en, args.corridor)
    print(f"corridor needs {len(needed)} tiles; querying STAC ...")
    assets = stac_tile_assets(en)
    missing = needed - set(assets)
    if missing:
        raise RuntimeError(f"STAC has no 0.5 m asset for tiles: {sorted(missing)}")

    for i, key in enumerate(sorted(needed), 1):
        href = assets[key]
        out = DEM_DIR / href.rsplit("/", 1)[-1]
        if out.exists() and out.stat().st_size > 0:
            print(f"[{i}/{len(needed)}] cached  {out.name}")
            continue
        print(f"[{i}/{len(needed)}] fetch   {out.name}")
        curl(href, out)
    total_mb = sum(f.stat().st_size for f in DEM_DIR.glob("*.tif")) / 1e6
    print(f"DEM tiles ready: {len(list(DEM_DIR.glob('*.tif')))} files, {total_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
