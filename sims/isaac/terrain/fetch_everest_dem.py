"""Fetch the NASA HMA 8 m DEM mosaic tile covering Mount Everest.

The DEM: NSIDC HMA_DEM8m_MOS v1 (High Mountain Asia 8 m DEM mosaics from
WorldView/GeoEye stereo, ~1.9 m RMSE validated at Everest). One 100 km tile
covers the whole massif:

    HMA_DEM8m_MOS_20170716_tile-677.tif   370 MB, 27.47-28.38 N / 86.50-87.51 E

Granule URL found via CMR search (short_name=HMA_DEM8m_MOS, bbox around the
summit). Downloading needs a free Earthdata login: the TEA endpoint at
data.nsidc.earthdatacloud.nasa.gov 302-redirects through urs.earthdata.nasa.gov
OAuth and back with a session cookie. curl.exe handles the dance natively with
a netrc file + cookie jar; netrc scoping sends the credentials only to the URS
host. Alternative: set EARTHDATA_TOKEN and no netrc is needed.

Credentials file (never in the repo): %USERPROFILE%\\_netrc, one line:

    machine urs.earthdata.nasa.gov login <user> password <pass>

Stdlib + tifffile (validation only); downloads go through curl.exe because
Python TLS on this box needs the Norton CA bundle while curl uses schannel and
just works (notes/setup.md).

Run in the isaac venv:
    %USERPROFILE%\\venvs\\isaac\\Scripts\\python.exe sims/isaac/terrain/fetch_everest_dem.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "data" / "hma_dem"

GRANULE_URL = (
    "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/"
    "HMA/HMA_DEM8m_MOS/1/2002/01/28/HMA_DEM8m_MOS_20170716_tile-677.tif"
)
MIN_SIZE = 300_000_000  # granule is 370 MB; anything smaller is an error page


def curl_download(url: str, out: Path, netrc_file: Path, retries: int = 3) -> None:
    """Authenticated resumable download. curl follows the URS OAuth redirects."""
    jar = out.parent / ".urs_cookies"
    token = os.environ.get("EARTHDATA_TOKEN")
    for attempt in range(1, retries + 1):
        cmd = ["curl.exe", "-sS", "--fail", "-L", "--max-time", "1800",
               "-C", "-", "-c", str(jar), "-b", str(jar), "-o", str(out), url]
        if token:
            cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
        else:
            cmd[1:1] = ["--netrc-file", str(netrc_file)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            return
        # curl exits 33 when the server won't resume a complete file — treat as done
        if proc.returncode == 33 and out.exists() and out.stat().st_size >= MIN_SIZE:
            return
        err = proc.stderr.decode(errors="replace").strip()
        print(f"  curl attempt {attempt}/{retries} failed (exit {proc.returncode}): {err}")
        if "401" in err or "403" in err:
            raise RuntimeError(
                "Earthdata rejected the credentials — check %USERPROFILE%\\_netrc "
                "(machine urs.earthdata.nasa.gov login <user> password <pass>) "
                "or the EARTHDATA_TOKEN env var"
            )
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


def validate(path: Path) -> None:
    size = path.stat().st_size
    if size < MIN_SIZE:
        raise RuntimeError(f"{path.name} is {size/1e6:.0f} MB, expected ~370 MB — "
                           "likely an HTML error page; delete it and check credentials")
    magic = path.open("rb").read(4)
    if magic not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"{path.name} is not a TIFF (magic {magic!r}) — "
                           "delete it and re-run; probably an auth redirect page")
    import tifffile

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        sx, sy, _ = page.tags["ModelPixelScaleTag"].value
        _, _, _, e0, n0, _ = page.tags["ModelTiepointTag"].value
        print(f"  {page.shape[0]}x{page.shape[1]} px, pixel {sx}x{sy} m, "
              f"tiepoint E={e0:.0f} N={n0:.0f}")
        geo = page.tags.get("GeoKeyDirectoryTag")
        if geo is not None:
            keys = geo.value
            epsg = next((keys[i + 3] for i in range(0, len(keys) - 3, 4)
                         if keys[i] == 3072), None)
            print(f"  ProjectedCSTypeGeoKey (EPSG): {epsg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--granule-url", default=GRANULE_URL)
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / GRANULE_URL.rsplit("/", 1)[-1])
    ap.add_argument("--netrc-file", type=Path,
                    default=Path.home() / "_netrc",
                    help="netrc with the urs.earthdata.nasa.gov credentials")
    args = ap.parse_args()

    if args.out.exists() and args.out.stat().st_size >= MIN_SIZE:
        print(f"cached: {args.out} ({args.out.stat().st_size/1e6:.0f} MB)")
        validate(args.out)
        return 0

    if not os.environ.get("EARTHDATA_TOKEN") and not args.netrc_file.exists():
        print(f"no {args.netrc_file} and no EARTHDATA_TOKEN — create a free account "
              "at https://urs.earthdata.nasa.gov, then write that file with one line:\n"
              "  machine urs.earthdata.nasa.gov login <user> password <pass>")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {args.granule_url.rsplit('/', 1)[-1]} (370 MB, resumable) ...")
    curl_download(args.granule_url, args.out, args.netrc_file)
    validate(args.out)
    print(f"DEM ready: {args.out} ({args.out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
