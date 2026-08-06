"""Video -> JPEG frame folder for LingBot-Map.

Phase 3. LingBot-Map's demos take an --image_folder, so every clip has to be
decoded to stills first. Two things about phone footage make this more than an
`ffmpeg -i x.mov out%06d.jpg`:

1. **HDR.** Recent iPhones record HEVC 10-bit in HLG (arib-std-b67) with a Dolby
   Vision RPU. Decoded naively to 8-bit sRGB the frames come out washed out and
   low-contrast, which is exactly the texture signal a feed-forward
   reconstruction model keys on. We tonemap HLG/PQ -> BT.709 SDR properly.
2. **Rotation.** A portrait recording is stored as a landscape stream plus a
   `displaymatrix` side-data rotation. ffmpeg autorotates by default; we assert
   the output orientation afterwards rather than trusting it silently.

Uses the ffmpeg binary bundled with imageio-ffmpeg -- same reasoning as the
video writer in Phase 1 (notes/setup.md): Windows has no system ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

# Transfer characteristics that mean "this is HDR and needs tonemapping".
HDR_TRC = {"arib-std-b67", "smpte2084"}


def ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(video: Path) -> dict:
    """Pull the handful of stream properties we branch on out of ffmpeg -i."""
    out = subprocess.run(
        [ffmpeg(), "-hide_banner", "-i", str(video)],
        capture_output=True,
        text=True,
        errors="replace",
    ).stderr

    info: dict = {"raw": out}

    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    info["duration_s"] = (
        int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    )

    vline = next((l for l in out.splitlines() if "Video:" in l), "")
    m = re.search(r"(\d{2,5})x(\d{2,5})", vline)
    info["width"], info["height"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)

    m = re.search(r"([\d.]+) fps", vline)
    info["fps"] = float(m.group(1)) if m else None

    # Colour tags live in the parenthesised pix_fmt group: yuv420p10le(tv, bt2020nc/bt2020/arib-std-b67)
    m = re.search(r"\(tv,\s*([\w-]+)/([\w-]+)/([\w-]+)\)", vline)
    if m:
        info["matrix"], info["primaries"], info["transfer"] = m.groups()
    else:
        info["matrix"] = info["primaries"] = info["transfer"] = None

    m = re.search(r"displaymatrix: rotation of (-?[\d.]+) degrees", out)
    info["rotation"] = float(m.group(1)) if m else 0.0

    info["is_hdr"] = info["transfer"] in HDR_TRC
    info["is_dovi"] = "DOVI configuration record" in out
    return info


def build_filter(info: dict, fps: float, long_side: int | None, rotate_cw: bool = False) -> str:
    """Frame-rate, tonemap and resize chain, in that order."""
    chain = [f"fps={fps}"]

    if info["is_hdr"]:
        # Linearise at 100 nits reference, tonemap in float, land back in BT.709.
        # `hable` preserves shadow detail better than `reinhard` here, and
        # desat=0 keeps colour saturation the model can use as texture.
        chain += [
            "zscale=t=linear:npl=100",
            "format=gbrpf32le",
            "zscale=p=bt709",
            "tonemap=hable:desat=0",
            "zscale=t=bt709:m=bt709:r=tv",
        ]
    elif info["matrix"] and info["matrix"] != "bt709":
        chain.append("zscale=t=bt709:m=bt709:p=bt709:r=tv")

    if long_side:
        # Scale the long edge, preserving aspect; -2 keeps the other edge even.
        chain.append(
            f"scale=w=-2:h={long_side}:force_original_aspect_ratio=decrease:flags=lanczos"
        )

    if rotate_cw:
        # transpose=1 is a lossless 90 deg clockwise rotation, applied *after*
        # ffmpeg's autorotation has already put the frame upright.
        chain.append("transpose=1")

    chain.append("format=yuv420p")
    return ",".join(chain)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--fps", type=float, default=10.0, help="sampling rate (default 10)")
    ap.add_argument(
        "--long-side",
        type=int,
        default=960,
        help="resize long edge to this many px (0 = native)",
    )
    ap.add_argument("--start", type=float, default=None, help="seek to t seconds")
    ap.add_argument("--duration", type=float, default=None, help="only take t seconds")
    ap.add_argument("--quality", type=int, default=2, help="mjpeg -q:v (2 = near-lossless)")
    ap.add_argument("--force", action="store_true", help="wipe out_dir if it exists")
    ap.add_argument(
        "--rotate-cw",
        action="store_true",
        help="rotate 90 deg clockwise after autorotation, turning portrait footage "
             "landscape. LingBot-Map fits width to 518 and only crops height, so a "
             "landscape frame costs ~43%% fewer tokens than the same portrait frame "
             "padded to a square -- with no loss of field of view.",
    )
    args = ap.parse_args()

    if not args.video.exists():
        print(f"no such video: {args.video}", file=sys.stderr)
        return 1

    info = probe(args.video)
    print(
        f"input : {args.video.name}\n"
        f"        {info['width']}x{info['height']} stored, rotation {info['rotation']:+.0f} deg, "
        f"{info['fps']} fps, {info['duration_s']:.2f} s\n"
        f"        transfer={info['transfer']} matrix={info['matrix']} "
        f"hdr={info['is_hdr']} dovi={info['is_dovi']}"
    )

    if args.out_dir.exists():
        if not args.force and any(args.out_dir.iterdir()):
            print(f"{args.out_dir} is not empty; pass --force", file=sys.stderr)
            return 1
        if args.force:
            shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vf = build_filter(info, args.fps, args.long_side or None, args.rotate_cw)
    print(f"filter: {vf}")

    cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-stats"]
    if args.start is not None:
        cmd += ["-ss", str(args.start)]
    cmd += ["-i", str(args.video)]
    if args.duration is not None:
        cmd += ["-t", str(args.duration)]
    cmd += [
        "-vf", vf,
        "-q:v", str(args.quality),
        "-an", "-sn", "-dn",           # drop audio / subs / the iPhone mebx data streams
        "-map", "0:v:0",
        str(args.out_dir / "%06d.jpg"),
    ]

    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("ffmpeg failed", file=sys.stderr)
        return r.returncode

    frames = sorted(args.out_dir.glob("*.jpg"))
    if not frames:
        print("ffmpeg wrote no frames", file=sys.stderr)
        return 1

    # Verify orientation actually got applied rather than assuming it.
    from PIL import Image  # noqa: PLC0415 -- optional, only needed for the check

    with Image.open(frames[0]) as im:
        w, h = im.size
    portrait_expected = (abs(info["rotation"]) == 90) != bool(args.rotate_cw)
    got_portrait = h > w
    ok = "OK" if portrait_expected == got_portrait else "MISMATCH"
    total_mb = sum(f.stat().st_size for f in frames) / 2**20

    print(
        f"\nwrote : {len(frames)} frames  {w}x{h}  ({total_mb:.0f} MB)\n"
        f"        orientation {ok} (expected {'portrait' if portrait_expected else 'landscape'})\n"
        f"        -> {args.out_dir}"
    )

    meta = {
        "source": str(args.video),
        "source_size": [info["width"], info["height"]],
        "source_fps": info["fps"],
        "duration_s": info["duration_s"],
        "rotation_deg": info["rotation"],
        "transfer": info["transfer"],
        "matrix": info["matrix"],
        "is_hdr": info["is_hdr"],
        "is_dovi": info["is_dovi"],
        "sample_fps": args.fps,
        "frame_size": [w, h],
        "n_frames": len(frames),
        "filter": vf,
        "start_s": args.start,
        "clip_duration_s": args.duration,
    }
    (args.out_dir.parent / "frames_meta.json").write_text(json.dumps(meta, indent=2))
    return 0 if ok == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
