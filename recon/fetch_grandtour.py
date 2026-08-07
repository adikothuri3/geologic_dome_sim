"""GrandTour mission -> LingBot-Map frame folder + CPT7 ground-truth trajectory.

Phase 3. Everything before this in `recon/` starts from a phone video; this starts
from the ETH Zurich RSL **GrandTour** dataset (arXiv 2602.18164), which is the only
footage we have that is both Everest-like *and* carries a metric ground truth. That
ground truth is the point: every reconstruction claim in this project so far has been
scored against `traj_length_over_extent`, a self-consistency proxy, because nothing
external existed to check it. GrandTour gives us ATE against a survey-grade GNSS/INS
and, as a side effect, the project's first ground-truth **scale** check.

Four things about this dataset make it more than "download and untar":

1. **The camera is not the one the paper advertises.** The sensor table lists the
   TierIV HDR cameras at 1920x1280 @ 30 fps. The *released* HDR stream is 10 Hz
   (`data/hdr_front.tar`'s zarr attrs say so outright), and it is an `equidistant`
   fisheye at 120 deg x 80 deg. `zed2i_left_images` is 15 Hz, `radtan`, and 16:9.
   For a feed-forward monocular model those three differences all point the same way
   -- see `--camera`'s help and lab-notebook/2026-W32.md.

2. **Distortion has to go before the model sees a pixel.** LingBot-Map's aggregator
   is trained on perspective imagery. A 120 deg equidistant frame fed in raw is not a
   hard failure, it is a *quiet* one: geometry comes back bent and nothing downstream
   flags it. We rectify to an explicit pinhole model and record the new intrinsics.

3. **One resampling, not two.** `reconstruct.py` resizes to 518 on the long side
   anyway, so rectifying at full resolution and letting it downsample would filter the
   image twice. The undistort map is built straight to the output size instead, so a
   source pixel is touched exactly once.

4. **Ground truth is a pose of the IMU, not of the camera.** `cpt7_ie_tc_odometry`
   gives `cpt7_imu` in `enu_origin`; the camera sits ~0.4 m away on the payload. That
   lever arm swings with every body rotation, which on a trotting quadruped is most of
   the motion, so we compose the full chain rather than comparing to the IMU track.

Ground truth is `cpt7_ie_tc` -- Novatel Inertial Explorer *tightly coupled*, the
highest-precision product in the release (the paper reports mean ATE 0.132 m). The
`ie_rt` variant is the real-time solution and is not the reference.

Nothing here is GrandTour-specific beyond the layout constants: any mission folder in
the HuggingFace release works, which is what makes the SNOW-2 low-texture run a flag
change rather than a second script.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HF_BASE = (
    "https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/"
    "resolve/main"
)

# Short names from the GrandTour paper's mission table -> HuggingFace folder.
# Only the ones this project has a reason to touch; the folder name always works
# directly, so this is a convenience, not a whitelist.
MISSIONS = {
    "eig-1": "2024-11-03-13-51-43",     # 429 s / 219.7 m, alpine, rock+gravel->stairs
    "snow-2": "2024-11-03-07-57-34",    # 261 s / 173.9 m, the low-texture stress case
    "arc-2": "2024-11-18-13-22-14",     # 424 s / 136.2 m
}

# Ground-truth topic. `ie_tc` = Inertial Explorer tightly coupled (highest precision);
# `ie_rt` is the real-time solution and must not be used as a reference.
GT_TOPIC = "cpt7_ie_tc_odometry"


# ── zarr (the release ships raw zarr chunks inside tars, not a zarr store) ────

def _read_zarr_tar(tar_path: Path, want: set[str] | None = None) -> tuple[dict, dict]:
    """Return (attrs, {array_name: ndarray}) from one `data/<topic>.tar`.

    The release ships raw zarr v2 directories inside tars. Two properties of that
    layout bite:

    * **Chunks are zero-padded to the declared chunk shape**, which is enormous
      (`pose_pos` is chunked [8388608, 3] to hold 88801 rows). Decoding is therefore
      not proportional to the data -- `pose_cov` would inflate to 2.4 GB of mostly
      zeros. `want` exists so callers decode only the streams they need.
    * Chunk keys are dot-joined per dimension (`0`, `0.0`, `0.0.0`), so a 2-D array
      does not have a chunk file called `0`.
    """
    import re

    import numcodecs

    attrs: dict = {}
    zarrays: dict[str, dict] = {}
    chunks: dict[str, bytes] = {}
    chunk_re = re.compile(r"^\d+(\.\d+)*$")

    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            parts = Path(m.name).parts
            base = Path(m.name).name
            name = parts[-2] if len(parts) >= 2 else ""
            if base == ".zattrs" and len(parts) == 2:
                attrs = json.loads(tf.extractfile(m).read())
            elif base == ".zarray":
                zarrays[name] = json.loads(tf.extractfile(m).read())
            elif chunk_re.match(base) and name:
                if want is not None and name not in want:
                    continue
                chunks[name] = tf.extractfile(m).read()

    arrays = {}
    for name, za in zarrays.items():
        if name not in chunks:
            continue
        codec = numcodecs.get_codec(za["compressor"])
        flat = np.frombuffer(codec.decode(chunks[name]), dtype=za["dtype"])
        # Reshape to the padded chunk, then trim each axis to the real extent.
        cshape = tuple(za["chunks"])
        shape = tuple(za["shape"])
        if flat.size == int(np.prod(cshape)):
            arr = flat.reshape(cshape)[tuple(slice(0, s) for s in shape)]
        else:
            arr = flat[: int(np.prod(shape))].reshape(shape)
        arrays[name] = np.ascontiguousarray(arr)
    return attrs, arrays


# ── download ─────────────────────────────────────────────────────────────────

def _fetch(mission: str, rel: str, dest: Path) -> Path:
    """Download `<mission>/<rel>` unless it is already on disk and non-empty."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {rel} ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    url = f"{HF_BASE}/{mission}/{rel}"
    print(f"  GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)
    print(f"  -> {dest} ({dest.stat().st_size/1e6:.0f} MB)")
    return dest


# ── geometry ─────────────────────────────────────────────────────────────────

def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """(...,4) xyzw -> (...,3,3). GrandTour stores orientation as xyzw."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,4) xyzw, via the largest-diagonal branch (numerically safe)."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.empty(R.shape[:-2] + (4,), dtype=np.float64)
    for idx in np.ndindex(R.shape[:-2]):
        M, tr = m[idx], t[idx]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (M[2, 1] - M[1, 2]) / s
            y = (M[0, 2] - M[2, 0]) / s
            z = (M[1, 0] - M[0, 1]) / s
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            w = (M[2, 1] - M[1, 2]) / s
            x, y, z = 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            w = (M[0, 2] - M[2, 0]) / s
            x, y, z = (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            w = (M[1, 0] - M[0, 1]) / s
            x, y, z = (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s
        q[idx] = (x, y, z, w)
    return q


def _transform_from_yaml_like(tr: dict) -> np.ndarray:
    """GrandTour metadata transform block -> 4x4 T_base_child."""
    r, t = tr["rotation"], tr["translation"]
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(np.array([r["x"], r["y"], r["z"], r["w"]]))
    T[:3, 3] = (t["x"], t["y"], t["z"])
    return T


def _build_undistort_map(ci: dict, out_w: int, out_h: int, hfov_deg: float | None):
    """Rectification map from the released camera model to a pinhole of our choosing.

    Returns (map1, map2, K_new). `alpha`-style cropping is deliberately not used:
    we set the output focal length from an explicit horizontal FOV so the result is
    reproducible and recorded, rather than depending on OpenCV's valid-pixel search.
    """
    K = np.array(ci["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(ci["D"], dtype=np.float64)
    model = ci["distortion_model"]
    src_w, src_h = ci["width"], ci["height"]

    if hfov_deg is None:
        # Default: keep the source camera's own angular scale, just rescaled to the
        # output raster. For a mild radtan lens this is very nearly identity and
        # throws away nothing; for a fisheye it would over-stretch, so equidistant
        # gets an explicit default instead.
        if model == "equidistant":
            hfov_deg = 90.0
        else:
            f_new_x = K[0, 0] * out_w / src_w
            hfov_deg = 2 * np.degrees(np.arctan(out_w / 2 / f_new_x))

    f_new = (out_w / 2) / np.tan(np.radians(hfov_deg) / 2)
    K_new = np.array(
        [[f_new, 0, out_w / 2 - 0.5], [0, f_new, out_h / 2 - 0.5], [0, 0, 1]],
        dtype=np.float64,
    )

    if model == "equidistant":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D.reshape(4, 1), np.eye(3), K_new, (out_w, out_h), cv2.CV_16SC2
        )
    elif model in ("radtan", "plumb_bob", "rational_polynomial"):
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, np.eye(3), K_new, (out_w, out_h), cv2.CV_16SC2
        )
    else:
        raise SystemExit(f"unsupported distortion_model {model!r}")
    return map1, map2, K_new, hfov_deg


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="GrandTour mission -> LingBot-Map frames + CPT7 ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mission", required=True,
                    help="short name (eig-1, snow-2, arc-2) or HF folder "
                         "(e.g. 2024-11-03-13-51-43)")
    ap.add_argument("--camera", default="zed2i_left_images",
                    help="zed2i_left_images (default: 15 Hz, radtan, 16:9 -- the "
                         "densest and least-distorted stream, and the only one whose "
                         "aspect lands on 518x294); hdr_front (10 Hz, equidistant "
                         "120deg, rolling shutter, 3:2); alphasense_front_center "
                         "(10 Hz, equidistant 126deg, global shutter)")
    ap.add_argument("--cache", type=Path,
                    default=Path("data/grandtour"),
                    help="where the downloaded tars live (re-used across runs)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output folder: frames/, gt_tum.txt, mission.json")
    ap.add_argument("--out_long_side", type=int, default=518,
                    help="long side of the emitted frames. 518 is LingBot-Map's own "
                         "input size, so the default resamples exactly once; raise it "
                         "only if you want frames for something other than the model")
    ap.add_argument("--hfov", type=float, default=None,
                    help="horizontal FOV of the rectified pinhole output, degrees. "
                         "Default: the source camera's own FOV for radtan lenses "
                         "(near-lossless), 90 for equidistant fisheyes")
    ap.add_argument("--start", type=float, default=0.0,
                    help="segment start, seconds from mission start")
    ap.add_argument("--end", type=float, default=None,
                    help="segment end, seconds from mission start")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth frame. Raising this widens inter-frame "
                         "motion, which is the failure mode measure_flow.py exists "
                         "to catch -- check before you use it")
    ap.add_argument("--rot180", action="store_true",
                    help="rotate frames 180 deg. The ZED2i is mounted upside-down on "
                         "the payload; whether the released JPEGs are already "
                         "corrected is checked by the preview this script writes")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    a = ap.parse_args()

    mission = MISSIONS.get(a.mission.lower(), a.mission)
    cache = a.cache / mission
    a.out.mkdir(parents=True, exist_ok=True)
    frames_dir = a.out / "frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"mission {a.mission} -> {mission}")
    print(f"camera  {a.camera}")

    # ── stream metadata (timestamps + intrinsics live in the small data tar) ──
    print("metadata:")
    cam_meta_tar = _fetch(mission, f"data/{a.camera}.tar", cache / "data" / f"{a.camera}.tar")
    cam_attrs, cam_arrays = _read_zarr_tar(cam_meta_tar, want={"timestamp"})
    if "camera_info" not in cam_attrs:
        raise SystemExit(f"{a.camera} has no camera_info -- not an image stream?")
    ci = cam_attrs["camera_info"]
    ts_cam = np.asarray(cam_arrays["timestamp"], dtype=np.float64)
    n_src = ts_cam.size
    dur = float(ts_cam[-1] - ts_cam[0])
    rate = (n_src - 1) / dur if dur > 0 else 0.0
    print(f"  {cam_attrs.get('description','')}")
    print(f"  {n_src} frames, {dur:.1f} s, {rate:.2f} Hz measured")
    print(f"  {ci['width']}x{ci['height']} model={ci['distortion_model']}")

    gaps = np.diff(ts_cam)
    n_gap = int((gaps > 3 * np.median(gaps)).sum())
    if n_gap:
        print(f"  WARNING: {n_gap} timing gaps > 3x median dt "
              f"(max {gaps.max():.3f} s) -- these are motion discontinuities the "
              f"model cannot see coming")

    # ── frame selection ──────────────────────────────────────────────────────
    t0 = ts_cam[0]
    rel = ts_cam - t0
    end = a.end if a.end is not None else rel[-1] + 1.0
    sel = np.where((rel >= a.start) & (rel <= end))[0][:: a.stride]
    if sel.size < 2:
        raise SystemExit("selection is empty -- check --start/--end")
    print(f"selected {sel.size} frames "
          f"({rel[sel[0]]:.1f}-{rel[sel[-1]]:.1f} s, stride {a.stride}, "
          f"{(sel.size-1)/max(rel[sel[-1]]-rel[sel[0]],1e-9):.2f} Hz effective)")

    # ── output geometry ──────────────────────────────────────────────────────
    src_w, src_h = ci["width"], ci["height"]
    out_w = a.out_long_side
    # LingBot-Map patches at 14 px. Landing on a multiple of 14 here means
    # reconstruct.py's own resize is a no-op and the pixels are filtered once.
    out_h = int(round(out_w * src_h / src_w / 14)) * 14
    map1, map2, K_new, hfov = _build_undistort_map(ci, out_w, out_h, a.hfov)
    print(f"rectify -> {out_w}x{out_h} pinhole, hFOV {hfov:.1f} deg, "
          f"f={K_new[0,0]:.1f}")

    # ── images ───────────────────────────────────────────────────────────────
    print("images:")
    img_tar = _fetch(mission, f"images/{a.camera}.tar",
                     cache / "images" / f"{a.camera}.tar")

    want = set(int(i) for i in sel)
    written, kept_idx = 0, []
    print(f"  extracting + rectifying {len(want)} of {n_src} frames...")
    with tarfile.open(img_tar) as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith((".jpeg", ".jpg")):
                continue
            try:
                idx = int(Path(m.name).stem)
            except ValueError:
                continue
            if idx not in want:
                continue
            buf = np.frombuffer(tf.extractfile(m).read(), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  WARNING: frame {idx} failed to decode, skipped")
                continue
            if img.shape[1] != src_w or img.shape[0] != src_h:
                raise SystemExit(
                    f"frame {idx} is {img.shape[1]}x{img.shape[0]} but camera_info "
                    f"says {src_w}x{src_h} -- intrinsics would be wrong")
            if a.rot180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            out = cv2.remap(img, map1, map2, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), out,
                        [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            kept_idx.append(idx)
            written += 1
            if written % 500 == 0:
                print(f"    {written}/{len(want)}")
    kept_idx = np.array(sorted(kept_idx))
    print(f"  wrote {written} frames -> {frames_dir}")
    if written != len(want):
        print(f"  WARNING: {len(want)-written} selected frames were missing "
              f"from the tar")

    # A 3x3 contact sheet so orientation (the ZED2i is mounted upside-down) and
    # rectification are checked by eye once, not assumed forever.
    if written:
        picks = kept_idx[np.linspace(0, len(kept_idx) - 1, 9).astype(int)]
        tiles = [cv2.imread(str(frames_dir / f"{i:06d}.jpg")) for i in picks]
        sheet = np.vstack([np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(3)])
        cv2.imwrite(str(a.out / "contact_sheet.jpg"), sheet)
        print(f"  contact sheet -> {a.out/'contact_sheet.jpg'} "
              f"(check the horizon is the right way up)")

    # ── ground truth ─────────────────────────────────────────────────────────
    print("ground truth:")
    gt_tar = _fetch(mission, f"data/{GT_TOPIC}.tar", cache / "data" / f"{GT_TOPIC}.tar")
    gt_attrs, gt = _read_zarr_tar(
        gt_tar, want={"timestamp", "pose_pos", "pose_orien"})
    print(f"  {gt_attrs.get('description','')}")
    ts_gt = np.asarray(gt["timestamp"], dtype=np.float64)
    pos_gt = np.asarray(gt["pose_pos"], dtype=np.float64)
    ori_gt = np.asarray(gt["pose_orien"], dtype=np.float64)
    print(f"  {ts_gt.size} samples, {(ts_gt.size-1)/(ts_gt[-1]-ts_gt[0]):.0f} Hz, "
          f"frame {gt_attrs.get('frame_id','?')}")

    # T_enu_cpt7(t) . T_cpt7_base . T_base_cam. On this payload cpt7_imu and box_base
    # coincide exactly (the released tf is identity), so the chain that actually
    # matters is the ~0.4 m camera lever arm, which swings with every body rotation.
    tf_cpt7 = _fetch(mission, f"metadata/{GT_TOPIC.replace('_odometry','_tf')}.yaml",
                     cache / "metadata" / f"{GT_TOPIC.replace('_odometry','_tf')}.yaml")
    tf_cam = _fetch(mission, f"metadata/{a.camera}.yaml",
                    cache / "metadata" / f"{a.camera}.yaml")
    import yaml
    T_base_cpt7 = _transform_from_yaml_like(
        yaml.safe_load(tf_cpt7.read_text())["transform"])
    T_base_cam = _transform_from_yaml_like(
        yaml.safe_load(tf_cam.read_text())["transform"])
    T_cpt7_cam = np.linalg.inv(T_base_cpt7) @ T_base_cam
    lever = float(np.linalg.norm(T_cpt7_cam[:3, 3]))
    print(f"  camera lever arm from CPT7: {lever:.3f} m "
          f"({'compensated' if lever > 1e-6 else 'none'})")

    # Interpolate GT to the kept camera timestamps. 200 Hz against 15 Hz imagery
    # means linear position interpolation is far below the GT's own 0.132 m ATE;
    # orientation is nearest-neighbour for the same reason.
    ts_want = ts_cam[kept_idx]
    inside = (ts_want >= ts_gt[0]) & (ts_want <= ts_gt[-1])
    if not inside.all():
        print(f"  WARNING: {int((~inside).sum())} frames fall outside the GT "
              f"interval and get no row")
    tw = ts_want[inside]
    idx_w = kept_idx[inside]
    p = np.stack([np.interp(tw, ts_gt, pos_gt[:, k]) for k in range(3)], axis=1)
    nn = np.searchsorted(ts_gt, tw).clip(0, ts_gt.size - 1)
    R_body = _quat_to_R(ori_gt[nn])

    T = np.tile(np.eye(4), (tw.size, 1, 1))
    T[:, :3, :3] = R_body
    T[:, :3, 3] = p
    T_cam = T @ T_cpt7_cam
    q_cam = _R_to_quat(T_cam[:, :3, :3])

    gt_path = a.out / "gt_tum.txt"
    with open(gt_path, "w") as f:
        f.write("# GrandTour %s %s -- cpt7_ie_tc, composed to the %s frame\n"
                % (mission, a.camera, a.camera))
        f.write("# frame_index timestamp tx ty tz qx qy qz qw\n")
        for i, k in enumerate(idx_w):
            f.write("%06d %.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
                    % (k, tw[i], *T_cam[i, :3, 3], *q_cam[i]))
    seg_len = float(np.linalg.norm(np.diff(T_cam[:, :3, 3], axis=0), axis=1).sum())
    extent = float(np.linalg.norm(T_cam[:, :3, 3].max(0) - T_cam[:, :3, 3].min(0)))
    print(f"  {tw.size} GT poses -> {gt_path}")
    print(f"  path length {seg_len:.1f} m, bbox diagonal {extent:.1f} m, "
          f"ratio {seg_len/max(extent,1e-9):.2f}")

    # ── provenance ───────────────────────────────────────────────────────────
    rec = {
        "dataset": "GrandTour (leggedrobotics/grand_tour_dataset)",
        "paper": "arXiv 2602.18164",
        "mission_short": a.mission,
        "mission_folder": mission,
        "camera": a.camera,
        "camera_description": cam_attrs.get("description", ""),
        "source": {
            "width": src_w, "height": src_h,
            "distortion_model": ci["distortion_model"],
            "K": ci["K"], "D": ci["D"],
            "n_frames_total": int(n_src),
            "duration_s": round(dur, 3),
            "rate_hz": round(rate, 3),
            "timing_gaps": n_gap,
        },
        "output": {
            "width": out_w, "height": out_h,
            "hfov_deg": round(float(hfov), 3),
            "K": [float(x) for x in K_new.ravel()],
            "rot180": bool(a.rot180),
            "n_frames": int(written),
            "start_s": a.start, "end_s": (None if a.end is None else a.end),
            "stride": a.stride,
            "effective_hz": round(
                float((len(kept_idx) - 1) /
                      max(ts_cam[kept_idx[-1]] - ts_cam[kept_idx[0]], 1e-9)), 3),
        },
        "ground_truth": {
            "topic": GT_TOPIC,
            "description": gt_attrs.get("description", ""),
            "n_poses": int(tw.size),
            "lever_arm_m": round(lever, 4),
            "path_length_m": round(seg_len, 3),
            "bbox_diagonal_m": round(extent, 3),
            "length_over_extent": round(seg_len / max(extent, 1e-9), 3),
        },
    }
    (a.out / "mission.json").write_text(json.dumps(rec, indent=2))
    print(f"\nrun record -> {a.out/'mission.json'}")
    print(f"next: python recon/measure_flow.py --frames {frames_dir}")


if __name__ == "__main__":
    main()
