"""LingBot-Map reconstruction with point-cloud + trajectory export.

Phase 3. Upstream's `demo.py` runs inference and opens the viser viewer, but it
never writes anything to disk -- close the browser tab and the reconstruction is
gone. Phase 3 needs an exported cloud and camera trajectory that Phase 4 can turn
into MuJoCo terrain, so this wraps the same model API and saves:

    <out>/cloud.ply        confidence-filtered coloured point cloud (binary PLY)
    <out>/trajectory.npz   extrinsics (c2w), intrinsics, camera centres
    <out>/run.json         config + timing + VRAM peak, for notes/experiments.md

We deliberately reuse upstream `demo.py`'s loader/postprocess helpers rather than
reimplementing them, so a checkpoint or pose-convention change upstream can't
silently desync our export from their viewer.

Defaults follow .claude/skills/lingbot-recon (8 GB VRAM box): SDPA attention (no
FlashInfer), windowed mode for long clips, CPU offload of results, and keyframing
that keeps the keyframe count under the ~320-view RoPE limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

LINGBOT_SRC = Path(os.environ.get("LINGBOT_SRC", Path.home() / "src" / "lingbot-map"))

# Upstream sets PYTORCH_CUDA_ALLOC_CONF at import time (before torch), so importing
# demo must happen before we touch torch ourselves.
sys.path.insert(0, str(LINGBOT_SRC))
import demo as lbdemo  # noqa: E402
import torch  # noqa: E402

# RoPE positional encoding was trained to ~320 views; beyond that pose drift grows
# sharply. Upstream README's windowed guidance and our skill both key off this.
ROPE_VIEW_LIMIT = 320


class Args:
    """demo.load_model/load_images read attributes off an argparse Namespace."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def load_model_lowmem(margs, device: torch.device):
    """Build GCTStream and load weights without ever holding two full copies.

    The checkpoint is a flat 1.16 B-param fp32 state dict -- 4.63 GB. Upstream's
    load_model() does torch.load(map_location=device), which on this box fails
    twice over: 4.63 GB of checkpoint on an 8 GB GPU plus a second allocation when
    model.to(device) runs, and 4.63 GB of checkpoint plus 4.63 GB of freshly
    constructed model in only 7.7 GB of WSL RAM.

    mmap=True keeps the checkpoint disk-backed and evictable, so the only real
    resident copy is the model itself. We cast to bf16 before the GPU transfer so
    the card never sees the fp32 trunk.
    """
    if getattr(margs, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    print("building model...")
    model = GCTStream(
        img_size=margs.image_size,
        patch_size=margs.patch_size,
        enable_3d_rope=margs.enable_3d_rope,
        max_frame_num=margs.max_frame_num,
        kv_cache_sliding_window=margs.kv_cache_sliding_window,
        kv_cache_scale_frames=margs.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=margs.use_sdpa,
        camera_num_iterations=margs.camera_num_iterations,
    )

    print(f"loading checkpoint (mmap): {margs.model_path}")
    ckpt = torch.load(margs.model_path, map_location="cpu", mmap=True, weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys, e.g. {missing[:3]}", file=sys.stderr)
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    del ckpt, state_dict

    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if getattr(model, "aggregator", None) is not None:
            print(f"casting aggregator to {dtype} (prediction heads stay fp32)")
            model.aggregator = model.aggregator.to(dtype=dtype)

    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"weights resident on GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, dtype


def write_ply(path: Path, xyz: np.ndarray, rgb01: np.ndarray) -> None:
    """Binary little-endian PLY. Open3D and MeshLab both read this directly."""
    rgb8 = (np.clip(rgb01, 0.0, 1.0) * 255).astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dt = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    arr = np.empty(len(xyz), dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb8[:, 0], rgb8[:, 1], rgb8[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())


def conf_threshold_from_percentile(conf: np.ndarray, pct: float) -> float:
    """Percentile threshold on a strided subsample (full array is ~GB-scale)."""
    flat = conf[:: max(1, len(conf) // 20)].ravel()
    return float(np.percentile(flat, pct))


def padding_mask(images: np.ndarray) -> np.ndarray:
    """True where a pixel is real image content rather than letterbox padding.

    `mode="pad"` pads the short axis with constant 1.0 (white) to reach a square.
    Those pixels are not observations -- the depth head still emits a value for
    them, and unprojecting it sprays a plane of fake geometry into the cloud.

    We detect the border rather than recomputing upstream's padding arithmetic, so
    a change in their resize logic can't silently desync us: a padded row/column is
    exactly 1.0 in every channel of every frame, which real (JPEG-compressed)
    content effectively never is.
    """
    exact_white = np.isclose(images, 1.0, atol=1e-6).all(axis=1).all(axis=0)  # (H, W)
    valid = ~exact_white

    # Only trust it as padding if it forms a contiguous border, not an interior
    # blown-out highlight (a window, a lamp).
    h, w = valid.shape
    mask = np.ones((h, w), dtype=bool)
    col_pad = exact_white.all(axis=0)
    row_pad = exact_white.all(axis=1)
    left = np.argmax(~col_pad) if col_pad.any() else 0
    right = w - np.argmax(~col_pad[::-1]) if col_pad.any() else w
    top = np.argmax(~row_pad) if row_pad.any() else 0
    bottom = h - np.argmax(~row_pad[::-1]) if row_pad.any() else h
    mask[:, :left] = False
    mask[:, right:] = False
    mask[:top, :] = False
    mask[bottom:, :] = False
    return mask


def sky_session(model_path: Path):
    """ONNX session for upstream's skyseg net, or None if it can't be had.

    Outdoor scenes need this. Confidence filtering does not remove sky: an
    overcast sky is a large, smooth, *confidently* wrong surface, so the model
    happily places it at some arbitrary depth and the cloud grows a halo that
    dwarfs the actual scene. Indoors it is pure cost, hence opt-in.
    """
    import onnxruntime

    if not model_path.exists():
        raise SystemExit(
            f"--mask_sky needs {model_path}\n"
            "  fetch: curl -L -o skyseg.onnx "
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
        )
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in onnxruntime.get_available_providers()]
    # CPU on purpose: the GPU is at its VRAM ceiling from the aggregator, and
    # 320x320 segmentation over a few hundred frames is a couple of seconds.
    return onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def build_cloud(preds: dict, images: np.ndarray, conf_pct: float, pixel_stride: int,
                skyseg=None, sky_threshold: float = 0.1, conf_abs: float | None = None):
    """Confidence-filtered coloured cloud, built one frame at a time.

    Streaming matters here: a 1320-frame clip at 518x518 would need ~4 GB just for
    an (F, H, W, 3) float32 world-point array, on top of the predictions already
    resident, and WSL only has 7.7 GB. We never materialise more than one frame of
    geometry plus the surviving points.

    The point head gives world coordinates directly, but the windowed inference
    path returns only depth (it aligns windows via depth + per-window transforms),
    so we unproject through the predicted intrinsics/extrinsics -- the same maths
    gct_base uses internally.
    """
    from lingbot_map.utils.geometry import depth_to_world_coords_points

    have_world = preds.get("world_points") is not None
    conf = preds["world_points_conf"] if have_world else preds["depth_conf"]
    thr = (float(conf_abs) if conf_abs is not None
           else conf_threshold_from_percentile(conf.astype(np.float32), conf_pct))

    n = conf.shape[0]
    content = padding_mask(images)
    stride_mask = np.zeros(conf.shape[1:], dtype=bool)
    stride_mask[::pixel_stride, ::pixel_stride] = True
    base_mask = stride_mask & content
    dropped_to_padding = int((~content).sum())

    if skyseg is not None:
        from lingbot_map.vis.sky_segmentation import segment_sky_from_array
        h, w = conf.shape[1:]
    sky_dropped = 0

    pts, cols = [], []
    for i in range(n):
        if have_world:
            wp = preds["world_points"][i].astype(np.float32)
            valid = np.isfinite(wp).all(axis=-1)
        else:
            d = preds["depth"][i]
            d = (d[..., 0] if d.ndim == 3 else d).astype(np.float32)
            e = np.eye(4, dtype=np.float32)
            e[:3, :4] = preds["extrinsic"][i]
            w2c = np.linalg.inv(e)[:3, :4]      # helper wants world-to-camera
            wp, _, valid = depth_to_world_coords_points(
                d, w2c, preds["intrinsic"][i].astype(np.float32)
            )
            valid = valid & np.isfinite(wp).all(axis=-1)

        m = (conf[i] > thr) & base_mask & valid
        if skyseg is not None:
            # segment on the *preprocessed* frame, so the mask lands on exactly
            # the grid the depths live on -- no crop/resize alignment to get wrong.
            non_sky = segment_sky_from_array(images[i], skyseg, h, w)
            keep = non_sky > sky_threshold
            sky_dropped += int((m & ~keep).sum())
            m = m & keep
        if m.any():
            pts.append(wp[m].astype(np.float32))
            cols.append(np.transpose(images[i], (1, 2, 0))[m].astype(np.float32))

    if not pts:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                thr, dropped_to_padding, sky_dropped)
    return (np.concatenate(pts, 0), np.concatenate(cols, 0),
            thr, dropped_to_padding, sky_dropped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True, help="folder of extracted JPEGs")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--mode", choices=["streaming", "windowed"], default="windowed")
    ap.add_argument("--window_size", type=int, default=48)
    # The paper's adaptive keyframe selection (arXiv 2604.14141 sec 4.4): predict pose
    # and depth for each incoming frame, compute optical flow against the most recent
    # keyframe, and promote the frame only once that flow exceeds a threshold. It
    # takes precedence over --keyframe_interval. Upstream's own demo-video pipeline
    # (demo_render/process_videos.sh) runs 25.0 px with a 100-frame cap; `demo.py`
    # exposes it nowhere, which is why the README one-liner is a weaker configuration
    # than the one behind their published clips.
    #
    # Windowed only: gct_stream.GCTStream (streaming/"Direct") does not implement it.
    ap.add_argument("--flow_threshold", type=float, default=0.0,
                    help="mean optical flow in px before a frame becomes a keyframe; "
                         "0 disables (fixed --keyframe_interval instead). Windowed only")
    ap.add_argument("--max_non_keyframe_gap", type=int, default=30,
                    help="force a keyframe after this many skipped frames (flow mode only)")
    ap.add_argument("--overlap_keyframes", type=int, default=4,
                    help="keyframes shared between consecutive windows; upstream's "
                         "preferred overlap control (overlap_size is the legacy path). "
                         "Negative means 'unset', which is what upstream's own scripts "
                         "pass and resolves to num_scale_frames inside the model")
    ap.add_argument("--keyframe_interval", type=int, default=None,
                    help="default: auto, to stay under the ~320-view RoPE limit")
    ap.add_argument("--num_scale_frames", type=int, default=4)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=48)
    ap.add_argument("--camera_num_iterations", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1, help="input frame stride")
    ap.add_argument("--first_k", type=int, default=None, help="only first K frames (smoke tests)")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--preprocess_mode", choices=["pad", "crop"], default="pad",
                    help="'pad' keeps the whole frame (portrait-safe); 'crop' is "
                         "upstream's default and centre-crops height to 518")
    ap.add_argument("--conf_percentile", type=float, default=55.0,
                    help="drop points below this percentile of world_points_conf")
    # Upstream filters on an absolute confidence value, not a percentile: demo.py's
    # viewer defaults to 1.5 and their demo-video renderer to 2.0. A percentile keeps
    # a fixed *fraction* of every run, which is the right default for comparing our
    # own runs to each other, and the wrong one for comparing a cloud to theirs.
    ap.add_argument("--conf_threshold", type=float, default=None,
                    help="absolute confidence cut, overriding --conf_percentile "
                         "(upstream: 1.5 in demo.py's viewer, 2.0 in their renderer)")
    ap.add_argument("--mask_sky", action="store_true",
                    help="drop sky pixels from the exported cloud (outdoor scenes; "
                         "upstream uses this for example/courthouse and university)")
    ap.add_argument("--skyseg_model", type=Path,
                    default=Path.home() / "ckpt" / "skyseg.onnx")
    ap.add_argument("--sky_threshold", type=float, default=0.1,
                    help="keep pixels whose non-sky confidence exceeds this "
                         "(upstream's _SKYSEG_SOFT_THRESHOLD)")
    ap.add_argument("--pixel_stride", type=int, default=3,
                    help="spatial subsample when building the cloud; every frame "
                         "contributes a full depth map, so stride 1 on a 1000+ frame "
                         "clip is tens of millions of points for no extra detail")
    ap.add_argument("--vram_fraction", type=float, default=0.85,
                    help="hard cap on the fraction of VRAM PyTorch may allocate")
    ap.add_argument("--viser", action="store_true", help="open the viser viewer after export")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point_size", type=float, default=0.002)
    ap.add_argument("--downsample_factor", type=int, default=4)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device -- this will be extremely slow", file=sys.stderr)
    else:
        # Cap PyTorch below the physical ceiling. Without this, overshooting VRAM
        # under WSL does not raise a clean torch OOM -- it surfaces as
        # "CUDA driver error: device not ready" and, once, took the entire WSL VM
        # down with Wsl/Service/E_UNEXPECTED. A hard fraction makes the failure a
        # catchable Python exception instead of a destabilised driver.
        torch.cuda.set_per_process_memory_fraction(a.vram_fraction)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM cap: {a.vram_fraction:.0%} of {total:.1f} GB "
              f"= {a.vram_fraction*total:.1f} GB")

    # ── Load frames ──────────────────────────────────────────────────────────
    t0 = time.time()
    paths = sorted(
        p for p in a.frames.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if a.first_k:
        paths = paths[: a.first_k]
    if a.stride > 1:
        paths = paths[:: a.stride]
    paths = [str(p) for p in paths]

    # We call load_and_preprocess_images directly instead of demo.load_images
    # because the latter hardcodes mode="crop", which fixes width at 518 and
    # centre-crops height. Our footage is portrait (921 px tall after the width
    # fit), so "crop" would silently discard ~44% of every frame -- including the
    # near-field floor, which is the part Phase 4 turns into terrain. "pad" keeps
    # the full field of view.
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    print(f"loading {len(paths)} images (mode={a.preprocess_mode})...")
    images = load_and_preprocess_images(
        paths, mode=a.preprocess_mode, image_size=a.image_size, patch_size=14
    )
    resolved_folder = str(a.frames)
    n_frames = images.shape[0]
    print(f"loaded {n_frames} frames {tuple(images.shape)} in {time.time()-t0:.1f}s")

    # Keyframe interval.
    #
    # Streaming mode keeps one KV cache for the whole sequence, so total keyframes
    # must stay under the ~320-view RoPE limit. Windowed mode starts a fresh cache
    # per window, so the limit applies per window (window_size counts *keyframes*),
    # and the interval only sets temporal density -- ~0.6 s between keyframes at
    # our 10 fps sampling is dense enough for a walking-pace clip without paying
    # for windows we don't need.
    # Flow-based selection replaces the fixed interval entirely, so the auto-interval
    # arithmetic below would only produce a misleading number in the run record.
    if a.flow_threshold > 0 and a.mode != "windowed":
        raise SystemExit(
            "--flow_threshold needs --mode windowed: adaptive keyframe selection lives "
            "in gct_stream_window.GCTStream, and streaming ('Direct' in the paper) has "
            "no implementation of it"
        )

    kfi = a.keyframe_interval
    if a.flow_threshold > 0:
        kfi = 1 if kfi is None else kfi
        print(f"flow-based keyframes: >{a.flow_threshold:.1f} px mean flow, "
              f"forced every {a.max_non_keyframe_gap} frames "
              f"(--keyframe_interval is ignored in this mode)")
    elif kfi is None:
        if a.mode == "streaming":
            kfi = max(1, int(np.ceil(n_frames / (ROPE_VIEW_LIMIT * 0.75))))
            print(f"auto keyframe_interval={kfi} -> ~{n_frames // kfi} keyframes "
                  f"(streaming; RoPE limit ~{ROPE_VIEW_LIMIT})")
        else:
            kfi = 6
            n_kf = n_frames // kfi
            print(f"auto keyframe_interval={kfi} -> ~{n_kf} keyframes across "
                  f"~{max(1, int(np.ceil(n_kf / max(a.window_size - a.overlap_keyframes, 1))))} windows "
                  f"of {a.window_size} keyframes")
    if a.mode == "windowed" and a.window_size > ROPE_VIEW_LIMIT:
        print(f"WARNING: window_size {a.window_size} exceeds the ~{ROPE_VIEW_LIMIT}-view "
              f"RoPE limit; pose drift will grow", file=sys.stderr)

    # ── Preflight: will the CPU-side predictions fit in RAM? ─────────────────
    #
    # Inference accumulates its outputs on the CPU in fp32: the input images plus
    # a full-resolution depth map and confidence map per frame. That is the real
    # ceiling on this box, and it is reached *during* inference -- long after the
    # point where failing is cheap. A 1320-frame clip needs ~7.1 GB against WSL's
    # 7.7 GB and gets OOM-killed ~20 minutes in, so estimate first and bail early.
    h, w = images.shape[-2:]
    bytes_per_frame = 4 * (3 * h * w + h * w + h * w)   # images + depth + depth_conf
    need_gb = n_frames * bytes_per_frame / 1e9
    avail_gb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_gb = int(line.split()[1]) * 1024 / 1e9
                    break
    except OSError:
        pass

    print(f"preflight: predictions need ~{need_gb:.1f} GB of CPU RAM"
          + (f", {avail_gb:.1f} GB available" if avail_gb else ""))
    if avail_gb is not None and need_gb > 0.75 * avail_gb:
        max_frames = int(0.75 * avail_gb * 1e9 / bytes_per_frame)
        print(
            f"\nABORT: ~{need_gb:.1f} GB needed but only {avail_gb:.1f} GB available.\n"
            f"  This box fits about {max_frames} frames at {w}x{h}.\n"
            f"  Re-run with --stride {int(np.ceil(n_frames / max(max_frames, 1)))} "
            f"(or fewer frames) to stay inside RAM.",
            file=sys.stderr,
        )
        return 2

    # ── Model ────────────────────────────────────────────────────────────────
    model_args = Args(
        mode=a.mode,
        image_size=a.image_size,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=max(1024, n_frames + 8),
        kv_cache_sliding_window=a.kv_cache_sliding_window,
        num_scale_frames=a.num_scale_frames,
        use_sdpa=True,          # FlashInfer intentionally not installed; see setup_lingbot.sh
        camera_num_iterations=a.camera_num_iterations,
        model_path=str(a.model_path),
    )
    model, dtype = load_model_lowmem(model_args, device)

    # Leave inputs on the CPU in BOTH modes. Each slices on whichever device
    # `images` lives on and moves only what it needs to the GPU, so peak VRAM is
    # O(window) or O(1) frames rather than O(sequence). At 1320 frames the full
    # fp32 tensor is 4.25 GB -- more than half this card.
    #
    # This is the point of streaming: the model is causal and keeps bounded state,
    # so a live camera feed never has more than a frame in flight. Materialising
    # the whole clip on the GPU first defeats exactly that property.
    print("keeping inputs on CPU; frames move to GPU as needed")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"running {a.mode} inference (dtype={dtype})...")
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        if a.mode == "streaming":
            preds = model.inference_streaming(
                images,
                num_scale_frames=a.num_scale_frames,
                keyframe_interval=kfi,
                output_device=torch.device("cpu"),
            )
        else:
            preds = model.inference_windowed(
                images,
                window_size=a.window_size,
                overlap_size=None,
                overlap_keyframes=(a.overlap_keyframes if a.overlap_keyframes >= 0
                                   else None),
                num_scale_frames=a.num_scale_frames,
                keyframe_interval=kfi,
                flow_threshold=a.flow_threshold,
                max_non_keyframe_gap=a.max_non_keyframe_gap,
                output_device=torch.device("cpu"),
            )
    infer_s = time.time() - t0
    print("prediction keys: " + ", ".join(
        f"{k}{tuple(v.shape)}" if hasattr(v, "shape") else k for k, v in preds.items()
    ))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    print(f"inference done in {infer_s:.1f}s ({n_frames/infer_s:.2f} fps), peak VRAM {peak_gb:.2f} GB")

    images_for_post = preds.get("images", images)
    del images
    if device.type == "cuda":
        torch.cuda.empty_cache()

    preds, images_cpu = lbdemo.postprocess(preds, images_for_post)

    # Halve the CPU footprint of the bulk tensors before they become numpy.
    # At 1320 frames the fp32 images/depth/conf trio is ~7 GB against 7.7 GB of
    # WSL RAM; fp16 is ample for colour and for a depth map we then filter by
    # confidence anyway (each frame is cast back to fp32 for the unprojection).
    for k in ("images", "depth", "depth_conf"):
        if isinstance(preds.get(k), torch.Tensor) and preds[k].dtype == torch.float32:
            preds[k] = preds[k].half()
    if isinstance(images_cpu, torch.Tensor) and images_cpu.dtype == torch.float32:
        images_cpu = images_cpu.half()

    vis = lbdemo.prepare_for_visualization(preds, images_cpu)
    del preds, images_cpu

    # ── Export cloud ─────────────────────────────────────────────────────────
    skyseg = sky_session(a.skyseg_model) if a.mask_sky else None
    pts, cols, thr, n_pad, n_sky = build_cloud(
        vis, vis["images"], a.conf_percentile, a.pixel_stride,
        skyseg=skyseg, sky_threshold=a.sky_threshold, conf_abs=a.conf_threshold,
    )
    if n_sky:
        print(f"masked {n_sky:,} sky points "
              f"({100*n_sky/max(n_sky+len(pts),1):.0f}% of what survived confidence)")
    if n_pad:
        print(f"masked {n_pad:,} letterbox-padding pixels per frame "
              f"({100*n_pad/vis['images'].shape[-1]**2:.0f}% of each frame)")
    ply_path = a.out / "cloud.ply"
    write_ply(ply_path, pts, cols)
    extent = (pts.max(0) - pts.min(0)) if len(pts) else np.zeros(3)
    print(f"cloud: {len(pts):,} points -> {ply_path}")
    print(f"       conf threshold {thr:.3f} "
          f"({'absolute' if a.conf_threshold is not None else f'p{a.conf_percentile:g}'}), "
          f"extent {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} (arbitrary scale)")

    # ── Export trajectory ────────────────────────────────────────────────────
    extrinsic = vis["extrinsic"]              # (F, 3, 4) camera-to-world
    intrinsic = vis["intrinsic"]              # (F, 3, 3)
    cam_centers = extrinsic[:, :3, 3]
    traj_path = a.out / "trajectory.npz"
    np.savez_compressed(
        traj_path,
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        cam_centers=cam_centers.astype(np.float32),
        frame_paths=np.array([str(p) for p in paths]),
    )
    # Cross-window alignment diagnostics. Each window is brought into the first
    # window's frame by a similarity transform; a scale far from 1.0 means the
    # pairwise alignment failed to find matching keyframes in the overlap, which
    # shows up downstream as a trajectory far longer than the scene it sits in.
    chunk_scales = vis.get("chunk_scales")
    n_chunks = 1
    scale_span = 1.0
    if chunk_scales is not None:
        cs = np.asarray(chunk_scales).ravel().astype(np.float64)
        n_chunks = int(cs.size)
        if cs.size and cs.min() > 0:
            scale_span = float(cs.max() / cs.min())
        print(f"windows stitched: {n_chunks}, per-window scale "
              f"min {cs.min():.3f} max {cs.max():.3f} (span {scale_span:.1f}x)")

    kf_mask = vis.get("is_keyframe")
    if kf_mask is not None:
        n_keyframes = int(np.asarray(kf_mask).astype(bool).sum())
        print(f"keyframes selected: {n_keyframes}/{n_frames} "
              f"({100*n_keyframes/max(n_frames,1):.0f}%)"
              + ("  -- every frame cleared the flow threshold; the input is sampled "
                 "more sparsely than the keyframe policy targets"
                 if n_keyframes == n_frames and a.flow_threshold > 0 else ""))
    elif a.flow_threshold > 0:
        n_keyframes = None
    else:
        n_keyframes = int(n_frames // kfi)

    seg = np.linalg.norm(np.diff(cam_centers, axis=0), axis=1)
    print(f"trajectory: {len(cam_centers)} poses -> {traj_path}")
    print(f"            path length {seg.sum():.2f} (arbitrary units), "
          f"max step {seg.max():.3f}" if len(seg) else "")

    # ── Run record ───────────────────────────────────────────────────────────
    record = {
        "frames_dir": str(a.frames),
        "n_frames": int(n_frames),
        "mode": a.mode,
        "window_size": a.window_size,
        "overlap_keyframes": a.overlap_keyframes if a.overlap_keyframes >= 0 else None,
        "keyframe_interval": int(kfi),
        # Under flow selection the model decides this per frame, so a count derived
        # from the interval would be fiction. It returns an `is_keyframe` mask, and
        # the *fraction* selected is the diagnostic: at 100% every frame cleared the
        # flow threshold, which means the footage is sampled more sparsely than
        # upstream's own keyframe spacing and there is no dense tracking in between.
        "n_keyframes": n_keyframes,
        "keyframe_frac": (None if n_keyframes is None
                          else round(n_keyframes / max(n_frames, 1), 3)),
        "flow_threshold": a.flow_threshold,
        "max_non_keyframe_gap": a.max_non_keyframe_gap if a.flow_threshold > 0 else None,
        "num_scale_frames": a.num_scale_frames,
        "kv_cache_sliding_window": a.kv_cache_sliding_window,
        "image_size": a.image_size,
        "model_path": str(a.model_path),
        "dtype": str(dtype),
        "use_sdpa": True,
        "inference_s": round(infer_s, 1),
        "fps": round(n_frames / infer_s, 3),
        "peak_vram_gb": round(peak_gb, 2),
        "conf_percentile": None if a.conf_threshold is not None else a.conf_percentile,
        "conf_threshold": round(thr, 4),
        "conf_mode": "absolute" if a.conf_threshold is not None else "percentile",
        "pixel_stride": a.pixel_stride,
        "preprocess_mode": a.preprocess_mode,
        "padding_px_masked": int(n_pad),
        "mask_sky": bool(a.mask_sky),
        "sky_points_masked": int(n_sky),
        "n_points": int(len(pts)),
        "extent_arbitrary": [round(float(v), 3) for v in extent],
        "traj_length_arbitrary": round(float(seg.sum()), 3) if len(seg) else 0.0,
        "n_windows_stitched": n_chunks,
        "window_scale_span": round(scale_span, 3),
        # A camera path far longer than the scene it moves through means the
        # per-window alignment drifted; near 1 or below is healthy.
        "traj_length_over_extent": round(
            float(seg.sum() / max(extent.max(), 1e-9)), 2) if len(seg) else 0.0,
    }
    (a.out / "run.json").write_text(json.dumps(record, indent=2))
    print(f"run record -> {a.out / 'run.json'}")

    # ── Optional viewer ──────────────────────────────────────────────────────
    if a.viser:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=vis,
            port=a.port,
            vis_threshold=thr,
            downsample_factor=a.downsample_factor,
            point_size=a.point_size,
            mask_sky=False,
            image_folder=resolved_folder,
        )
        print(f"viser at http://localhost:{a.port}  (ctrl-c to stop)")
        viewer.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
