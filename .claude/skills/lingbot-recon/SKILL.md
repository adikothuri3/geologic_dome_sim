---
name: lingbot-recon
description: Run a LingBot-Map streaming reconstruction with this project's defaults — keyframe_interval, windowed mode, confidence filtering, chunking for long clips — sized for 8 GB VRAM. Use for every video-to-point-cloud run.
---

# lingbot-recon

Video in → camera poses + dense point cloud out, on the 8 GB RTX 4060 Ti. The two VRAM killers are the KV cache and the ~320-view RoPE limit; every default below exists to manage them.

## Run it with

```bash
# 1. video -> frames (tonemaps HDR, applies rotation, --rotate-cw for portrait phone clips)
python recon/extract_frames.py clip.MOV runs/recon/<name>/frames --fps 10 --rotate-cw
# 2. frames -> cloud + trajectory   (frames must live on ext4, NOT /mnt/c -- see below)
~/venvs/lingbot/bin/python recon/reconstruct.py --frames ~/data/recon/<name>/frames \
    --out ~/data/recon/<name>/out --model_path ~/ckpt/lingbot-map/lingbot-map.pt \
    --mode windowed --window_size 24 --keyframe_interval 6 --preprocess_mode crop
# 3. inspect
~/venvs/dome/bin/python recon/inspect_cloud.py <out>     # Open3D stats + renders
~/venvs/dome/bin/python recon/view_viser.py  <out>       # browser, localhost:8080
```

## Defaults

- **Model**: **default to `lingbot-map-long.pt`.** Both checkpoints are downloaded (`~/ckpt/lingbot-map/`, 4.63 GB each) and load into the *same* architecture with no missing keys, so swapping is just `--model_path`. On a 23 s indoor clip the long checkpoint was decisively better than base (ratio 2.58 vs 6.4, and the only run whose render read as a room). Base is now only worth using as an A/B control.
- **Resolution**: 518 long edge (do not raise it on this box). **Feed landscape frames.** Upstream fits width to 518 and only crops height, so a landscape frame becomes 518×294 while the same portrait frame pads to 518×518 — 43% of the token budget spent on white pixels. Rotating portrait footage with `--rotate-cw` measured −1.8 GB VRAM, +28% fps and *more* points, at no cost in field of view.
- **`--preprocess_mode`**: `crop` for landscape input (no padding, no cropping — height lands under 518). `pad` only for portrait input, where it preserves the near-field floor that `crop` would slice off; `reconstruct.py` then masks the padding so it can't inject fake geometry.
- **Mode**: both work and agree exactly at short lengths. `windowed --window_size 24` (32 OOMs) or `streaming --kv_cache_sliding_window 24` (32 OOMs). Streaming holds VRAM flat over any clip length and is what Phases 6–7 will use; windowed is marginally better on medium clips. **Leave inputs on the CPU in both** — `reconstruct.py` does this now; moving the tensor to the GPU for streaming was a bug that made it look unusable.
- **One window is a *frame-count* budget, not a keyframe count.** Upstream's own formula (`gct_stream_window.inference_windowed` docstring):
  `capacity = num_scale_frames + (window_size - num_scale_frames) * keyframe_interval`
  At the ws=24 / nsf=4 / kfi=6 defaults that is exactly **124 frames** — which is why the one old run that looked passable was 124 frames. One frame over and it silently splits into two windows and stitches them. **Check `n_windows_stitched` in `run.json` every time.** `overlap_keyframes` reduces the advance per window, so leave it at `0` when you want a single window.
- **Sampling rate: 10 fps for handheld.** The old "≥5 fps" rule was derived from `traj_length_over_extent`, which is not a quality metric (below). Measured directly on handheld panning footage, going 5 → 10 fps cut the worst pose jump from 0.732 to **0.205**. Halving inter-frame motion is the cheapest real improvement available.
- **Keyframing**: with 10 fps input use `keyframe_interval 5` so the 24-slot cache spans ~12 s rather than ~3 s. In windowed mode each window gets a fresh cache, so the ~320-view RoPE limit is per window; in streaming the sliding window evicts old keyframes, so the cache — not the RoPE limit — is what binds.
- **Raise `--num_scale_frames 8` and `--camera_num_iterations 8`** (both default to 4). Better scale reference and more pose-refinement iterations, for ~0.5 GB VRAM. Used in the best run to date.
- **`--image_size` is not tunable.** The checkpoint's position embeddings are baked for 518 (1370 tokens); anything else fails to load.
- **Confidence filtering**: `--conf_percentile 55` (a percentile, not an absolute threshold, since the confidence scale shifts between runs — it landed at 2.4 on the single-window room clip and 11.6 on a 24-frame one). Converged value pending real trail footage.

> [!warning] `traj_length_over_extent` is a drift *detector*, not a quality gate
> A low ratio is necessary but **nowhere near sufficient**. A 2026-08-05 run hit ratio **2.87** — comfortably "healthy" — while its render was still a torn, streaked mess. The ratio only says the camera path is short relative to the scene; it says nothing about whether surfaces landed on top of each other. **Always look at the renders before calling a reconstruction good.** Use ratio ≲3 and `n_windows_stitched == 1` as a floor, then judge visually.
>
> Measured drift by clip length on indoor low-texture footage: 25 s → 2.8, 49 s → 9.7–11.8, 132 s → 31–46. Windowed fails by scale jumps at window boundaries; streaming fails by smooth accumulating drift once the scene leaves the KV cache. **There is no loop closure** (upstream issues #60/#78, both open) — so never end a clip where it started, or the same wall comes back as a second offset layer.
>
> Practical consequence: **shoot for coverage, not duration**, and keep the whole clip inside one continuous space — walking in through a doorway and back out is the worst case.

> [!tip] Per-frame geometry is excellent; only pose accumulation degrades
> A 6-frame reconstruction of the same footage comes out clean and correctly placed. That means `reconstruct.py`'s depth→world unprojection is sound, and any mess in a long run is pose drift stacking good surfaces in slightly wrong places — visible as "corduroy" ribbing on walls. Do not go bug-hunting in the export; spend the effort on capture and on `lingbot-map-long`.
>
> **Capture rule this implies: translate, don't pan.** Sidestep and walk the camera through the space; a fast rotation in place gives no parallax and is what produces the ribbing.

## Chunking rules for long clips

- Split anything over **~10 minutes** into chunks at natural pauses (turns, stops), with a few seconds of overlap.
- Reconstruct chunks independently; treat each chunk as its own scene until the chunk-stitching question in `notes/open-questions.md` is resolved. Do not silently merge clouds from different chunks.

## Run hygiene

- **Stage frames on ext4 (`~/data/...`), never `/mnt/c`.** Loading ~1300 JPEGs over drvfs with 8 worker threads dies with `OSError: [Errno 12] Cannot allocate memory`. Copy them in first; write outputs wherever.
- **Never run anything else on the GPU during a reconstruction.** An Open3D inspect running alongside a run is what tipped one into the OOM killer.
- Watch `nvidia-smi`; if VRAM climbs toward 8 GB mid-clip, stop and raise `keyframe_interval` or shrink the window — don't ride it out. Keep `--vram_fraction` set: overshooting VRAM under WSL has killed the whole distro, not just the process.
- **Don't trust a pipeline's exit code.** `python ... | grep ...` reports grep's status, so an OOM-killed run looks like a success. Redirect to a log and check `$?` on the python process.
- Inspect every result in the viser viewer (trajectory + cloud) before passing downstream.
- Expect degradation on snow/low-texture and motion blur; lock exposure at capture time (see `notes/capture-protocol.md`).
- Log each run as a row in `notes/experiments.md` (`n_envs` = `—`), commit hash included.

Output feeds the `open3d-cleanup` skill.
