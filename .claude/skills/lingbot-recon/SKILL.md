---
name: lingbot-recon
description: Run a LingBot-Map streaming reconstruction with this project's defaults — keyframe_interval, windowed mode, confidence filtering, chunking for long clips — sized for 8 GB VRAM. Use for every video-to-point-cloud run.
---

# lingbot-recon

Video in → camera poses + dense point cloud out, on the 8 GB RTX 4060 Ti. The two VRAM killers are the KV cache and the ~320-view RoPE limit; every default below exists to manage them.

## Defaults

- **Model**: `lingbot-map-long` for anything outdoors / longer than a couple of minutes; base model only for short indoor tests.
- **Resolution**: 518×378 (the standard inference size; do not raise it on this box).
- **Keyframing**: set `keyframe_interval` so total keyframes stay comfortably under the ~320-view RoPE limit — for a clip of N seconds at ~20 fps, interval ≥ `N * 20 / 300`. Prefer windowed mode on top for anything over ~2 minutes.
- **Confidence filtering**: export with the confidence channel and drop low-confidence points before saving — first defense against noise; the threshold that proves out on real trail footage becomes the default here (record it when converged).

## Chunking rules for long clips

- Split anything over **~10 minutes** into chunks at natural pauses (turns, stops), with a few seconds of overlap.
- Reconstruct chunks independently; treat each chunk as its own scene until the chunk-stitching question in `notes/open-questions.md` is resolved. Do not silently merge clouds from different chunks.

## Run hygiene

- Watch `nvidia-smi`; if VRAM climbs toward 8 GB mid-clip, stop and raise `keyframe_interval` or shrink the window — don't ride it out.
- Inspect every result in the viser viewer (trajectory + cloud) before passing downstream.
- Expect degradation on snow/low-texture and motion blur; lock exposure at capture time (see `notes/capture-protocol.md`).
- Log each run as a row in `notes/experiments.md` (`n_envs` = `—`), commit hash included.

Output feeds the `open3d-cleanup` skill.
