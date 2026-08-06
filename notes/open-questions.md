---
title: Open questions & hard problems
updated: 2026-08-06
status: current
---

# Open questions & hard problems

The known-hard parts of the pipeline and what's genuinely unresolved. When one gets answered, move the answer to its owning note ([[pipeline]], [[setup]], [[capture-protocol]]) and delete it here.

## Snow / low-texture reconstruction failure

Monocular models need texture; snow is texture-poor and overexposed, so LingBot-Map is expected to degrade on exactly the terrain that makes this Everest. Mitigation plan: test on white/low-texture proxies early (a white wall ≈ snow proxy, Phase 3), lock exposure, and expect best results on the rocky sections — most of the route below C2 is rock. **Open:** where exactly the failure threshold is, and whether any capture rule ([[capture-protocol]]) moves it.

**First proxy result (2026-08-05):** `room_map.MOV` — plain cream walls and plain carpet — is
a decent snow proxy, and per-frame geometry held up fine: within a single window the room is
coherent and the trajectory is smooth. What broke was *cross-window scale alignment* (see
below), i.e. low texture cost us global consistency long before it cost us local depth. Two
other failure modes visible in the cloud: radial "flying pixel" streaks at occlusion edges
(doorframes), and the bright window blowing out to no geometry. A real snow test still needs
outdoor overexposed footage.

## Scale calibration

Monocular reconstruction has no absolute scale, and terrain that's 10% too large changes step heights the policy trains on. Candidate calibrations: measured markers (home), Pemba's known dimensions, GPS track length (expedition).

**Partly answered (2026-08-06).** `recon/calibrate_scale.py` ships a working *camera-height* anchor: fit the ground plane, take the median camera height above it, divide into an assumed eye height. On `example/loop` it gives 2.4938 m/unit and an 18 × 11 × 27 m office floor — plausible against every cross-check (0.7 cm point spacing, 2.3 m structure height, 89 m walked). Two lessons worth keeping:

- **Do not pick the ground plane by inlier count.** A corridor's walls outvote its carpet, and the first fit returned camera heights spanning 0.33–6.22 units. Pick the plane that keeps camera height *constant* — that is what "floor" physically means for a walk. The chosen plane is then handed to `clean_cloud.py` rather than re-fitted.
- **The anchor dies with the trajectory.** It needs trustworthy poses, so the script refuses above `traj_length_over_extent` 6. On the courthouse run (ratio 25) there is no camera anchor at all and scale would have to come from a known object.

**Still open:** the accuracy. Eye height was *assumed* at 1.5 m, and the p5–p95 camera-height spread is 37% of the median, so this is good to maybe ±15% — untested against any measured ground truth, and nobody has established what scale error locomotion fine-tuning actually tolerates. The expedition anchors (markers, GPS, Pemba's dimensions) plug into the same `--anchor factor` path but remain unexercised.

## Drift on very long sequences

Even streaming models accumulate pose error over a trek-length walk. Mitigation: chunk the trek (≤10-min pieces), `lingbot-map-long`, keyframe interval, windowed mode. **Open:** how much drift survives chunking, and whether chunks can be stitched into one consistent terrain or must stay separate scenes.

**First hard evidence (2026-08-05, indoor):** cross-window stitching failed badly on a
2-minute room walkthrough. `inference_windowed` warps each window into the first window's
frame with a similarity transform estimated from overlapping keyframes; on plain cream walls
that estimate goes wrong. 13 windows gave a camera path **34×** the scene extent with a 2.14
pose jump at a window boundary (median step 0.071); widening to 9 windows with triple the
overlap made it **38×**. The same clip cut to a single window is clean at **2.8×**. So the
failure is scale estimation on low-texture overlap, *not* insufficient overlap — which also
means this is entangled with the snow problem above, since snow is the same texture regime.
**Streaming does not dodge this (2026-08-05).** Streaming mode has no stitching at all, and it
runs the whole clip in flat memory — but it drifts instead: 25 s → 2.8, 49 s → 11.8, 132 s → 34.
Windowed on the same footage gives 2.8 / 9.7 / 31. **The two modes agree exactly at 25 s and fail
together past it**, because the KV cache that keeps streaming memory bounded is also the model's
memory horizon (~24 keyframes ≈ 14 s on 8 GB). So this is one problem with two faces, not two
problems: *global consistency decays once the scene leaves the cache*, showing up as scale jumps
at boundaries (windowed) or smooth drift (streaming).

The practical rule is ~25 s of footage per globally consistent scene on this hardware, so
Phase 4 terrain should be built per-segment. **Open:** how much a larger GPU actually buys —
cache size is the lever, and this makes the cloud-GPU decision a *correctness* requirement for
Phase 7, not just a throughput one. Also untested: `lingbot-map-long` (built for this regime,
not yet downloaded) and an external pose prior from Pemba's odometry, which is the one signal
the expedition has that a phone clip does not.

## Sim can't model snow physics

MuJoCo contacts are rigid — no sinkage, no compliance. This is a scope boundary, not a bug: the pipeline targets *geometry* (rocks, slopes, steps); GenTe-style force modeling is explicitly future work. **Open:** whether `solref`/`solimp` tuning can fake enough compliance to be worth doing.

## Overfitting to reconstruction artifacts

Reconstruction noise becomes "features" the policy exploits. Mitigation is terrain randomization around the reconstruction (Phase 5, see [[pipeline]]). **Open:** how much randomization is enough — measurable as the gap between performance on the raw recon vs. perturbed variants.

## Upstream's memory horizon is unreachable here

`demo.py` defaults to `kv_cache_sliding_window=64` with `keyframe_interval=1` for anything under the 320-view RoPE limit, and that is the config behind every published LingBot-Map demo. **We cannot run it.** Measured Aug 6 on `example/courthouse` (286 frames): kvsw 64 OOMs, 48 OOMs, 24 OOMs at frame 240/286, and only 16 completes. Halving the sequence to 143 frames with `--stride 2` did *not* buy a bigger cache — **kvsw ≈ 24 is a hard ceiling set by cache size, not sequence length**. `lingbot-map-long` is bigger still and cannot run 286 frames at all.

The consequence is not cosmetic: at kvsw 16 the courthouse poses scribble (`traj_length_over_extent` **24.9** against loop's 3.36) while per-frame geometry stays excellent. Doubling the horizon via `keyframe_interval 2` moved it to 25.1 — nothing. So this is a cliff, not a gradient: either the scene fits the cache or global consistency collapses. Upstream's own courthouse result is reproducible only on a bigger card.

**Open:** where exactly the cliff is in cache-views-per-scene, and whether it is the *number* of cached views or their *temporal span* that matters — the `keyframe_interval 2` result hints at the former, which would make VRAM the only lever and settle the cloud-GPU question for Phase 7.

## 8 GB VRAM ceiling

LingBot-Map's KV cache and MJX's env count both eat VRAM ([[setup]] has the per-workload tactics). **Open:** whether Phase 5 per-iteration debugging genuinely fits locally, and the cloud-GPU budget/choice for sweeps and the expedition window.

## Closed loop on the expedition

Redeploying fine-tuned policies onto Pemba mid-expedition may be too risky, and the G1 policy stack is partly closed (factory controller ≠ our policy). Current posture is sim-validated recommendations only ([[decisions]]). **Open:** what evidence bar would make expedition leads green-light a real redeployment.
