---
title: Open questions & hard problems
updated: 2026-08-07
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

**That test now has a target (2026-08-06):** GrandTour **SNOW-2** (`2024-11-03-07-57-34`,
261 s / 173.9 m), reachable through `recon/fetch_grandtour.py --mission snow-2` and carrying the
same CPT7 ground truth as EIG-1 — so for the first time the low-texture failure can be measured
in metres instead of described. Prediction on record: raising `--overlap_keyframes` to 16 will
*not* rescue it, because tripling the overlap on `room_map` (Aug 4) did not, and the failure
there was scale estimation on low-texture overlap rather than overlap size.

## Scale calibration

Monocular reconstruction has no absolute scale, and terrain that's 10% too large changes step heights the policy trains on. Candidate calibrations: measured markers (home), Pemba's known dimensions, GPS track length (expedition).

**Partly answered (2026-08-06).** `recon/calibrate_scale.py` ships a working *camera-height* anchor: fit the ground plane, take the median camera height above it, divide into an assumed eye height. On `example/loop` it gives 2.4938 m/unit and an 18 × 11 × 27 m office floor — plausible against every cross-check (0.7 cm point spacing, 2.3 m structure height, 89 m walked). Two lessons worth keeping:

- **Do not pick the ground plane by inlier count.** A corridor's walls outvote its carpet, and the first fit returned camera heights spanning 0.33–6.22 units. Pick the plane that keeps camera height *constant* — that is what "floor" physically means for a walk. The chosen plane is then handed to `clean_cloud.py` rather than re-fitted.
- **The anchor dies with the trajectory.** It needs trustworthy poses, so the script refuses above `traj_length_over_extent` 6. On the courthouse run (ratio 25) there is no camera anchor at all and scale would have to come from a known object.

**A ground truth now exists (2026-08-06).** GrandTour's CPT7 reference makes the Umeyama scale in `recon/eval_ate.py` a *measured* metres-per-unit rather than an assumed one: **5.4436 m/unit** on the EIG-1 25 s segment. `eval_ate.py` prints it against `calibrate_scale.py`'s camera anchor whenever a `scale.json` exists, so the anchor's accuracy — not just its repeatability — is finally checkable. Running that comparison is the next step and has not been done yet.

**Still open:** the accuracy of the *camera-height* anchor, which is what the expedition will actually use. Eye height was *assumed* at 1.5 m, and the p5–p95 camera-height spread is 37% of the median, so this is good to maybe ±15% — and its repeatability is measured at 14% on identical footage. Nobody has established what scale error locomotion fine-tuning actually tolerates. The expedition anchors (markers, GPS, Pemba's dimensions) plug into the same `--anchor factor` path but remain unexercised.

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
Phase 7, not just a throughput one. Also untested: an external pose prior from Pemba's odometry,
which is the one signal the expedition has that a phone clip does not.

**The mechanism is now visible, on outdoor footage with a ground truth (2026-08-06).** On
GrandTour EIG-1 — rocky alpine trail, high texture — a 25 s segment gives
`traj_length_over_extent` **1.27**, the healthiest number this project has produced, so the
25 s rule above is a property of *low-texture indoor* footage and not a universal ceiling.
What did survive is the stitching cost: 5 windows over 23 m, per-window **scale span 2.1×**, and
an ATE-vs-distance curve that is a **sawtooth with humps at the window boundaries** (~4 / 11 /
17.5 / 22 m against a boundary every 4.6 m). ATE 1.168 m over 23.0 m of path, 5.07%.

That reframes the question. Window count is the lever, and there are two ways to cut it:
`keyframe_interval` (costs keyframe spacing — the courthouse failure) and `window_size` (costs
VRAM only). At 518×294 the fitted cost is `VRAM ≈ 3.15 + 0.128 × window_size` GB, so `ws=256`
is ~36 GB — free on an 80 GB card, impossible on 8 GB. **Open:** whether raising `window_size`
at constant keyframe density actually removes the sawtooth, which is sweep A in
`lab-notebook/2026-W32.md` and is the first test of the paper's compounding-alignment claim
this project has been able to run.

## Sim can't model snow physics

Isaac Sim's PhysX rigid-body contacts have no sinkage or compliance — same scope boundary the legacy MuJoCo track had. The pipeline targets *geometry* (rocks, slopes, steps); GenTe-style force modeling is explicitly future work. The team's stated mitigation is **domain randomization over snow/ice friction and wind gusts** (`EventManager` terms), not soft contacts. Isaac does ship deformable/particle simulation, but it is out of scope for locomotion training at our env counts. (Legacy: the MuJoCo-side question — whether `solref`/`solimp` tuning could fake compliance — was never run.) **Open:** whether friction-range DR alone transfers to real snow, and what friction range even represents ice-glazed rock vs powder.

## Overfitting to reconstruction artifacts

Reconstruction noise becomes "features" the policy exploits. Mitigation is terrain randomization around the reconstruction (Phase 5, see [[pipeline]]). **Open:** how much randomization is enough — measurable as the gap between performance on the raw recon vs. perturbed variants.

## Upstream's memory horizon is unreachable here

`demo.py` defaults to `kv_cache_sliding_window=64` with `keyframe_interval=1` for anything under the 320-view RoPE limit, and that is the config behind every published LingBot-Map demo. **We cannot run it.** Measured Aug 6 on `example/courthouse` (286 frames): kvsw 64 OOMs, 48 OOMs, 24 OOMs at frame 240/286, and only 16 completes. Halving the sequence to 143 frames with `--stride 2` did *not* buy a bigger cache — **kvsw ≈ 24 is a hard ceiling set by cache size, not sequence length**.

> [!note] `lingbot-map-long` costs no extra VRAM — corrected 2026-08-06
> This note used to add "`lingbot-map-long` is bigger still and cannot run 286 frames at all",
> which was an inference from one OOM inside a batch where every variant OOM'd. Both
> checkpoints are **4,632,303,465 bytes, 1342 tensors, 1.158 B parameters, same key set and
> same shapes**, so `-long` costs exactly the same VRAM as the base model.
>
> Same *size*, not the same *file*: SHA256 `ee665103…` vs `832bc82c…`, and **1341 of 1342
> tensors differ numerically** (up to ~45% relative on some attention norms). Same architecture,
> separately trained. Both hashes match `robbyant/lingbot-map` upstream, which lists both at that
> identical byte count — verified 2026-08-06, so the equal size is the release's property, not a
> bad download. Upstream also ships `lingbot-map-stage1.pt` (4,762,944,015 bytes), which we do
> not have.

The consequence is not cosmetic: at kvsw 16 the courthouse poses scribble (`traj_length_over_extent` **24.9** against loop's 3.36) while per-frame geometry stays excellent.

> [!warning] The cache hypothesis was wrong — tested and killed 2026-08-06
> This note used to conclude "upstream's own courthouse result is reproducible only on a bigger
> card." That was inference, not measurement, and an A100 80 GB falsified it. Sweeping
> `kv_cache_sliding_window` over **16 → 128** on courthouse moved the ratio from 24.84 to 25.59 —
> **8× the cache, 3% change, in the wrong direction.** The discriminator pair (`kvsw 16 / kfi 2`
> = 25.19 against `kvsw 32 / kfi 1` = 25.17) came back null too, so it is neither the number of
> cached views nor their temporal span. Controls confirm the comparison: kvsw 16 on the A100 gives
> 24.84 against 24.88 locally, and loop gives 3.37 against 3.36.
>
> **The real cause is frame spacing.** Consecutive frames in `example/courthouse` are ~47 px apart
> (phase correlation at 518 px width); loop's are ~2 px. Upstream's own demo pipeline targets
> **25 px between keyframes** with every intermediate frame densely tracked, so courthouse's
> shipped frames are already ~2× past their *keyframe* spacing with nothing in between.
> `example/courthouse` is a decimated teaser, not the sequence behind their published video —
> see the two-pipelines section in [[pipeline]].

The 8 GB ceiling on `kv_cache_sliding_window` is still real and still bounds what runs locally. What is *not* established is that lifting it buys quality — the one direct test of that failed, so the same "bigger cache, longer horizon" reasoning applied to the ~25 s clip limit below is now also unsupported and needs its own run.

**Open:** where exactly the cliff is in cache-views-per-scene, and whether it is the *number* of cached views or their *temporal span* that matters — the `keyframe_interval 2` result hints at the former, which would make VRAM the only lever and settle the cloud-GPU question for Phase 7.

**Answered 2026-08-06** by `colab/lingbot_map_colab.ipynb` on an A100 80 GB — see the warning above. Neither quantity is the lever. What remains genuinely open is whether a bigger cache helps on footage that is *inside* the model's sampling regime, which courthouse never was; the room-map clip at 49 s and 132 s is the honest test and has not been run on a big card.

## Below Isaac's minimum spec on both axes

This box is 8 GB VRAM / 16 GB RAM against Isaac Sim's official 16/32 minimum — and LingBot-Map's KV cache eats the same 8 GB ([[setup]] has the per-workload tactics). **Partly answered (2026-08-07):** the headless SimulationApp opens fine (~8 s), and the full-collision G1 task trains at 64 envs / ~690 steps/s locally — the below-minimum warning applies to the GUI/renderer path, not headless physics at small env counts. **Still open:** the actual local `num_envs`/VRAM ceiling for the G1 velocity task (only 8 and 64 tested); whether a full 1500-iteration flat-plane run is viable locally or needs cloud from the start; and the cloud-GPU budget/provider choice for Phase 4a/5 runs and the expedition window.

## Closed loop on the expedition

Redeploying fine-tuned policies onto Pemba mid-expedition may be too risky, and the G1 policy stack is partly closed (factory controller ≠ our policy). Current posture is sim-validated recommendations only ([[decisions]]). **Open:** what evidence bar would make expedition leads green-light a real redeployment.
