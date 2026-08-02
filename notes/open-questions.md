---
title: Open questions & hard problems
updated: 2026-08-01
status: current
---

# Open questions & hard problems

The known-hard parts of the pipeline and what's genuinely unresolved. When one gets answered, move the answer to its owning note ([[pipeline]], [[setup]], [[capture-protocol]]) and delete it here.

## Snow / low-texture reconstruction failure

Monocular models need texture; snow is texture-poor and overexposed, so LingBot-Map is expected to degrade on exactly the terrain that makes this Everest. Mitigation plan: test on white/low-texture proxies early (a white wall ≈ snow proxy, Phase 3), lock exposure, and expect best results on the rocky sections — most of the route below C2 is rock. **Open:** where exactly the failure threshold is, and whether any capture rule ([[capture-protocol]]) moves it.

## Scale calibration

Monocular reconstruction has no absolute scale, and terrain that's 10% too large changes step heights the policy trains on. Candidate calibrations: measured markers (home), Pemba's known dimensions, GPS track length (expedition). **Open:** which method is accurate enough, and what the acceptable scale error even is for locomotion fine-tuning.

## Drift on very long sequences

Even streaming models accumulate pose error over a trek-length walk. Mitigation: chunk the trek (≤10-min pieces), `lingbot-map-long`, keyframe interval, windowed mode. **Open:** how much drift survives chunking, and whether chunks can be stitched into one consistent terrain or must stay separate scenes.

## Sim can't model snow physics

MuJoCo contacts are rigid — no sinkage, no compliance. This is a scope boundary, not a bug: the pipeline targets *geometry* (rocks, slopes, steps); GenTe-style force modeling is explicitly future work. **Open:** whether `solref`/`solimp` tuning can fake enough compliance to be worth doing.

## Overfitting to reconstruction artifacts

Reconstruction noise becomes "features" the policy exploits. Mitigation is terrain randomization around the reconstruction (Phase 5, see [[pipeline]]). **Open:** how much randomization is enough — measurable as the gap between performance on the raw recon vs. perturbed variants.

## 8 GB VRAM ceiling

LingBot-Map's KV cache and MJX's env count both eat VRAM ([[setup]] has the per-workload tactics). **Open:** whether Phase 5 per-iteration debugging genuinely fits locally, and the cloud-GPU budget/choice for sweeps and the expedition window.

## Closed loop on the expedition

Redeploying fine-tuned policies onto Pemba mid-expedition may be too risky, and the G1 policy stack is partly closed (factory controller ≠ our policy). Current posture is sim-validated recommendations only ([[decisions]]). **Open:** what evidence bar would make expedition leads green-light a real redeployment.
