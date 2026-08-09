---
title: Pipeline architecture
updated: 2026-08-08
status: current
---

# Pipeline architecture

The real→sim→real loop, stage by stage. Mission context and schedule live in [[overview]]; hardware constraints in [[setup]]; unresolved risks in [[open-questions]].

## The loop on one page

| # | Stage | In → Out | Tool |
| --- | --- | --- | --- |
| 1 | Capture | Pemba walks the trek → RGB video @ 20–30 fps | Unitree G1 onboard camera (rules: [[capture-protocol]]) |
| 2 | Reconstruction | Video → camera poses + dense point cloud, streaming frame-by-frame | LingBot-Map |
| 3 | Cleanup | Point cloud → filtered, downsampled, scale-calibrated, ground-aligned cloud | Open3D |
| 4 | Terrain build | Clean cloud → mesh (Open3D) → OBJ → USD collision asset (PLY is not accepted — convert first) | Open3D → Isaac Lab `MeshConverter` (`triangleMesh`) |
| 5 | Scene assembly | Terrain USD + `G1_CFG` (full collision) → simulated Pemba on reconstructed terrain | Isaac Lab `TerrainImporterCfg(terrain_type="usd")` + `isaaclab_assets` |
| 6 | RL training | Baseline velocity policy → policy trained on *that* terrain, with terrain + domain randomization (friction/ice, mass, pushes as wind-gust proxy) | Isaac Lab + RSL-RL PPO, `EventManager` |
| 7 | Validation | Policy export → sim2sim check in vanilla MuJoCo at 50 Hz — the legacy track's permanent job | MuJoCo (`sims/mujoco/`) |
| 8 | Deployment | Validated policy / analytics → back to Pemba (or sim-validated recommendations to the team) | DimensionalOS |

If LingBot-Map output proves unusable for a scene, stage 4 falls back to **procedural mountain
terrain** (`TerrainGeneratorCfg` + `Hf*TerrainCfg` sub-terrains) — the loop from stage 5 on is
unchanged. LingBot-Map is used only when it works; it is no longer on the critical path.

## The three core technologies

### Isaac Sim + Isaac Lab (primary since 2026-08-07)

NVIDIA's GPU-native robotics simulator (PhysX, USD scene format) and its RL framework. This is the Robot Everest team's actual stack, which is why we pivoted to it (see [[decisions]], 2026-08-07). Pinned: **Isaac Sim 5.1.0 + Isaac Lab 2.3.x, Python 3.11, native Windows 11 via pip** — not Isaac Lab 3.0 beta (Ubuntu-only). Install and smoke ladder: `sims/isaac/README.md`.

What we stand on rather than build: `isaaclab_assets` ships **`G1_CFG`** (full-body collision — our Phase 2 requirement, built-in here) and the `Isaac-Velocity-Flat-G1-v0` / `Isaac-Velocity-Rough-G1-v0` tasks; `unitreerobotics/unitree_rl_lab` is Unitree's own Isaac Lab repo (RSL-RL, sim2real-tested G1 configs). Terrain comes in as USD (`TerrainImporterCfg`: `plane | generator | usd`), meshes convert via `MeshConverterCfg` (OBJ/STL/FBX — **not PLY**), and domain randomization runs through `EventManager` terms (`randomize_rigid_body_material`, `randomize_rigid_body_mass`, `push_by_setting_velocity`).

Things to master: USD composition, ArticulationCfg, the manager-based env (observations/rewards/events as config), TerrainGenerator curricula, RSL-RL's train/play loop, headless workflows.

Links: [IsaacLab](https://github.com/isaac-sim/IsaacLab) · [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/) · [Isaac Sim docs](https://docs.isaacsim.omniverse.nvidia.com/) · [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)

#### MuJoCo (+ MJX + Playground) — legacy + sim2sim validation

DeepMind's physics engine carried Phases 1–2 (its contact dynamics are the reason stage 7 validates in it): full-body-collision G1, self-trained joystick policy, the whole Aug 6 video→terrain→standing-G1 chain. All of it stays runnable in `sims/mujoco/` — see `sims/mujoco/README.md` for gates and runtime. Reward terms and DR of the trained policy: [[locomotion-policy]].

### LingBot-Map

Open-source (Apache 2.0) **streaming** 3D-reconstruction foundation model from Robbyant (Ant Group's embodied-AI arm), released April 2026. Ordinary RGB video in → camera poses + dense point clouds out, frame-by-frame at ~20 FPS @ 518×378, stable past 10,000 frames. No LiDAR, no depth camera, no offline COLMAP run. The trek is one long continuous walk — exactly the long-streaming regime it leads benchmarks on (Oxford Spires, Tanks & Temples, ETH3D). The `lingbot-map-long` variant plus `keyframe_interval` / windowed mode exist specifically for long outdoor runs.

Things to master: streaming vs. offline reconstruction, the Geometric Context Transformer (anchor context + pose-reference window + trajectory memory), monocular scale ambiguity, confidence filtering, the viser viewer. Known limits — low-texture surfaces (snow!), motion blur, exposure swings, the ~320-view RoPE limit — are tracked in [[open-questions]].

Our wrapper lives in `recon/` (see [[setup]] for the install and this box's measured ceilings). Streaming works as advertised — bounded KV cache, flat VRAM over any clip length — which is what makes the Phase 6 DimOS module and Phase 7 live pipeline possible at all. What actually bounds a reconstruction here is **drift over clip length**: roughly 25 s of footage per globally consistent scene on 8 GB, in *either* mode, because the KV cache that bounds memory also bounds how far back the model remembers ([[open-questions]]). Upstream's `demo.py` also exports nothing, so `recon/reconstruct.py` writes the cloud, the trajectory and a run record itself.

Links: paper arXiv 2604.14141 · official Robbyant repo (`batch_demo.py`, viser demo) · MarkTechPost hands-on tutorial (July 2026)

#### Upstream ships two inference pipelines, and the README shows the weaker one

Established 2026-08-06 by reading the paper against the repo. `demo.py` — the README's one-liner —
is **not** what produced their published demo videos. Those came from `demo_render/batch_demo.py`,
driven by `demo_render/process_videos.sh`.

| | **Direct** (`demo.py`) | **VO** (`batch_demo.py`) |
| --- | --- | --- |
| paper | §4.5, the config behind every benchmark number | §4.4, *"for the large-scale demo videos … we use VO mode"* |
| `--mode` | `streaming` | `windowed` |
| pose-reference window | k = 64 | k = 64, `window_size` 64 keyframes |
| keyframes | fixed interval, m = 1 | **adaptive optical flow**: 25.0 px, forced every 100 |
| their input | the `example/` folders | the source video, `TARGET_FRAMES=4000`, `IMAGE_STRIDE=1` |

The keyframe mechanism (§4.4) predicts pose and depth per incoming frame, measures optical flow
against the most recent keyframe, and promotes the frame only once that flow clears a threshold —
this is what bounds cache growth on long sequences. **`demo.py` exposes it nowhere**, and
`gct_stream.py` (Direct) does not implement it at all; it lives only in `gct_stream_window.py`.
Every windowed run logged in [[experiments]] before Aug 6 therefore used fixed intervals.

`recon/reconstruct.py` now takes `--flow_threshold` and `--max_non_keyframe_gap` (windowed only,
enforced) plus `--conf_threshold` for upstream's absolute confidence cut instead of our percentile.
The run record gains **`keyframe_frac`**, read from the model's `is_keyframe` mask — the diagnostic
that matters: at 100% every frame cleared the threshold, meaning the footage is sampled more
sparsely than the keyframe policy targets and nothing is densely tracked in between. That is the
state `example/courthouse` is in, and it is a property of the frames, not of any config
([[open-questions]]).

VO's cost is stated plainly in the paper: it fuses windows by Sim(3) alignment over their overlap
and *"incurs extra alignment error that compounds with the number of windows"*. Direct is more
accurate whenever the sequence fits inside ~3,000 frames.

### DimensionalOS (DimOS)

Open-source, Python-first "agentic operating system" for robots — the framework the expedition actually runs. Modules (perception, SLAM, planners, motor control) communicate over typed pub/sub channels (`In[T]`/`Out[T]`); LLM agents are first-class modules; no ROS. Unitree support ships in the box: `uv pip install 'dimos[base,unitree]'`.

Killer feature for this project: **replay datasets** — DimOS replays recorded robot sessions (camera + lidar + state) with no hardware, so the whole pipeline is developed against real robot data streams from a desk (`dimos --replay --replay-dir ...`). Phase 6 turns reconstruction into a live DimOS capability: Blueprint wiring replayed camera → LingBot-Map module → terrain-export module → saved terrain asset (**USD** for Isaac, the primary target; MJCF export kept for the sim2sim validator). DimOS's MuJoCo backend still runs retrained policies inside the same framework for validation.

Links: [dimos](https://github.com/dimensionalOS/dimos) (README + AGENTS.md first) · deepwiki.com/dimensionalOS/dimos

## Terrain conversion

**Isaac real-DEM path (built + gated 2026-08-08):** for *long* terrain, skip reconstruction
entirely — a survey-grade national DEM has no scale ambiguity and no drift ceiling. First
asset: the **Eiger Trail** (swissALTI3D 0.5 m, 5.4 km, 713 m descent, same terrain family as
GrandTour EIG-1), straightened into an arc-length × cross-track strip → OBJ → USD
(`TriangleMeshPropertiesCfg` exact collision) → `Dome-G1FullCollision-EigerTrail-v0` with
custom on-trail env origins. Commands and design: `sims/isaac/terrain/README.md`; rationale:
the 2026-08-08 terrain entry in [[decisions]]. Gate: `sims/isaac/scripts/check_trail.py`.
Second asset (built + gated 2026-08-08): the **Everest summit patch**
(NASA HMA 8 m DEM tile-677, 2 km axis-aligned around the summit pyramid, 1750 m relief,
slope-filtered spawn origins) → `Dome-G1FullCollision-Everest-v0`
(`sims/isaac/tasks/dome_g1/everest_env_cfg.py`); gate: `sims/isaac/scripts/check_everest.py`;
rationale: the Everest-DEM entry in [[decisions]].

**Isaac recon path (per-segment, Phase 4b — to build):** clean metric cloud → Open3D surface
reconstruction (Poisson / ball-pivoting) → decimate → **OBJ** → USD via `MeshConverterCfg`
(exact triangle-mesh collision). Bounded by the ~25 s/~25 m consistency horizon
([[open-questions]]). Fallback when a reconstruction isn't usable: procedural mountain
terrain from `TerrainGeneratorCfg` sub-terrains. Converters land in `sims/isaac/terrain/`.

**MuJoCo path (legacy, built and gated):** grid XY at 5–10 cm cells, robust max-z (or
`--surface ground` indoors), fill holes → `hfield` (`sims/mujoco/terrain/cloud_to_hfield.py`).
The mesh variant was never built.

Scale calibration is mandatory before either path (monocular reconstruction has arbitrary scale): at home, film two markers a measured distance apart; on the expedition, use Pemba's known dimensions or GPS track length. See [[glossary]] for terms.

### The legacy chain, as it actually ran (Aug 6, MuJoCo)

```bash
recon/fetch_grandtour.py --mission eig-1 --out <run>   # benchmark footage + CPT7 GT (optional)
recon/measure_flow.py   --frames <dir>             # preflight gate, before any VRAM is spent
recon/reconstruct.py    --frames <dir> --out <run> --model_path <ckpt> [--mask_sky]
recon/eval_ate.py       --run <run> --gt <run>/gt_tum.txt  # GT scoring (GrandTour input only)
recon/calibrate_scale.py  <run>                    # -> scale.json (m/unit + ground plane)
recon/clean_cloud.py      <run> --scale auto       # -> cloud_clean.ply, in metres
sims/mujoco/terrain/cloud_to_hfield.py  <run> --name <asset> --crop [--surface ground] [--smooth 2]
sims/mujoco/terrain/drop_test.py        --asset <asset>          # terrain gate
sims/mujoco/scripts/settle_g1_recon.py  --asset <asset> --render # robot gate
```

### Benchmark footage with a ground truth: GrandTour

Added 2026-08-06. Every reconstruction before this was scored by `traj_length_over_extent`, a
self-consistency proxy. The **GrandTour** dataset (ETH Zurich RSL, arXiv 2602.18164 — ANYmal-D
walked across Switzerland) supplies outdoor legged-robot footage *with* a survey-grade CPT7
GNSS/INS reference, so `recon/eval_ate.py` can report ATE, RPE and — the part that matters most
here — a **metres-per-unit scale measured against a real reference** instead of an assumed eye
height ([[open-questions]]).

`recon/fetch_grandtour.py` pulls a mission from HuggingFace (zarr + JPEG tars, no registration),
rectifies the released camera model to an explicit pinhole straight into the output raster, and
composes the ground truth through the camera's 0.417 m lever arm. Two things it encodes that are
easy to get wrong: the released `hdr_front` stream is **10 Hz, not the paper's 30 fps**, and
`zed2i_left_images` is the right stream for this model — 14.91 Hz, `radtan` with negligible
distortion, and 16:9 so it lands on 518×294 rather than 518×350.

This is *benchmark* footage, not expedition footage: it validates and tunes the toolchain, and
its missions (EIG-1 alpine rock, SNOW-2 low texture) stand in for terrain the Everest route has.
It does not replace [[capture-protocol]] or Pemba's own camera.

Three things this shipped that the plan did not anticipate:

- **`--surface ground` vs `top`.** Robust max-z is right for an outdoor trail, where the upper surface *is* what you walk on. Indoors it makes every partition a 2.3 m spire and the hfield a canyon, so `ground` takes a near-minimum per cell and rejects cells more than `--ground-tol` above the floor. Furniture and walls are then absent — this terrain is a walking surface, not an obstacle course.
- **Unobserved cells are filled flat, not interpolated.** A walkthrough observes a corridor-shaped sliver of its bounding box (9% here). Nearest-fill smears walls across regions nobody looked at. The observation mask ships beside the asset.
- **Smoothing is not cosmetic.** Monocular depth noise puts ~5 cm of roughness on flat carpet, which topples a keyframe-posed G1. `--smooth 2` (10 cm sigma) is what makes it stand; the sigma is recorded in the asset's JSON because it changes the terrain a policy sees.

The Isaac chain replaces the last three lines from stage 4b on; until it exists, this remains
the only proven video→standing-robot path in the repo.

### Contact parameters (measured — MuJoCo legacy)

Terrain geom: `solref="0.008 1" solimp="0.9 0.95 0.001"`, `friction="1 0.005 0.0001"`, `condim="3"`, timestep 0.002 s.

`solref` 0.008 is 4× the timestep and mixes with Menagerie's foot geoms to 0.014. Menagerie's ankle geoms carry their own `solref` (0.02) that wins on foot contacts, so a settled G1 sinks **0.3 mm on flat ground and ~5 mm where a foot corner loads a sloped 5 cm cell**. That is foot compliance, not terrain error — the gate in `sims/mujoco/scripts/settle_g1_recon.py` allows 10 mm and counts *terrain* contacts only.

## Training recipe (Phase 5, Isaac Lab)

Curriculum: flat → gentle terrain → full terrain (Isaac Lab's `TerrainGeneratorCfg` has
curriculum + `difficulty_range` built in). **Terrain randomization**: sub-terrain variants
(noise, tilt, bump scale ±20%) so the policy generalizes instead of overfitting to
reconstruction artifacts. **Domain randomization** via `EventManager` terms: material
friction/restitution (snow/ice), base mass, motor strength, `push_by_setting_velocity`
(pushes + wind-gust proxy), sensor latency (real G1 command latency ≈ 18–30 ms — model it).
**Evaluation harness**: success rate over N rollouts, mean distance before fall,
velocity-tracking error — terrain vs. flat, before vs. after. Numbers, not vibes; every run
is a row in [[experiments]]. Training compute: cloud GPU (see [[setup]] — local is smoke-test
only).

## Reference literature

- **DISCOVERSE** (air-discoverse.github.io) — Gaussian-Splatting + MuJoCo real2sim2real; the closest existing system. We differ: outdoor, streaming, locomotion.
- **Splatting Physical Scenes** (arXiv 2506.04120) — end-to-end real-to-sim with differentiable MJX. Advanced; skim.
- **GenTe** (arXiv 2504.09997) — realistic terrain generation for locomotion training; honest about what rigid-contact sims can't model (snow sinkage).
- **humanoid-gym** (roboterax) — the standard sim2sim recipe: train fast, validate in MuJoCo, deploy.
- **G1-Playground** (AlexandreBrown) — minimal G1 sim2sim/sim2real repo sharing the real robot's DDS code path.
