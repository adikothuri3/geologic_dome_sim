---
title: Glossary
updated: 2026-08-01
status: current
---

# Glossary

Project-specific terms, alphabetical. Where a term is central to a stage, the owning note is linked.

- **50 Hz control loop** — the standard control frequency for G1 locomotion policies; sim2sim validation runs vanilla MuJoCo at 50 Hz to match deployment.
- **Blueprint (DimOS)** — a wiring of DimOS Modules into a pipeline; Phase 6's blueprint is replayed camera → LingBot-Map module → terrain-export module → MJCF asset ([[pipeline]]).
- **confidence filtering** — dropping low-confidence points from LingBot-Map's output before terrain building; first defense against reconstruction noise.
- **DDS** — the middleware the real G1 uses for command/state transport; G1-Playground shares this code path, which is why it matters for eventual deployment.
- **domain randomization** — randomizing physics parameters during training (friction, base mass, motor strength, pushes, sensor latency) so a policy survives the sim-to-real gap.
- **Geometric Context Transformer** — LingBot-Map's core mechanism: anchor context + pose-reference window + trajectory memory, which is what makes streaming reconstruction stable over long walks.
- **hfield (heightfield)** — MuJoCo terrain asset: a 2D grid of heights. Fast and robust for walking terrain; cannot represent overhangs (that's what meshes are for). Primary terrain path ([[decisions]]).
- **keyframe_interval** — LingBot-Map setting controlling how often frames become keyframes; the main lever on KV-cache growth (VRAM) and the ~320-view RoPE limit ([[setup]]).
- **KV cache** — the growing transformer memory during streaming inference; LingBot-Map's main VRAM consumer on long clips.
- **lingbot-map-long** — the LingBot-Map variant built for long outdoor sequences; used with keyframing and windowed mode for trek-length footage.
- **Menagerie** — google-deepmind/mujoco_menagerie, curated MJCF robot models; source of the official `unitree_g1` model.
- **MJCF** — MuJoCo's XML scene/robot description format. Terrain assets, the G1, and every simulated scene are MJCF.
- **MJX** — MuJoCo reimplemented in JAX: thousands of parallel environments on one GPU, which is what makes RL fine-tuning feasible on a schedule.
- **monocular scale ambiguity** — a single-camera reconstruction has no absolute scale; everything must be rescaled against a known length (markers, robot dimensions, GPS track) before it becomes terrain ([[pipeline]]).
- **MuJoCo Playground** — ready-made MJX RL environments, including the G1 joystick-locomotion env this project fine-tunes.
- **Pemba** — the expedition's Unitree G1 humanoid (also the name convention for the robot throughout these notes).
- **replay dataset (DimOS)** — a recorded robot session (camera + lidar + state) that DimOS replays with no hardware attached; how the whole pipeline is developed from a desk.
- **RoPE limit (~320 views)** — LingBot-Map's positional-encoding ceiling on views held in context; managed via `keyframe_interval` and windowed mode.
- **sim2sim** — validating a policy trained in MJX by running it in plain MuJoCo at 50 Hz; the standard credibility check before any hardware conversation.
- **solref / solimp** — MuJoCo contact-solver parameters governing contact stiffness/damping; they matter enormously for legged locomotion realism and get tuned in Phase 4.
- **streaming reconstruction** — reconstruction that updates frame-by-frame as video arrives (LingBot-Map), vs. offline batch reconstruction (COLMAP). The project's core bet ([[decisions]]).
- **terrain randomization** — generating perturbed variants of the reconstructed terrain (noise, tilt, bump scale ±20%) so the policy generalizes around reconstruction artifacts instead of overfitting to them.
- **viser** — the web-based 3D viewer used to inspect LingBot-Map point clouds and camera trajectories.
