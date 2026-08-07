---
title: Glossary
updated: 2026-08-07
status: current
---

# Glossary

Project-specific terms, alphabetical. Where a term is central to a stage, the owning note is linked. Terms tagged *(legacy sim)* belong to the MuJoCo track (`sims/mujoco/`), kept for Phases 1–2 history and sim2sim validation after the 2026-08-07 Isaac pivot ([[decisions]]).

- **50 Hz control loop** — the standard control frequency for G1 locomotion policies; sim2sim validation runs vanilla MuJoCo at 50 Hz to match deployment.
- **Blueprint (DimOS)** — a wiring of DimOS Modules into a pipeline; Phase 6's blueprint is replayed camera → LingBot-Map module → terrain-export module → terrain asset (USD primary, MJCF for the validator) ([[pipeline]]).
- **confidence filtering** — dropping low-confidence points from LingBot-Map's output before terrain building; first defense against reconstruction noise.
- **DDS** — the middleware the real G1 uses for command/state transport; shared by the sim2real reference repos, which is why it matters for eventual deployment.
- **domain randomization** — randomizing physics parameters during training (friction, base mass, motor strength, pushes, sensor latency) so a policy survives the sim-to-real gap. In Isaac Lab this runs through the **EventManager**.
- **EventManager / EventTermCfg** — Isaac Lab's mechanism for scheduled and randomized events: `randomize_rigid_body_material` (friction/restitution — snow/ice), `randomize_rigid_body_mass`, `push_by_setting_velocity` (pushes, wind-gust proxy), applied at `startup`/`reset`/`interval`.
- **G1_CFG / G1_MINIMAL_CFG** — Isaac Lab's built-in Unitree G1 articulation configs (`isaaclab_assets`); `G1_CFG` is full-body collision — the thing the legacy track had to generate by hand.
- **Geometric Context Transformer** — LingBot-Map's core mechanism: anchor context + pose-reference window + trajectory memory, which is what makes streaming reconstruction stable over long walks.
- **headless mode** — running Isaac Sim without the GUI/RTX viewport (`--headless`); mandatory on this box, where the renderer alone eats ~7 GB of the 8 GB card ([[setup]]).
- **hfield (heightfield)** *(legacy sim)* — MuJoCo terrain asset: a 2D grid of heights; the legacy track's primary terrain path. Isaac's analogue is `Hf*TerrainCfg` sub-terrains or an imported USD mesh.
- **Isaac Lab** — NVIDIA's RL framework on top of Isaac Sim (manager-based envs, TerrainImporter/Generator, EventManager, RSL-RL runners). Pinned 2.3.x — the last Windows-supported line ([[setup]]).
- **Isaac Sim** — NVIDIA's GPU-native robotics simulator (PhysX physics, USD scenes, RTX rendering). The primary simulator since 2026-08-07; pinned 5.1.0, pip-installed, native Windows.
- **keyframe_interval** — LingBot-Map setting controlling how often frames become keyframes; the main lever on KV-cache growth (VRAM) and the ~320-view RoPE limit ([[setup]]).
- **KV cache** — the growing transformer memory during streaming inference; LingBot-Map's main VRAM consumer on long clips.
- **lingbot-map-long** — the LingBot-Map variant built for long outdoor sequences; used with keyframing and windowed mode for trek-length footage.
- **Menagerie** *(legacy sim)* — google-deepmind/mujoco_menagerie, curated MJCF robot models; geometry source for the legacy track's G1.
- **MeshConverter (MeshConverterCfg)** — Isaac Lab's mesh → USD converter; accepts **OBJ/STL/FBX, not PLY** (export OBJ from Open3D first); `collision_approximation="triangleMesh"` for static terrain.
- **MJCF** *(legacy sim)* — MuJoCo's XML scene/robot description format; everything under `sims/mujoco/xmls/`.
- **MJX** *(legacy sim)* — MuJoCo reimplemented in JAX: thousands of parallel environments on one GPU. What trained the Phase 2 policy.
- **monocular scale ambiguity** — a single-camera reconstruction has no absolute scale; everything must be rescaled against a known length (markers, robot dimensions, GPS track) before it becomes terrain ([[pipeline]]).
- **MuJoCo Playground** *(legacy sim)* — ready-made MJX RL environments, including the G1 joystick-locomotion env Phase 2 fine-tuned.
- **Pemba** — the expedition's Unitree G1 humanoid (also the name convention for the robot throughout these notes).
- **PhysX** — Isaac Sim's physics engine. Rigid-body like MuJoCo — no snow sinkage; the team's answer is domain randomization over friction, not soft contacts ([[open-questions]]).
- **replay dataset (DimOS)** — a recorded robot session (camera + lidar + state) that DimOS replays with no hardware attached; how the whole pipeline is developed from a desk.
- **RoPE limit (~320 views)** — LingBot-Map's positional-encoding ceiling on views held in context; managed via `keyframe_interval` and windowed mode.
- **RSL-RL** — the PPO training library Isaac Lab's velocity tasks and `unitree_rl_lab` use; replaces brax in the primary track.
- **sim2sim** — validating a policy trained in one simulator by running it in another; here: Isaac Lab-trained policy replayed in vanilla MuJoCo at 50 Hz (`sims/mujoco/`), the standard credibility check before any hardware conversation.
- **solref / solimp** *(legacy sim)* — MuJoCo contact-solver parameters governing contact stiffness/damping; tuned and measured during the legacy Phase 4 chain ([[pipeline]]).
- **streaming reconstruction** — reconstruction that updates frame-by-frame as video arrives (LingBot-Map), vs. offline batch reconstruction (COLMAP). The project's core bet ([[decisions]]).
- **TerrainImporter / TerrainGenerator** — Isaac Lab terrain machinery: `TerrainImporterCfg.terrain_type ∈ plane | generator | usd`; `TerrainGeneratorCfg` composes procedural sub-terrains (`HfRandomUniformTerrainCfg`, `HfPyramidSlopedTerrainCfg`, …) with curriculum — the mountain-terrain fallback when LingBot-Map output isn't usable.
- **terrain randomization** — generating perturbed variants of the terrain (noise, tilt, bump scale ±20%) so the policy generalizes around reconstruction artifacts instead of overfitting to them.
- **unitree_rl_lab** — Unitree's official Isaac Lab repo (`Unitree-G1-29dof-Velocity`, RSL-RL); the reference for sim2real-tested G1 configs in the primary track.
- **USD (Universal Scene Description)** — Isaac Sim's scene format; terrain, robots and scenes are USD prims. Terrain meshes must be converted to USD before import.
- **viser** — the web-based 3D viewer used to inspect LingBot-Map point clouds and camera trajectories.
