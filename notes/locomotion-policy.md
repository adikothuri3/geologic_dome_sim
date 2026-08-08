---
title: The G1 locomotion policy — rewards, observations, randomization
updated: 2026-08-08
status: current
---

# The G1 locomotion policy

> [!important] This note covers both sims — MuJoCo first, Isaac last
> Everything up to *Scaling on this box* documents the **Phase-2 MuJoCo/MJX joystick
> policy**; the sim was pivoted to Isaac Lab on 2026-08-07 ([[decisions]]). It is kept
> because it is the record of what Phases 1–2 trained, the vocabulary the Isaac section
> is written against, and the reference for the sim2sim validator in `sims/mujoco/`.
> The **Isaac task** — the one being trained now — is the last section, and it is written
> as a diff against the MuJoCo one rather than from scratch.

What Phase 2 actually trained, and what every part of it means. Everything here is read from
the installed `mujoco_playground` G1 joystick environment, not from memory — re-derive with
`sims/mujoco/scripts/check_phase2.py` and the sources under `~/src/playground/.../locomotion/g1/`.

The reward function is **upstream's, unmodified**. That was the [[decisions]] call from
2026-08-01: the novelty of this project is the terrain loop, not the locomotion baseline.

## The task

Not "walk" — **follow a velocity command**. Each episode samples a command and the policy must
realise it:

| Channel | Range |
| --- | --- |
| forward velocity `vx` | −1.0 … 1.0 m/s |
| lateral velocity `vy` | −0.5 … 0.5 m/s |
| yaw rate `ωz` | −1.0 … 1.0 rad/s |

Channels are **zeroed at random** (probability of *keeping* each is `[0.9, 0.25, 0.5]`), so a
zero command — stand still — is part of the training distribution. One policy therefore covers
walking, strafing, turning and standing. Episodes are 1000 steps at `ctrl_dt = 0.02`: 20 s of
simulated time at the project's 50 Hz convention (see [[pipeline]]).

## Observations

Two vectors. The policy network sees only `state`; the critic additionally sees
`privileged_state`, which contains quantities a real robot could not measure.

**`state` (103)** — everything here is deliberately corrupted with noise (see randomization):

| Component | Size |
| --- | --- |
| pelvis linear velocity (local frame) | 3 |
| gyro | 3 |
| gravity direction in the pelvis IMU frame | 3 |
| the current command | 3 |
| joint angles, **relative to the default pose** | 29 |
| joint velocities | 29 |
| previous action | 29 |
| gait phase, as `cos` and `sin` | 4 |

Two details that matter: joint angles are given as *deviation from the nominal pose*, not
absolute, so the network learns corrections rather than absolute posture; and the gait phase is
supplied as a `(cos, sin)` pair per leg so it is continuous across the wrap from 2π to 0.

**`privileged_state` (216)** — `state` plus the *clean* (noise-free) gyro, accelerometer,
gravity, linear velocity and global angular velocity, plus true joint angles/velocities, root
height, actuator forces (29), foot contact flags (2), foot velocities and feet-air-time. This
is asymmetric actor-critic: the critic gets ground truth for a better value estimate, while the
actor is restricted to what the hardware can actually sense.

## Reward terms

Combined as `reward = Σ(scale × term) × dt`, with `dt = 0.02`, summed over the episode. Twelve
terms are active; ten are wired up but scaled to zero.

**Active — the two that pay, and the one that dominates:**

| Term | Scale | What it does |
| --- | --- | --- |
| `tracking_lin_vel` | **+1.0** | `exp(−err²/0.25)` on commanded vs actual velocity. A Gaussian bump in [0,1], so being *close* pays nearly as well as being exact — not a linear penalty |
| `tracking_ang_vel` | **+0.75** | the same shape, for yaw rate |
| `feet_air_time` | **+2.0** | rewards time feet spend in swing. **This is what buys stepping instead of shuffling** — the largest positive scale |
| `feet_phase` | **+1.0** | rewards matching the reference gait phase |
| `termination` | **−100.0** | falling. Two orders of magnitude above everything else |
| `orientation` | −2.0 | torso tilt away from upright |
| `stand_still` | −1.0 | deviation from the default pose **when the command is ~zero** — i.e. "when told to stand, hold still" |
| `dof_pos_limits` | −1.0 | joints pressed against their limits |
| `feet_slip` | −0.25 | feet sliding while in contact |
| `joint_deviation_hip` | −0.25 | hips drifting from nominal |
| `ang_vel_xy` | −0.15 | roll/pitch rate — damps wobble |
| `joint_deviation_knee`, `pose`, `collision` | −0.1 each | knees/body near nominal; self-collision |
| `contact_force` | −0.01 | contact forces above 500 N |

**Zeroed upstream** — the machinery exists, the scale is `0.0`: `lin_vel_z`, `base_height`,
`torques`, `action_rate`, `energy`, `dof_acc`, `feet_clearance`, `feet_height`, `alive`. Any of
these is a ready-made knob for Phase 5; `energy` and `action_rate` are the usual candidates for
smoother, more transferable gaits.

**Termination** fires when the torso's gravity-z goes negative (tipped past horizontal) or when
feet/shins contact each other. It is a *cost*, not merely an episode end — falling is actively
punished, not just unrewarded.

> [!warning] Leg-crossing termination is dead in the feet-only model
> `_get_termination()` reads `left_foot_right_shin_found` / `right_foot_left_shin_found`, but
> the contact `<pair>`s those sensors need are commented out in Playground's XML — no pair, no
> contact, so the sensors can never fire. Our full-collision model fixes this as a side effect;
> see [[decisions]].

## Domain randomization

Already active in every run, and applied **per environment**: `domain_randomize` is
`@jax.vmap`'d over the rng with `in_axes` mapping each field to axis 0, so all 2048 parallel
envs train on *different physics simultaneously* — not one shared domain.

**Dynamics, resampled per env:**

| Parameter | Range |
| --- | --- |
| floor/foot friction | U(0.4, 1.0) |
| joint friction loss | ×U(0.5, 2.0) |
| armature | ×U(1.0, 1.05) |
| every link mass | ×U(0.9, 1.1) |
| torso mass | +U(−1.0, +1.0) kg |
| initial joint pose | ±0.05 rad |

**Observation noise, resampled every step** — this is randomization too, and it is what stops
the policy trusting sensors it will not have on hardware:

| Signal | Noise |
| --- | --- |
| joint position | ±0.03 |
| joint velocity | ±1.5 |
| gravity | ±0.05 |
| linear velocity | ±0.1 |
| gyro | ±0.2 |

**Disturbances:** a random push of magnitude 0.1–2.0 every 5–10 s.

What is **not** randomized yet, and belongs to Phase 5 per the roadmap: terrain variants (noise,
tilt, bump scale ±20 %), motor strength, and sensor/command latency (the real G1 is ≈18–30 ms).

## What the policy actually outputs

29 joint *position targets*, at 50 Hz. Position actuators (`kp=75` in the MJX model) then chase
them. So "learning to walk" is learning what 29 angles to ask for, fifty times a second, given
what the robot currently feels. The body is fixed; only the network changes.

## Scaling on this box

`num_envs = 2048` (upstream default 8192 OOMs on 8 GB), holding `num_minibatches = 32` and
deriving `batch_size` so brax's `batch_size × num_minibatches == num_envs` relation is preserved
and the gradient maths is identical to upstream — only parallelism shrinks. See [[setup]] for
the VRAM budget and [[experiments]] for what each run actually produced.

## The Isaac task (Phase 4a) — `Dome-G1FullCollision-Flat-DR-v0`

Built 2026-08-07 in `sims/isaac/tasks/dome_g1/flat_env_cfg.py`. Same job as everything above —
follow a velocity command, under randomized physics — on Isaac Lab's manager-based velocity
task with `G1_CFG` (full-body collision **built in**, which Phase 2 had to generate by hand
with `make_full_collision_xml.py`).

**Rewards and observations are upstream Isaac's, unmodified**, exactly as the MuJoCo side used
Playground's. They are *not* the same function as the table above — Isaac's G1 uses
`termination_penalty −200`, `track_lin_vel_xy_yaw_frame_exp +1.0`, `track_ang_vel_z_world_exp
+1.0`, `feet_air_time +0.75`, `feet_slide −0.1`, per-group `joint_deviation_l1` terms and small
torque/acceleration/action-rate penalties. Same shape of objective, different tuning; the
project's [[decisions]] call is to not rebuild either baseline.

> [!warning] Upstream's G1 config disables almost all of Isaac's own randomization
> `G1RoughEnvCfg.__post_init__` sets `push_robot = None`, `add_base_mass = None`,
> `base_com = None`, zeroes the reset velocities, pins the joint-reset scale to `(1.0, 1.0)`,
> and collapses the physics-material ranges to point values (static `0.8→0.8`, dynamic
> `0.6→0.6`). What is left in stock `Isaac-Velocity-Flat-G1-v0` is **observation noise and
> nothing else** — no dynamics randomization at all. Taking "Isaac has DR built in" at face
> value would have produced a policy trained on one fixed robot. Hence `DomeG1DREventCfg`, and
> hence `check_isaac.py --gate c`, which reads the per-environment spread back out of PhysX
> rather than believing the config.

### What is randomized, against the MuJoCo table above

| Playground `domain_randomize` | Isaac `DomeG1DREventCfg` |
| --- | --- |
| floor/foot friction U(0.4, 1.0) | `physics_material` static (0.4, 1.0), dynamic (0.3, 0.9) |
| every link mass ×U(0.9, 1.1) | `link_mass`, scale, `body_names=".*"` |
| torso mass +U(−1, +1) kg | `torso_mass`, add, `torso_link` |
| armature ×U(1.0, 1.05) | `joint_parameters`, armature, scale |
| joint friction ×U(0.5, 2.0) | **no counterpart** — see below |
| initial joint pose ±0.05 rad | `reset_robot_joints`, *offset* not scale |
| push 0.1–2.0 m/s every 5–10 s | `push_robot`, interval 5–10 s, box ±1.4 m/s |
| obs noise (joint pos ±0.03, vel ±1.5, gravity ±0.05, lin vel ±0.1, gyro ±0.2) | identical, except Isaac ships joint pos at ±0.01 — raised to ±0.03 for parity |

Four things worth knowing, all of them things that would otherwise have failed silently:

- **Joint friction has no Isaac counterpart.** The G1 USD ships `0.0` friction on all 37
  joints, and a `scale` operation cannot lift zero, so the term would have sat in the config
  multiplying nothing. Playground scales a nonzero `frictionloss` this asset does not carry.
  Gate C asserts the zero, so a future asset that *does* carry joint friction surfaces as a
  prompt rather than staying dark.
- **`reset_joints_by_offset`, not `reset_joints_by_scale`.** Upstream scales the nominal pose;
  17 of the G1's joints have a default of exactly 0.0, which scaling leaves untouched.
  Playground's ±0.05 rad is absolute, so the offset term is the faithful one.
- **The events are installed *after* the parent `__post_init__` chain**, not declared as a
  configclass field — otherwise `G1RoughEnvCfg` reaches in and nulls `push_robot` and rewrites
  the joint-reset range, disabling the two terms this task most needs.
- **`torso_com` (±5 cm) has no MuJoCo counterpart.** It is upstream Isaac's own term, and the
  standard sim2real companion to a mass offset: an unmodelled payload moves the centre of mass
  as well as its magnitude.

Still not randomized, and still Phase 5 per the roadmap: terrain variants, **motor strength**
(`randomize_actuator_gains` is the term), and sensor/command latency (the real G1 is ≈18–30 ms).

### The command distribution

Two corrections to upstream, both needed for this to be the same *task* as the joystick env:

- **`lin_vel_x` (0.0, 1.0) → (−1.0, 1.0).** Upstream's G1 trains forward-only. A joystick
  policy that cannot walk backwards is not the policy Phase 2 trained.
- **`heading_command = True → False`.** Upstream derives yaw rate from a heading controller in
  *every* env (`rel_heading_envs = 1.0`), so the sampled `ang_vel_z` range never reaches the
  robot. Playground commanded all three channels directly.

`rel_standing_envs = 0.05` stands in for Playground's per-channel zeroing (keep-probabilities
`[0.9, 0.25, 0.5]`), which Isaac's command term cannot express — it offers only all-or-nothing
standing envs. This is the one place the two task definitions genuinely differ in kind rather
than in tuning.

### Status: trained twice, walks in neither — and the second failure is understood

> [!warning] The Isaac policy does **not** walk yet. Do not treat this section as delivered.
> Two full-size attempts, both in [[experiments]]:
>
> 1. **256 envs × 1500** — genuinely undertrained (9.2M samples against upstream's 147M).
>    Learned to stand: 100 % survival, zero translation. This is the run that exposed the
>    "256 envs is the ceiling" guess as wrong by 16× (see [[setup]]).
> 2. **4096 envs × 1792** — upstream's own sample count, and it *still* does not translate.
>    It turns instead: 0.75 of a commanded 1.0 rad/s, with MAE **equal to the command** on
>    all three linear channels. `feet_air_time` was flat at ~0.01 from iteration 199 to 1599.
>
> The second is not a compute problem, and the flat reward-term curve is how you can tell —
> mean reward rose the whole time (−30 → +4.11) on the yaw term alone. **Mean reward is not
> a progress signal for this task**; `feet_air_time` and `track_lin_vel_xy_exp` are.
>
> Cause and fix are in [[decisions]] (2026-08-08): upstream's stepping reward is gated on the
> *linear* command norm, so a commanded turn earns nothing for stepping — which is right for
> upstream's decaying heading command and wrong for our persistent direct-yaw joystick, where
> pivoting on the spot collects the whole yaw reward for free. `dome_g1/mdp.py` widens the gate
> to all three channels. **That fix is committed and unvalidated**; a 150-iteration probe was
> too short to discriminate and is logged as such rather than as evidence.
>
> Next: the cloud run in `sims/isaac/README.md` § The cloud run, which carries the abort
> criterion — if `feet_air_time` is still flat by iteration ~500, the gate was not the cause
> and `action_rate_l2` (dominant at −0.41) is the next suspect, at the cost of gait smoothness.

### Comparing against the MuJoCo baseline

`sims/isaac/scripts/play_g1_flat.py` holds one command for a whole rollout and reports
per-channel tracking MAE and survival over a five-command sweep (forward, backward, strafe,
turn, stand). Those are the same quantities `sims/mujoco/scripts/compare_gaits.py` produces, so
the trained MJX policy stays useful as the **gait sanity baseline**: stride timing, duty factor
and velocity-tracking error give numbers an Isaac policy on flat ground should land near.
