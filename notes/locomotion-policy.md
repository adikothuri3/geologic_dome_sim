---
title: The G1 locomotion policy — rewards, observations, randomization
updated: 2026-08-04
status: current
---

# The G1 locomotion policy

What Phase 2 actually trains, and what every part of it means. Everything here is read from
the installed `mujoco_playground` G1 joystick environment, not from memory — re-derive with
`scripts/check_phase2.py` and the sources under `~/src/playground/.../locomotion/g1/`.

The reward function is **upstream's, unmodified**. That is the [[decisions]] call from
2026-08-01: the novelty of this project is the terrain loop, not the locomotion baseline.
Reward tuning belongs in Phase 5, against reconstructed terrain.

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
