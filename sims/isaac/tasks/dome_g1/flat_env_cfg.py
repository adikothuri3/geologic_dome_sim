"""Full-body-collision G1 on a flat plane — env + PPO runner configs.

Two tasks live here:

  Dome-G1FullCollision-Flat-v0      baseline: upstream's flat velocity task, full-collision robot
  Dome-G1FullCollision-Flat-DR-v0   the same, plus the Phase-2 domain randomization set

Both subclass Isaac Lab's stock G1 flat velocity task and swap the robot from
G1_MINIMAL_CFG (collision meshes stripped) to G1_CFG (full collision). Rewards and
observations stay upstream's, unmodified — the project's standing decision is that the
novelty is the terrain loop, not the locomotion baseline (notes/decisions.md 2026-08-01,
reaffirmed 2026-08-07).

Verified against Isaac Lab v2.3.2: G1FlatEnvCfg inherits G1RoughEnvCfg, whose
__post_init__ sets `self.scene.robot = G1_MINIMAL_CFG.replace(...)` — so the swap
must happen after super().__post_init__().

> Why the DR variant exists. `G1RoughEnvCfg.__post_init__` *disables* most of upstream's
> randomization: `push_robot = None`, `add_base_mass = None`, `base_com = None`, reset
> velocities zeroed, `reset_robot_joints` scale pinned to (1.0, 1.0), and the physics
> material ranges collapsed to point values (static 0.8→0.8, dynamic 0.6→0.6). What is
> left in the stock G1 flat task is observation noise and nothing else — no dynamics
> randomization at all. The MuJoCo Phase-2 policy (notes/locomotion-policy.md) trained
> under a per-environment `domain_randomize`: friction, joint friction, armature, every
> link mass, torso mass, initial joint pose, and pushes. DomeG1DREventCfg below is that
> set, rebuilt with Isaac's EventManager terms.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets import G1_CFG  # full-collision g1.usd

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)

from . import mdp as dome_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)


@configclass
class G1FullCollisionFlatEnvCfg(G1FlatEnvCfg):
    """Stock flat velocity task, full-collision robot. No dynamics randomization.

    Kept as the A/B baseline for the DR variant below — same rewards, same commands,
    same robot, so a run against this isolates what the randomization costs and buys.
    """

    def __post_init__(self):
        super().__post_init__()
        # The one change: every link's collision mesh stays live.
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


##
# Domain randomization
##


@configclass
class DomeG1DREventCfg:
    """Per-environment domain randomization, mirroring MuJoCo Playground's `domain_randomize`.

    Declared from scratch rather than by patching upstream's `EventCfg`, because upstream's
    term names assume a body called `base`; the G1 calls it `torso_link`, and silently
    randomizing nothing is the failure mode this whole class exists to fix. Every term here
    is asserted live by `check_isaac.py --gate c`, which reads the per-env spread back out
    of PhysX rather than trusting the config.

    Mapping from notes/locomotion-policy.md ("Domain randomization"):

    | MuJoCo Playground          | here                                            |
    | -------------------------- | ----------------------------------------------- |
    | floor/foot friction U(.4,1) | `physics_material` static friction (0.4, 1.0)   |
    | joint friction ×U(0.5, 2)   | **no counterpart** — the G1 USD has 0.0 friction |
    | armature ×U(1.0, 1.05)      | `joint_parameters` armature, scale              |
    | every link mass ×U(.9, 1.1) | `link_mass`, scale, body_names=".*"             |
    | torso mass +U(-1, +1) kg    | `torso_mass`, add, body_names="torso_link"      |
    | initial joint pose ±0.05    | `reset_robot_joints` (offset, not scale)        |
    | push 0.1–2.0 m/s / 5–10 s   | `push_robot`, interval 5–10 s                   |

    Two deliberate departures, both documented in notes/locomotion-policy.md:

    * `torso_com` has no MuJoCo counterpart. It is upstream Isaac's own term (which
      `G1RoughEnvCfg` disables) and is the standard sim2real companion to a mass offset —
      an unmodelled payload shifts the centre of mass as well as its magnitude.
    * MuJoCo pushes at a sampled *magnitude* in [0.1, 2.0] m/s in a random direction;
      Isaac's `push_by_setting_velocity` samples a box in (vx, vy). The box below is
      ±1.4, whose corner magnitude is 1.98 m/s — the same worst case, a milder mean.
    """

    # -- startup: the physics each environment is stuck with for its lifetime

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Terrain friction combines multiplicatively with the robot's (the plane is
            # 1.0/1.0), so these ranges ARE the effective coefficients. 0.4 is packed
            # snow; the low end is what the Everest terrain is eventually for.
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,  # dynamic <= static, or PhysX is being lied to
        },
    )

    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (1.0, 1.05),
            "operation": "scale",
            # No `friction_distribution_params`, and that is a finding rather than an
            # omission: the G1 USD ships joint friction of exactly 0.0 on all 37 joints,
            # and a `scale` operation cannot lift zero — the term would sit in the config
            # looking like randomization while multiplying nothing. MuJoCo's ×U(0.5, 2.0)
            # scales a nonzero `frictionloss` that this asset simply does not carry.
            # `check_isaac.py --gate c` asserts the zero, so a future asset that does have
            # joint friction shows up as a prompt to randomize it instead of staying dark.
        },
    )

    # -- reset: sampled fresh at the start of every episode

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},  # spawn at rest, as Playground does
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # Playground perturbs the nominal pose by an ABSOLUTE ±0.05 rad. Upstream
            # Isaac scales it instead (`reset_joints_by_scale`), which leaves joints whose
            # default is 0.0 exactly untouched — 17 of the G1's 29. Offset, not scale.
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # -- interval: the disturbance

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-1.4, 1.4), "y": (-1.4, 1.4)}},
    )


@configclass
class G1FullCollisionFlatDREnvCfg(G1FullCollisionFlatEnvCfg):
    """The Phase-4a task: velocity-command tracking under domain randomization.

    The Isaac counterpart of the MuJoCo Phase-2 joystick policy. Rewards are untouched;
    what changes is the physics distribution the policy is trained across, the command
    distribution it is asked to realise, and the sensor noise it must not trust.
    """

    def __post_init__(self):
        super().__post_init__()

        # -- the randomization ------------------------------------------------------
        # Installed HERE, after the whole parent chain, and not as a class field.
        # `G1RoughEnvCfg.__post_init__` reaches into `self.events` and rewrites it —
        # `push_robot = None`, `add_base_mass = None`, `reset_robot_joints`'s range
        # pinned to (1.0, 1.0). Declaring `events: DomeG1DREventCfg` as a field would
        # hand it our object to dismantle, and the two terms it disables are the two
        # this task most needs. Assigning last means upstream's edits land on the cfg
        # we are about to discard.
        self.events = DomeG1DREventCfg()

        # -- commands: match the joystick task the MuJoCo policy learned -----------
        # Upstream G1 flat trains forward-only (lin_vel_x 0…1) and derives yaw rate from
        # a heading controller in every env (heading_command=True, rel_heading_envs=1.0),
        # so the sampled ang_vel_z range never reaches the robot. Playground commanded
        # all three channels directly. A joystick policy that cannot walk backwards is
        # not the same policy, so both get corrected here.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.heading = None  # unused; None silences the warn
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # Playground zeroes each channel independently (keep-probabilities 0.9/0.25/0.5),
        # which makes "stand still" a real and frequent command. Isaac's command term only
        # offers all-or-nothing standing envs; 5% is the closest single knob. Uniform
        # sampling already supplies near-zero commands in every channel.
        self.commands.base_velocity.rel_standing_envs = 0.05

        # -- the stepping reward has to know about yaw -----------------------------
        # Upstream gates feet_air_time on the LINEAR command norm, so a commanded turn
        # earns nothing for stepping. That is right for upstream, whose yaw command is a
        # decaying heading correction, and wrong here, where yaw is commanded directly
        # and persistently: run 2026-08-08-isaac-g1fc-flat-dr-4096 collected the whole
        # yaw reward by pivoting on the spot and never learned to translate at all.
        # `.func` only — weight, params and the rest of the term stay upstream's.
        self.rewards.feet_air_time.func = dome_mdp.feet_air_time_joystick

        # -- observation noise: Playground's magnitudes ----------------------------
        # lin_vel ±0.1, ang_vel ±0.2, gravity ±0.05 and joint_vel ±1.5 already match
        # upstream Isaac exactly. Joint position is the one that does not: Isaac corrupts
        # it by ±0.01, Playground by ±0.03. Encoder noise on the real G1 is the whole
        # reason this term exists, so take the larger, hardware-honest number.
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.03, n_max=0.03)


@configclass
class G1FullCollisionFlatDRPlayEnvCfg(G1FullCollisionFlatDREnvCfg):
    """Evaluation variant: a handful of envs, clean sensors, no pushes.

    Randomized *dynamics* stay on — the point of the DR policy is that it works on
    physics it never saw, and an eval that quietly restores nominal physics cannot show
    that. What is removed is anything that makes a rollout unreadable: sensor corruption
    (so tracking error measures the policy, not the noise) and shoves (so a fall is the
    gait failing rather than a 1.4 m/s kick).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        # Hold one command for the whole clip; play_g1_flat.py pins it anyway, but a
        # 10 s resample would otherwise change the behaviour mid-video.
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)


##
# PPO runners
##


@configclass
class DomeG1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    """Upstream PPO hyperparameters; only the experiment name is ours."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "dome_g1fc_flat"


@configclass
class DomeG1FlatDRPPORunnerCfg(G1FlatPPORunnerCfg):
    """Same hyperparameters again — the variable under test is the physics, not PPO."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "dome_g1fc_flat_dr"
