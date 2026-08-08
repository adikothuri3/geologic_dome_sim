"""Reward terms that differ from upstream's, and why.

The project rule is that rewards are upstream's, unmodified (notes/decisions.md,
2026-08-01). This module is the one exception, and it exists because upstream's term is
correct for upstream's task and wrong for ours in a way that is not a matter of taste.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def feet_air_time_joystick(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """`feet_air_time_positive_biped`, gated on the full command instead of its linear part.

    Upstream's version ends with::

        reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    — the norm of the **linear** channels only. A commanded turn with no translation
    therefore earns nothing for stepping, which is right for upstream's task and wrong
    for ours:

    * Upstream sets ``heading_command=True`` with ``rel_heading_envs=1.0``, so its yaw
      command is a *decaying heading correction* — the robot turns to face a target and
      the yaw command goes to zero. Yaw is a transient there, never a standing incentive.
    * We command yaw directly (``heading_command=False``), matching the Playground
      joystick task. Yaw is then a **persistent** command worth the full
      ``track_ang_vel_z_exp`` weight of 1.0, indefinitely.

    Measured consequence, run `2026-08-08-isaac-g1fc-flat-dr-4096` (notes/experiments.md):
    the policy discovered it could collect the entire yaw reward by pivoting on the spot —
    no gait, no swing phase, and near-zero ``action_rate_l2``, which was the dominant term
    at −0.41. ``feet_air_time`` sat at ~0.01 from iteration 199 to 1599 and
    ``track_lin_vel_xy_exp`` plateaued at 0.37 (what standing still scores) by iteration
    200. It converged to "turn, never step" by iteration ~400 and 1,400 further iterations
    changed nothing — so this is a reward-structure problem and no amount of compute fixes
    it. The eval bears it out: 0.75 of a commanded 1.0 rad/s yaw, and **zero** translation.

    Using the full 3-vector norm makes a commanded turn a *stepping* turn, which is what a
    humanoid turning under a joystick actually does, and what Playground's own reward
    achieves by a different route (a ``feet_phase`` gait-clock term that Isaac's G1 has no
    equivalent of). Everything else — single-stance requirement, the air/contact-time
    minimum, the threshold clamp — is upstream's, untouched.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # The one change: all three commanded channels count, not just the linear two.
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward
