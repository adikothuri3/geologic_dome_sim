"""Dome task registrations: the full-body-collision G1 velocity tasks.

Import this package AFTER AppLauncher has started the SimulationApp (isaaclab_tasks
imports pull in omni.* modules that need a running Kit). Registers:

  Dome-G1FullCollision-Flat-v0   flat plane, G1_CFG (every collision mesh live)

The stock `Isaac-Velocity-Flat-G1-v0` uses G1_MINIMAL_CFG, which strips most collision
meshes for speed -- the same feet-only shortcut MuJoCo Playground took, and the same
reason it is wrong for this project (see notes/decisions.md, 2026-08-04: a policy that
has never felt a shin or hip meet rock has a blind spot). G1_CFG's g1.usd keeps the
full collision geometry.
"""

import gymnasium as gym

from .flat_env_cfg import DomeG1FlatPPORunnerCfg, G1FullCollisionFlatEnvCfg

gym.register(
    id="Dome-G1FullCollision-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FullCollisionFlatEnvCfg,
        "rsl_rl_cfg_entry_point": DomeG1FlatPPORunnerCfg,
    },
)
