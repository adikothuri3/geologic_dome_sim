"""Dome task registrations: the full-body-collision G1 velocity tasks.

Import this package AFTER AppLauncher has started the SimulationApp (isaaclab_tasks
imports pull in omni.* modules that need a running Kit). Registers:

  Dome-G1FullCollision-Flat-v0        flat plane, G1_CFG, no dynamics randomization
  Dome-G1FullCollision-Flat-DR-v0     the same + the Phase-2 domain randomization set
  Dome-G1FullCollision-Flat-DR-Play-v0  16 envs, clean sensors, no pushes — for eval

The stock `Isaac-Velocity-Flat-G1-v0` uses G1_MINIMAL_CFG, which strips most collision
meshes for speed -- the same feet-only shortcut MuJoCo Playground took, and the same
reason it is wrong for this project (see notes/decisions.md, 2026-08-04: a policy that
has never felt a shin or hip meet rock has a blind spot). G1_CFG's g1.usd keeps the
full collision geometry.

`-DR-v0` is the Phase-4a training task and the Isaac counterpart of the MuJoCo Phase-2
joystick policy; `Flat-v0` is retained as its no-randomization A/B baseline. Why a
separate task rather than a flag: upstream's own G1 config *disables* almost all of
Isaac's randomization, so "does this task randomize anything" has to be answerable by
reading a task id (and by `check_isaac.py --gate c`), not by tracing a boolean.
"""

import gymnasium as gym

from .flat_env_cfg import (
    DomeG1FlatDRPPORunnerCfg,
    DomeG1FlatPPORunnerCfg,
    G1FullCollisionFlatDREnvCfg,
    G1FullCollisionFlatDRPlayEnvCfg,
    G1FullCollisionFlatEnvCfg,
)

gym.register(
    id="Dome-G1FullCollision-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FullCollisionFlatEnvCfg,
        "rsl_rl_cfg_entry_point": DomeG1FlatPPORunnerCfg,
    },
)

gym.register(
    id="Dome-G1FullCollision-Flat-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FullCollisionFlatDREnvCfg,
        "rsl_rl_cfg_entry_point": DomeG1FlatDRPPORunnerCfg,
    },
)

gym.register(
    id="Dome-G1FullCollision-Flat-DR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FullCollisionFlatDRPlayEnvCfg,
        "rsl_rl_cfg_entry_point": DomeG1FlatDRPPORunnerCfg,
    },
)
