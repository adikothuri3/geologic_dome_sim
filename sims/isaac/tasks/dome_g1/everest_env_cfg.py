"""Full-body-collision G1 on the Everest summit pyramid — env + PPO runner configs.

The terrain is a ~2 km axis-aligned patch of the real summit pyramid (HMA 8 m
DEM, tile-677), built by sims/isaac/terrain/{fetch_everest_dem,
build_everest_terrain,mesh_to_usd}.py into data/everest/terrain/. Mirrors the
real Robot Everest 2026 objective terrain the same way the Eiger task mirrors
the GrandTour benchmark footage.

Registers (registration lives in this module, not __init__.py, same as the
Eiger task, so it stays importable standalone):

  Dome-G1FullCollision-Everest-v0        the training task
  Dome-G1FullCollision-Everest-Play-v0   16 envs, clean sensors — for eval

Reuses EigerTrailImporter from trail_env_cfg: it is already generic (loads
cfg.origins_path, tiles the (K,3) origins modulo num_envs — nothing
Eiger-specific), and importing the module only registers the Eiger gym ids,
which is side-effect-benign. Same three deviations from the stock rough task,
all forced by terrain_type="usd": custom origins, no terrain-level curriculum,
full-collision G1.

Caveat vs the Eiger strip: origins here are cells with slope <= 15 deg on an
8 m grid — sub-cell ledges the DEM cannot see may still exist under a spawn
point. check_everest.py's settle gate is the arbiter.
"""

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG  # full-collision g1.usd

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)

from .trail_env_cfg import EigerTrailImporterCfg

_REPO = Path(__file__).resolve().parents[4]
TERRAIN_DIR = _REPO / "data" / "everest" / "terrain"
USD_PATH = TERRAIN_DIR / "usd" / "everest_summit.usd"
ORIGINS_PATH = TERRAIN_DIR / "origins.npz"


@configclass
class G1FullCollisionEverestEnvCfg(G1RoughEnvCfg):
    """Stock rough velocity task; full-collision robot; real-summit USD terrain."""

    def __post_init__(self):
        super().__post_init__()
        if not USD_PATH.exists() or not ORIGINS_PATH.exists():
            raise FileNotFoundError(
                f"Everest terrain assets missing under {TERRAIN_DIR} — run "
                "sims/isaac/terrain/fetch_everest_dem.py, build_everest_terrain.py, "
                "then mesh_to_usd.py (see sims/isaac/terrain/README.md)"
            )

        # full-collision robot, same one change as the flat task
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # real-summit terrain instead of the procedural generator; keep upstream's
        # physics material (snow/ice friction DR over it is Phase 5)
        stock = self.scene.terrain
        self.scene.terrain = EigerTrailImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=str(USD_PATH),
            origins_path=str(ORIGINS_PATH),
            collision_group=-1,
            physics_material=stock.physics_material,
            env_spacing=2.0,  # unused by the importer; kept valid for the base checks
            debug_vis=False,
        )

        # terrain-level curriculum needs generator row/col origins — not available on usd
        self.curriculum.terrain_levels = None


@configclass
class G1FullCollisionEverestPlayEnvCfg(G1FullCollisionEverestEnvCfg):
    """Eval variant: few envs, clean observations, no external pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class DomeG1EverestPPORunnerCfg(G1RoughPPORunnerCfg):
    """Upstream rough-task PPO hyperparameters; only the experiment name is ours."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "dome_g1fc_everest"


# -- registration ------------------------------------------------------------
import gymnasium as gym  # noqa: E402

gym.register(
    id="Dome-G1FullCollision-Everest-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FullCollisionEverestEnvCfg,
        "rsl_rl_cfg_entry_point": DomeG1EverestPPORunnerCfg,
    },
)

gym.register(
    id="Dome-G1FullCollision-Everest-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FullCollisionEverestPlayEnvCfg,
        "rsl_rl_cfg_entry_point": DomeG1EverestPPORunnerCfg,
    },
)
