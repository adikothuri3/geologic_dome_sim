"""Full-body-collision G1 on the real Eiger Trail — env + PPO runner configs.

The terrain is a straightened 5.4 km x 24 m corridor of the actual Eiger Trail
(Eigergletscher -> Alpiglen, 713 m of descent, swissALTI3D 0.5 m DEM), built by
sims/isaac/terrain/{fetch_eiger_trail,build_trail_terrain,mesh_to_usd}.py into
data/eiger_trail/terrain/. Same terrain family as the GrandTour EIG-1 footage
the recon toolchain is benchmarked on: alpine rock-and-gravel descent.

Registers (registration lives in this module, not __init__.py, so the trail
task is importable standalone and the flat-task file stays untouched):

  Dome-G1FullCollision-EigerTrail-v0        the training task
  Dome-G1FullCollision-EigerTrail-Play-v0   16 envs, clean sensors — for eval

Three deliberate deviations from the stock rough task, all forced by
`terrain_type="usd"` (verified against Isaac Lab v2.3.2 source):

  1. Env origins: the stock usd path computes a grid at z=0 — meaningless on a
     strip that climbs 713 m. `EigerTrailImporter.configure_env_origins` loads
     the walkable spawn points build_trail_terrain.py wrote (origins.npz:
     centerline points filtered to <=1 m relief over a 2 m patch, so robots
     spawn standing; the cliffs stay in the mesh as terrain to walk into).
  2. `curriculum.terrain_levels = None`: terrain-level curriculum requires the
     terrain *generator*'s row/col origin layout. Distance-along-trail
     curriculum is a Phase 5 candidate (origins are ordered by arc length).
  3. The robot is G1_CFG (full collision), same swap as flat_env_cfg.py.

The height scanner needs no change: it raycasts mesh_prim_paths=["/World/ground"],
and the RayCaster takes the first Mesh child under that prim — our imported USD.

Phase 5 note: this cfg carries only upstream's rough-task events. The snow/ice
friction + wind-gust DR set belongs on top of this terrain eventually, mirroring
the flat -DR- task.
"""

from pathlib import Path

import numpy as np
import torch

from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG  # full-collision g1.usd

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg,
)

_REPO = Path(__file__).resolve().parents[4]
TERRAIN_DIR = _REPO / "data" / "eiger_trail" / "terrain"
USD_PATH = TERRAIN_DIR / "usd" / "eiger_trail.usd"
ORIGINS_PATH = TERRAIN_DIR / "origins.npz"


class EigerTrailImporter(TerrainImporter):
    """TerrainImporter whose env origins come from the trail's walkable spawn points.

    The base class's usd path would lay origins in a grid at z=0 (verified in
    v2.3.2 `_compute_env_origins_grid`); on this strip that puts every robot
    inside a mountain or hundreds of metres above it.
    """

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        spawn = np.load(self.cfg.origins_path)["origins"]  # (K, 3), z = ground height
        idx = np.arange(self.cfg.num_envs) % len(spawn)  # tile when num_envs > K
        self.terrain_origins = None
        self.env_origins = torch.tensor(spawn[idx], dtype=torch.float, device=self.device)


@configclass
class EigerTrailImporterCfg(TerrainImporterCfg):
    class_type: type = EigerTrailImporter
    origins_path: str = ""
    """Path to origins.npz written by build_trail_terrain.py."""


@configclass
class G1FullCollisionEigerTrailEnvCfg(G1RoughEnvCfg):
    """Stock rough velocity task; full-collision robot; real-trail USD terrain."""

    def __post_init__(self):
        super().__post_init__()
        if not USD_PATH.exists() or not ORIGINS_PATH.exists():
            raise FileNotFoundError(
                f"trail terrain assets missing under {TERRAIN_DIR} — run "
                "sims/isaac/terrain/fetch_eiger_trail.py, build_trail_terrain.py, "
                "then mesh_to_usd.py (see sims/isaac/terrain/README.md)"
            )

        # full-collision robot, same one change as the flat task
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # real-trail terrain instead of the procedural generator; keep upstream's
        # physics material (friction randomization over it is Phase 5)
        stock = self.scene.terrain
        self.scene.terrain = EigerTrailImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=str(USD_PATH),
            origins_path=str(ORIGINS_PATH),
            collision_group=-1,
            physics_material=stock.physics_material,
            env_spacing=2.0,  # unused by EigerTrailImporter; kept valid for the base checks
            debug_vis=False,
        )

        # terrain-level curriculum needs generator row/col origins — not available on usd
        self.curriculum.terrain_levels = None


@configclass
class G1FullCollisionEigerTrailPlayEnvCfg(G1FullCollisionEigerTrailEnvCfg):
    """Eval variant: few envs, clean observations, no external pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class DomeG1EigerTrailPPORunnerCfg(G1RoughPPORunnerCfg):
    """Upstream rough-task PPO hyperparameters; only the experiment name is ours."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "dome_g1fc_eiger_trail"


# -- registration ------------------------------------------------------------
import gymnasium as gym  # noqa: E402

gym.register(
    id="Dome-G1FullCollision-EigerTrail-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FullCollisionEigerTrailEnvCfg,
        "rsl_rl_cfg_entry_point": DomeG1EigerTrailPPORunnerCfg,
    },
)

gym.register(
    id="Dome-G1FullCollision-EigerTrail-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1FullCollisionEigerTrailPlayEnvCfg,
        "rsl_rl_cfg_entry_point": DomeG1EigerTrailPPORunnerCfg,
    },
)
