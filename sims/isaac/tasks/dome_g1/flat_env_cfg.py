"""Full-body-collision G1 on a flat plane — env + PPO runner configs.

Subclasses Isaac Lab's stock G1 flat velocity task and swaps the robot from
G1_MINIMAL_CFG (collision meshes stripped) to G1_CFG (full collision). Everything
else — rewards, observations, domain randomization events — is upstream's,
unmodified, matching the project's standing decision to not rebuild locomotion
baselines from scratch (notes/decisions.md 2026-08-01, reaffirmed 2026-08-07).

Verified against Isaac Lab v2.3.2: G1FlatEnvCfg inherits G1RoughEnvCfg, whose
__post_init__ sets `self.scene.robot = G1_MINIMAL_CFG.replace(...)` — so the swap
must happen after super().__post_init__().
"""

from isaaclab.utils import configclass
from isaaclab_assets import G1_CFG  # full-collision g1.usd

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class G1FullCollisionFlatEnvCfg(G1FlatEnvCfg):
    """Stock flat velocity task, full-collision robot."""

    def __post_init__(self):
        super().__post_init__()
        # The one change: every link's collision mesh stays live.
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class DomeG1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    """Upstream PPO hyperparameters; only the experiment name is ours."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "dome_g1fc_flat"
