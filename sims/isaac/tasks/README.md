# `sims/isaac/tasks/`

- `dome_g1/` — registers **`Dome-G1FullCollision-Flat-v0`**: Isaac Lab's stock G1 flat
  velocity task with the full-collision `G1_CFG` swapped in for `G1_MINIMAL_CFG`
  (env cfg + PPO runner cfg in `flat_env_cfg.py`). Import the package after
  AppLauncher starts the app; `check_isaac.py --gate b` and `train_g1_flat.py` do this.

Terrain-variant tasks (Phase 4b: recon/procedural mountain terrain) land here next.
