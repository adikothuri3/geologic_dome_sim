"""Phase 2 smoke test: does JAX see the GPU, and does the G1 joystick env actually run?

    python scripts/check_phase2.py

Cheap and fast. Run this before any training so a broken CUDA install or a Playground API
change fails in seconds rather than twenty minutes into a run.
"""

import sys


def main() -> None:
    import jax

    print("== jax ==")
    print(f"  version {jax.__version__}")
    devices = jax.devices()
    for d in devices:
        print(f"  device  {d} (platform={d.platform})")
    gpus = [d for d in devices if d.platform == "gpu"]
    if not gpus:
        sys.exit(
            "FAIL: JAX sees no GPU.\n"
            "  - check `nvidia-smi` works inside WSL\n"
            "  - reinstall with: uv pip install -U 'jax[cuda12]'"
        )

    try:
        stats = gpus[0].memory_stats() or {}
        limit = stats.get("bytes_limit")
        if limit:
            print(f"  vram    {limit / 1024**3:.1f} GiB")
    except Exception:
        pass

    print("\n== mujoco playground ==")
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    env_name = "G1JoystickFlatTerrain"
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    print(f"  loaded {env_name}")
    print(f"  ctrl_dt {env_cfg.ctrl_dt}  sim_dt {env_cfg.sim_dt}  "
          f"episode_length {env_cfg.episode_length}")

    import jax.numpy as jp

    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    action = jp.zeros(env.action_size)
    state = jax.jit(env.step)(state, action)
    print(f"  action_size {env.action_size}")
    obs = state.obs
    if hasattr(obs, "items"):
        for k, v in obs.items():
            print(f"  obs[{k!r}] {v.shape}")
    else:
        print(f"  obs {obs.shape}")
    print(f"  reward after one step: {float(state.reward):.4f}")

    print("\n== ppo defaults (upstream) ==")
    p = locomotion_params.brax_ppo_config(env_name)
    for k in ("num_timesteps", "num_envs", "batch_size", "num_minibatches",
              "unroll_length", "num_evals", "learning_rate", "episode_length"):
        if k in p:
            print(f"  {k:20s} {p[k]}")
    print("\n  num_envs 8192 will OOM on an 8 GB 4060 Ti -- scripts/train_g1.py scales it "
          "down\n  and keeps batch_size * num_minibatches == num_envs.")

    print("\nPhase 2 environment OK.")


if __name__ == "__main__":
    main()
