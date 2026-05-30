import gymnasium as gym


gym.register(
    id="HCIS-DiningCleanup-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dining_cleanup_env_cfg:DiningCleanupEnvCfg",
    },
)
