import gymnasium as gym


for task_id in (
    "LeIsaac-HCIS-DiningCleanup-SingleArm-v0",
    "HCIS-DiningCleanup-SingleArm-v0",
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.dining_cleanup_env_cfg:DiningCleanupEnvCfg",
        },
    )
