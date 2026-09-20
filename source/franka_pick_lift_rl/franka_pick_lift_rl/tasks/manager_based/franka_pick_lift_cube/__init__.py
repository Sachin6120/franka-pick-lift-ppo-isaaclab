# Copyright (c) 2026 Sachin Kumar Pal

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Franka-PickLift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_pick_lift_cube_env_cfg:FrankaPickLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPickLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
