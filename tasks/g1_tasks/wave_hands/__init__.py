
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym

from . import wave_g1_env_cfg


gym.register(
    id="Isaac-Wave-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": wave_g1_env_cfg.FixedBaseUpperBodyIKG1EnvCfg,
    },
    disable_env_checker=True,
)

