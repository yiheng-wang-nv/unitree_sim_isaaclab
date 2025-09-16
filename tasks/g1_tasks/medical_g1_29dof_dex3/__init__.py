
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym
import os

from . import medical_g1_29dof_dex3_joint_env_cfg


gym.register(
    id="Isaac-medical-G129-Dex3-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": medical_g1_29dof_dex3_joint_env_cfg.MedicalG129DEX3JointEnvCfg,
    },
    disable_env_checker=True,
)

