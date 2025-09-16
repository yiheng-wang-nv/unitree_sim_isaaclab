# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaacsim.core.utils.torch.rotations import euler_angles_to_quats
from isaaclab.envs import mdp

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.envs import ManagerBasedEnv

from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
# 如需 Dex3：
from tasks.common_observations.dex3_state import get_robot_dex3_joint_states

def time_out_self_defined(env: ManagerBasedEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


G1_LOCOMANIPULATION_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            "omniverse://isaac-dev.ov.nvidia.com/Projects/agile/Robots/Collected_g1/g1_minimal_with_leg_collision.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,

        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.79),
        rot=(0.7071, 0, 0, 0.7071),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            # waist joints
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            
            # arms joints
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            
            # fingers joints
            "left_hand_index_0_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            
            "right_hand_index_0_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", 
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint", 
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=0.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            damping={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=1200.0,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_.*_joint": 400.0,
                ".*_elbow_joint": 600.0,
                ".*_wrist_.*_joint": 600.0,
            },
            damping={
                ".*_shoulder_.*_joint": 4.0,
                ".*_elbow_joint": 6.0,
                ".*_wrist_.*_joint": 6.0,
            },
            armature=None,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_index_.*_joint",
                ".*_hand_middle_.*_joint",
                ".*_hand_thumb_.*_joint"
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*": 100.0,
            },
            damping={
                ".*": 10.0,
            },
            armature={
                ".*": 0.1
            },
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)

USD_PATH = "omniverse://isaac-dev.ov.nvidia.com/Library/IsaacHealthcare/0.3.0"


##
# Scene definition
##
@configclass
class FixedBaseUpperBodyIKG1SceneCfg(InteractiveSceneCfg):
    """Scene configuration for fixed base upper body IK environment with G1 robot.

    This configuration sets up the G1 humanoid robot with fixed pelvis and legs,
    allowing only arm manipulation while the base remains stationary. The robot is
    controlled using upper body IK.
    """

    # Unitree G1 Humanoid robot - fixed base configuration
    robot: ArticulationCfg = G1_LOCOMANIPULATION_ROBOT_CFG.replace(spawn=G1_LOCOMANIPULATION_ROBOT_CFG.spawn.replace(usd_path="/localhome/local-vennw/code/data/g1_29dof_with_hand_rev_1_0_bck.usd"))

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(semantic_tags=[("class", "ground")]),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot_pov_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotPOVCam",
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=8, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(0.0978, 0.0917, 1.3136), rot=euler_angles_to_quats(torch.tensor([-123, 0.0, 0.0]), degrees=True), convention="ros"),
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={
            "class:table": (0, 255, 0, 255),
            "class:organ": (0, 0, 255, 255),
            "class:robot": (255, 255, 0, 255),
            "class:ground": (255, 0, 0, 255),
            "class:object": (0, 255, 255, 255),
            "class:black_sorting_bin": (255, 0, 255, 255),
            "class:UNLABELLED": (0, 0, 0, 255),
        },
    )

    def __post_init__(self):
        """Post initialization."""
        # Set the robot to fixed base
        self.robot.spawn.articulation_props.fix_root_link = True


hand_joint_names = [
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint',
    'left_hand_index_0_joint',
    'left_hand_middle_0_joint',
    'left_hand_thumb_0_joint',
    'right_hand_index_0_joint',
    'right_hand_middle_0_joint',
    'right_hand_thumb_0_joint',
    'left_hand_index_1_joint',
    'left_hand_middle_1_joint',
    'left_hand_thumb_1_joint',
    'right_hand_index_1_joint',
    'right_hand_middle_1_joint',
    'right_hand_thumb_1_joint',
    'left_hand_thumb_2_joint',
    'right_hand_thumb_2_joint',
]
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=hand_joint_names, use_default_offset=False, preserve_order=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    This class is required by the environment configuration but not used in this implementation
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        robot_joint_state = ObsTerm(func=get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=get_robot_dex3_joint_states)

        image_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    success = DoneTerm(func=time_out_self_defined, time_out=True)


def reset_robot_to_fixed_pose(env, env_ids):
    robot = env.scene["robot"]
    
    fixed_root_pos = torch.tensor([0.0, 0.0, 1.05], device=env.device).repeat(len(env_ids), 1)
    fixed_root_quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=env.device).repeat(len(env_ids), 1)
    fixed_root_vel = torch.zeros((len(env_ids), 6), device=env.device)
    
    fixed_joint_pos = torch.zeros((len(env_ids), robot.num_joints), device=env.device)
    fixed_joint_vel = torch.zeros((len(env_ids), robot.num_joints), device=env.device)
    
    joint_names = robot.joint_names
    joint_targets = {
        "left_hip_yaw_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "left_hip_pitch_joint": -0.05,
        "left_knee_joint": 0.2,
        "left_ankle_pitch_joint": -0.15,
        "left_ankle_roll_joint": 0.0,
        
        "right_hip_yaw_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.05,
        "right_knee_joint": 0.2,
        "right_ankle_pitch_joint": -0.15,
        "right_ankle_roll_joint": 0.0,
    }
    
    for joint_name, target_pos in joint_targets.items():
        if joint_name in joint_names:
            joint_idx = joint_names.index(joint_name)
            fixed_joint_pos[:, joint_idx] = target_pos
    
    fixed_root_pos += env.scene.env_origins[env_ids]
    
    robot.write_root_pose_to_sim(
        torch.cat([fixed_root_pos, fixed_root_quat], dim=1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(fixed_root_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(fixed_joint_pos, fixed_joint_vel, env_ids=env_ids)


@configclass 
class EventCfg:
    """Configuration for events."""

    # the reset scene to event function already resets all rigid objects and articulations to their default states.
    # this needs to be executed before any other reset function, to not overwrite the reset scene to default.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_pose = EventTerm(func=reset_robot_to_fixed_pose, mode="reset")


@configclass
class FixedBaseUpperBodyIKG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 fixed base upper body IK environment.

    This environment is designed for manipulation tasks where the G1 humanoid robot
    has a fixed pelvis and legs, allowing only arm and hand movements for manipulation. The robot is
    controlled using upper body IK.
    """

    # Scene settings
    scene: FixedBaseUpperBodyIKG1SceneCfg = FixedBaseUpperBodyIKG1SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        # self.episode_length_s = 20.0
        self.episode_length_s = 200.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.viewer.eye = (0.0, 1.8, 1.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
