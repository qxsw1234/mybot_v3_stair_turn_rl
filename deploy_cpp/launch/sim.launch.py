"""Launch file for MuJoCo simulation mode.

Starts the C++ deploy_node in sim_mode, which communicates with
the Python MuJoCo simulation node via ROS2 topics.

NOTE: The Python MuJoCo sim node must be started separately in the
      mujoco_sim conda environment:
        conda activate mujoco_sim
        source /opt/ros/humble/setup.bash
        python3 sim/mujoco_sim_node.py

Usage (terminal 1 - MuJoCo sim):
  cd /home/getting/humble/Quadruped/elmap-rl-controller/deploy_cpp
  conda activate mujoco_sim && source /opt/ros/humble/setup.bash
  python3 sim/mujoco_sim_node.py --robot-config config/robots/mybot_v2_1_cse.yaml

Usage (terminal 2 - deploy_node):
  source /opt/ros/humble/setup.bash
  source ~/humble/Quadruped/HIMLoco/install/setup.bash
  ros2 launch deploy_cpp sim.launch.py
"""

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_path(pkg_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(pkg_dir, path_value)


def _launch_setup(context):
    pkg_dir = get_package_share_directory('deploy_cpp')
    cfg_file = LaunchConfiguration('robot_config_file').perform(context)

    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    urdf_relpath = cfg.get('urdf_relpath', '')
    if not urdf_relpath:
        raise RuntimeError('Missing urdf_relpath in robot_config_file')

    urdf_file = _resolve_path(pkg_dir, urdf_relpath)
    with open(urdf_file, 'r', encoding='utf-8') as f:
        robot_description = f.read()

    return [
        # Deploy node in simulation mode
        Node(
            package='deploy_cpp',
            executable='deploy_node',
            name='deploy_node',
            output='screen',
            parameters=[{
                'robot_config_file': LaunchConfiguration('robot_config_file'),
                'sim_mode': True,
                'sim_pingpong_mode': LaunchConfiguration('sim_pingpong_mode'),
                'debug_no_motor': False,
            }],
        ),

        # Robot state publisher (for RViz, optional)
        # MuJoCo sim node already publishes /joint_states
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
            }],
        ),
    ]


def generate_launch_description():
    pkg_dir = get_package_share_directory('deploy_cpp')
    default_cfg = os.path.join(pkg_dir, 'config', 'robots', 'mybot_v2_1_cse.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('robot_config_file', default_value=default_cfg,
                              description='Path to robot runtime yaml config'),
        DeclareLaunchArgument('sim_pingpong_mode', default_value='false',
                              description='Enable state-triggered ping-pong control timing in deploy_node'),
        OpaqueFunction(function=_launch_setup),
    ])
