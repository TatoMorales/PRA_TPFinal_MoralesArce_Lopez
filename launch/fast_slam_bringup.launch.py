"""Launch the FastSLAM node together with RViz for visualization."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_rviz_config = os.path.join(
        get_package_share_directory("tp_final_package"),
        "rviz",
        "fast_slam.rviz",
    )

    fast_slam = Node(
        package="tp_final_package",
        executable="fast_slam_node",
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz_config,
                description="RViz config file to load.",
            ),
            fast_slam,
            rviz,
        ]
    )
