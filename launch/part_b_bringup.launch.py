"""Bring-up completo para la Parte B: Gazebo + map_server + localización + planner + RViz."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("tp_final_package")
    default_map = os.path.join(pkg_share, "maps", "parte_a.yaml")

    world_arg = DeclareLaunchArgument(
        "world_launch",
        default_value="custom_casa.launch.py",
        description="Launch file del paquete turtlebot3_custom_simulation para el entorno.",
    )

    map_arg = DeclareLaunchArgument(
        "map",
        default_value=default_map,
        description="Mapa YAML generado en la Parte A.",
    )

    tb3_pkg = get_package_share_directory("turtlebot3_custom_simulation")
    world_condition = DeclareLaunchArgument(
        "launch_world",
        default_value="false",
        description="Set to true to lanzar también la simulación de TurtleBot3.",
    )

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    tb3_pkg,
                    "launch",
                    LaunchConfiguration("world_launch"),
                ]
            )
        ),
        condition=IfCondition(LaunchConfiguration("launch_world")),
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{"yaml_filename": LaunchConfiguration("map")}, {"frame_id": "map"}],
    )

    localization = Node(
        package="tp_final_package",
        executable="fast_slam_localization_node",
        name="fast_slam_localization_node",
        output="screen",
    )

    planner = Node(
        package="tp_final_package",
        executable="a_estrella_node",
        name="a_estrella_node",
        output="screen",
    )

    rviz_config = os.path.join(pkg_share, "rviz", "fast_slam.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
    )

    return LaunchDescription(
        [
            world_arg,
            world_condition,
            map_arg,
            world_launch,
            map_server,
            localization,
            planner,
            rviz,
        ]
    )
