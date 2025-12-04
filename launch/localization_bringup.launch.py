"""Launch map_server with Parte A map and the FastSLAM-based localization node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("tp_final_package")
    default_map = os.path.join(pkg_share, "maps", "parte_a.yaml")

    map_arg = DeclareLaunchArgument(
        "map",
        default_value=default_map,
        description="Ruta del archivo YAML del mapa a cargar en el map_server.",
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"yaml_filename": LaunchConfiguration("map")},
            {"frame_id": "map"},
        ],
    )

    localization = Node(
        package="tp_final_package",
        executable="fast_slam_localization_node",
        name="fast_slam_localization_node",
        output="screen",
    )

    return LaunchDescription([map_arg, map_server, localization])
