from glob import glob
from setuptools import find_packages, setup

package_name = 'tp_final_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/maps', glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tato',
    maintainer_email='lmoralesarce@udesa.edu.ar',
    description='TODO: Package description',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            "obstacle_navigation_node = tp_final_package.obstacle_navigation:main",
            "fast_slam_node = tp_final_package.fast_slam:main",
            # Nuevo nodo de localización basado en fast_slam.py
            "fast_slam_localization_node = tp_final_package.fast_slam_localization:main",
            # Planner + controlador (A*)
            "a_estrella_node = tp_final_package.a_estrella:main",
        ],
    },
)
