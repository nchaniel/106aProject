"""
Merged Launch File: Lab 5 (Point Cloud Pick & Place) + Lab 7 (AR Tag Detection)
================================================================================

This launch file starts everything needed for the merged pipeline:
  - RealSense camera (point cloud)
  - Static TF broadcaster (camera → wrist_3_link)
  - Point cloud processor (publishes /cube_pose)
  - MoveIt (IK planning)
  - AR tag detection (from Lab 7's visual_servoing package)

Usage:
  ros2 launch planning merged_bringup.launch.py

Then in another terminal:
  ros2 run planning main_merged --ar_marker 0
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    EmitEvent,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # ─── Arguments ───────────────────────────────────────────────────
    ur_type     = LaunchConfiguration('ur_type',     default='ur7e')
    launch_rviz = LaunchConfiguration('launch_rviz', default='true')

    # ─── 1. RealSense Camera ────────────────────────────────────────
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch', 'rs_launch.py',
            )
        ),
        launch_arguments={
            'pointcloud.enable':        'true',
            'rgb_camera.color_profile': '1920x1080x30',
        }.items(),
    )

    # ─── 2. Static TF: camera_depth_optical_frame → wrist_3_link ───
    planning_tf_node = Node(
        package='planning',
        executable='tf',
        name='tf_node',
        output='screen',
    )

    # ─── 3. Point-Cloud Perception (publishes /cube_pose) ───────────
    perception_node = Node(
        package='perception',
        executable='process_pointcloud',
        name='process_pointcloud',
        output='screen',
    )

    # ─── 4. MoveIt (IK + motion planning services) ──────────────────
    moveit_launch_file = os.path.join(
        get_package_share_directory('ur_moveit_config'),
        'launch', 'ur_moveit.launch.py',
    )
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_launch_file),
        launch_arguments={
            'ur_type':     ur_type,
            'launch_rviz': launch_rviz,
        }.items(),
    )

    # ─── 5. AR Tag Detection (Lab 7) ────────────────────────────────
    # Lab 7 uses visual_servoing's lab7.launch.py which starts the AR
    # tag detector and publishes TF frames for ar_marker_<id>.
    # If your lab7.launch.py also launches RViz and the camera, you may
    # want to use only the AR-tag node instead to avoid duplicate camera
    # launches. Adjust the package/executable below to match your setup.
    #
    # OPTION A: Include the full lab7 launch (comment out realsense_launch
    #           above if lab7.launch.py already starts the camera)
    #
    # OPTION B: Launch only the AR tag detector node directly.
    #           Uncomment the appropriate block below.

    # ── OPTION A: full lab7 launch ──
    # NOTE: If lab7.launch.py already starts the RealSense camera,
    # comment out `realsense_launch` above to avoid conflicts.
    #
    # ar_tag_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory('visual_servoing'),
    #             'launch', 'lab7.launch.py',
    #         )
    #     ),
    # )

    # ── OPTION B: just the AR tag tracker node ──
    # This is the ros2_aruco or similar AR-tag node that publishes
    # TF frames like ar_marker_0. Adjust package/executable to match
    # whatever your lab7.launch.py uses internally.
    ar_tag_node = Node(
        package='ros2_aruco',          # ← adjust if your package name differs
        executable='aruco_node',       # ← adjust if your executable name differs
        name='aruco_node',
        output='screen',
        # If the AR node needs camera topic remapping:
        remappings=[
            ('image', '/camera/camera/color/image_raw'),
            ('camera_info', '/camera/camera/color/camera_info'),
        ],
    )

    # ─── 6. IK Planner node (Lab 5) ────────────────────────────────
    ik_planner_node = Node(
        package='planning',
        executable='ik',
        name='ik_planner',
        output='screen',
    )

    # ─── Global shutdown on any node crash ──────────────────────────
    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            on_exit=[EmitEvent(event=Shutdown(reason='A launched process exited'))],
        )
    )

    # ─── Assemble ───────────────────────────────────────────────────
    return LaunchDescription([
        realsense_launch,
        planning_tf_node,
        perception_node,
        moveit_launch,
        ar_tag_node,             # swap with ar_tag_launch for Option A
        ik_planner_node,
        shutdown_handler,
    ])
