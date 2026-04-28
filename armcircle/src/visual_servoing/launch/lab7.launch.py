from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, EmitEvent, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # -------------------------
    # Declare args
    # -------------------------

    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="true") # make false if you don't want rviz to launch when launching moveit

    # -------------------------
    # Includes & Nodes
    # -------------------------

    # RealSense (include rs_launch.py)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
    )

    # TF Broadcaster (wrist -> camera)
    static_tf_node = Node(
        package='visual_servoing',
        executable='tf',
        name='tf_node',
        output='screen'
    )

    # Aruco Launch
    aruco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros2_aruco'),
                'launch',
                'aruco_recognition.launch.py'
            )
        ),
    )

    # MoveIt include
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ur_moveit_config"),
                "launch",
                "ur_moveit.launch.py"
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz
        }.items(),
    )

    # -----------------------------------
    # Global shutdown on any process exit
    # -----------------------------------

    shutdown_on_any_exit = RegisterEventHandler(
        OnProcessExit(
            on_exit=[EmitEvent(event=Shutdown(reason='A launched process exited'))]
        )
    )

    # -------------------------
    # LaunchDescription
    # -------------------------

    return LaunchDescription([

        # Actions
        realsense_launch,
        aruco_launch,
        moveit_launch,
        static_tf_node,

        # Global handler (keep at end)
        shutdown_on_any_exit,
    ])
