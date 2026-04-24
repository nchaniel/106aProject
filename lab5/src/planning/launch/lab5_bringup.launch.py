from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, EmitEvent
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # -------------------------
    # Declare args
    # -------------------------

    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="true")
    robot_ip = LaunchConfiguration("robot_ip", default="192.168.1.102")
    shutdown_on_exit = LaunchConfiguration("shutdown_on_exit", default="true")

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
        launch_arguments={
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '640x480x30',
        }.items(),
    )

    # UR robot driver (loads URDF into robot_state_publisher, publishes TF chain)
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_robot_driver'),
                'launch',
                'ur_control.launch.py'
            )
        ),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': robot_ip,
            'launch_rviz': 'false',
            'initial_joint_controller': 'scaled_joint_trajectory_controller',
        }.items(),
    )

    # Perception node
    perception_node = Node(
        package='perception',
        executable='detection_node',
        name='detection_node',
        output='screen'
    )

    # Planning TF node
    planning_tf_node = Node(
        package='planning',
        executable='tf',
        name='tf_node',
        output='screen'
    )

    # MoveIt include
    moveit_launch_file = os.path.join(
        get_package_share_directory("ur_moveit_config"),
        "launch",
        "ur_moveit.launch.py"
    )
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_launch_file),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz
        }.items(),
    )

    # IK Planner node
    ik_planner_node = Node(
        package='planning',
        executable='ik',
        name='ik_planner',
        output='screen'
    )

    # -------------------------
    # Global shutdown on any process exit (gated — disable with shutdown_on_exit:=false for debugging)
    # -------------------------
    shutdown_on_any_exit = RegisterEventHandler(
        OnProcessExit(
            on_exit=[EmitEvent(event=Shutdown(reason='A launched process exited'))]
        ),
        condition=IfCondition(shutdown_on_exit),
    )

    # -------------------------
    # LaunchDescription
    # -------------------------
    return LaunchDescription([

        # Actions
        realsense_launch,
        ur_control_launch,
        perception_node,
        planning_tf_node,
        moveit_launch,
        ik_planner_node,

        # Global handler (keep at end)
        shutdown_on_any_exit,
    ])
