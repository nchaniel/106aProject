#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import sys
import tty
import termios
import threading
import select

<<<<<<< HEAD
<<<<<<< HEAD
class JointPosController(Node):
    def __init__(self, angles):
        super().__init__('joint_pos_controller')
=======
# Key mappings
INCREMENT_KEYS = ['1','2','3','4','5','6']
DECREMENT_KEYS = ['q','w','e','r','t','y']
JOINT_STEP = 0.15 # radians per key press

class KeyboardController(Node):
    def __init__(self, shoulder_lift_joint,
            elbow_joint,
            wrist_1_joint,
            wrist_2_joint,
            wrist_3_joint,
            shoulder_pan_joint):
        super().__init__('ur7e_keyboard_controller')
>>>>>>> 0445bcc (Complete lab3)
=======
class JointController(Node):
    def __init__(self, joint_array):
        super().__init__('ur7e_joint_controller')
>>>>>>> e2b32b5 (Complete lab 3)
        
        self.joint_names = [
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'shoulder_pan_joint',
        ]
<<<<<<< HEAD
<<<<<<< HEAD

        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_validated', 10)
    
        self.send_trajectory(angles)

    # same send_trajectory logic from keyboard_controller
    def send_trajectory(self, angles):
        #setup
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        # points 
        point = JointTrajectoryPoint()
        point.positions = angles
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 5 # Set to 5 for safety, as said in the instructions
    
        traj.points.append(point)
        
        self.pub.publish(traj)
        self.get_logger().info(f"Published target: {angles}")

def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) != 7:
        print(
            "Usage:\n"
            "ros2 run joint_control joint_pos_controller "
            "lift elbow w1 w2 w3 pan"
        )
        return

    # We use float() here to ensure they are numbers, not strings
    try:
        angles = [float(a) for a in sys.argv[1:]]
        node = JointPosController(angles)
        
        # We spin once 
        rclpy.spin(node)
    except ValueError:
        print("Error: All arguments must be numbers (radians).")
    finally:
=======
        self.JointAngles = [elf, shoulder_lift_joint,
            elbow_joint,
            wrist_1_joint,
            wrist_2_joint,
            wrist_3_joint,
            shoulder_pan_joint]

        self.joint_positions = [0.0] * 6
        self.got_joint_states = False  # Failsafe: don't publish until joint states received
        
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
=======
        self.JointAngles = joint_array
>>>>>>> e2b32b5 (Complete lab 3)
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_validated', 10)

        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = self.JointAngles
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 1
        traj.points.append(point)
        self.pub.publish(traj)
    

def main(args=None):
    rclpy.init(args=args)
    joints = [float(sys.argv[1]),float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])]
    print(joints)
    node = JointController(joints)
    try:
        rclpy.spin_once(node)
        sleep(1000)
    except KeyboardInterrupt:
        node.running = False
        print("\nExiting keyboard controller...")
    finally:
        node.destroy_node()
>>>>>>> 0445bcc (Complete lab3)
        rclpy.shutdown()

if __name__ == "__main__":
    main()
