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
        
        self.joint_names = [
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'shoulder_pan_joint',
        ]
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
        
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_validated', 10)
        
        self.running = True
        threading.Thread(target=self.keyboard_loop, daemon=True).start()

    def joint_state_callback(self, msg: JointState):
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
        self.got_joint_states = True

    def keyboard_loop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            self.get_logger().info("Keyboard controller running. Increment: 123456 | Decrement: qwerty | Ctrl+C to exit")
            while self.running:
                try:
                    if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == '\x03':
                            return
                        self.handle_key(key)
                except KeyboardInterrupt:
                    self.running = False
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def handle_key(self, key):
        if not self.got_joint_states:
            self.get_logger().info("Waiting for joint states...")
            return
        
        new_positions = self.joint_positions.copy()
        if key in INCREMENT_KEYS:
            idx = INCREMENT_KEYS.index(key)
            new_positions[idx] = JointAngles[key]
        
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = new_positions
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 1
        traj.points.append(point)
        self.pub.publish(traj)

        self.joint_positions = new_positions

def main(args=None):
    rclpy.init(args=args)
    joints = sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
    node = KeyboardController(joints)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.running = False
        print("\nExiting keyboard controller...")
    finally:
        node.destroy_node()
>>>>>>> 0445bcc (Complete lab3)
        rclpy.shutdown()

if __name__ == "__main__":
    main()
