import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

from roboticstoolbox import models

robot = models.UR5()


class TrajectoryClient(Node):
    def __init__(self):
        super().__init__('traj_client')

        self.joint_state = None #Initializes the variable storing the current joint state

        self.client = ActionClient( # This connects the node with the driver that actually moves the joints
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.create_subscription( # This creates a connection between the node and the current joint states from the sensors
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
    def joint_state_callback(self, msg): # This retreives the joint positions from the /joint_state message
        self.joint_state = msg
    def get_current_positions(self, joint_names): # This puts names to the joint values
        if self.joint_state is None:
            raise RuntimeError("No joint state received yet")

        name_to_index = {
            name: i for i, name in enumerate(self.joint_state.name)
        }

        return [
            self.joint_state.position[name_to_index[j]]
            for j in joint_names
        ]

    def forward_kinematics(self, q):
        return self.robot.fkine(q).t
    def compute_jacobian(self, q):
        return self.robot.jacob0(q)
    def compute_ik(self, target_pos, max_iters=100, alpha=0.1):
        q = np.array(self.get_current_positions([
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]))

        for _ in range(max_iters):
            current_pos = self.forward_kinematics(q)
            error = target_pos - current_pos

            if np.linalg.norm(error) < 1e-3:
                break

            J = self.compute_jacobian(q)
            dq = alpha * np.linalg.pinv(J) @ error

            q = q + dq

        return q.tolist()
    def send_goal(self):
        self.client.wait_for_server()

        goal = FollowJointTrajectory.Goal()

        joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        current_positions = self.get_current_positions(joint_names) # This sets current_positions to the current joint positions

        start_point = JointTrajectoryPoint()
        start_point.positions = current_positions
        start_point.time_from_start.sec = 0

        target_point = JointTrajectoryPoint()
        target_pos = np.array([0.4, -0.2, 0.3])

        ik_solution = self.compute_ik(target_pos)

        target_point.positions = ik_solution
        target_point.time_from_start.sec = 2

        goal.trajectory.joint_names = joint_names
        goal.trajectory.points = [start_point, target_point]

        self.get_logger().info("Sending trajectory...")
        self.client.send_goal_async(goal)


def main():
    rclpy.init()

    node = TrajectoryClient()
    rclpy.spin_once(node)
    node.send_goal()
    node.destroy_node()


if __name__ == '__main__':
    main()