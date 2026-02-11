#!usr/bin/env python
import numpy as np
import scipy as sp
import kin_func_skeleton as kfs 

def ur7e_foward_kinematics_from_angles(joint_angles):
    """
    Calculate the orientation of the ur7e's end-effector tool given
    the joint angles of each joint in radians

    Parameters:
    ------------
    joint_angles ((6x) np.ndarray): 6 joint angles (s0, s1, e0, w1, w2, w3)

    Returns: 
    ------------
    (4x4) np.ndarray: homogenous transformation matrix
    """
    q0 = np.ndarray((3, 6)) # Points on each joint axis in the zero config
    w0 = np.ndarray((3, 6)) # Axis vector of each joint axis in the zero config


    q0[:, 0] = [0., 0., 0.1625] # shoulder pan joint - shoulder_link from base link
    q0[:, 1] = [0., 0., 0.1625] # shoulder lift joint - upper_arm_link from shoulder_link
    q0[:, 2] = [0.425, 0., 0.1625] # elbow_joint - forearm_link from shoulder_lift_joint
    q0[:, 3] = [0.817, 0.1333, 0.1625] # wrist 1 - wrist_1_link from elbow_joint
    q0[:, 4] = [0.817, 0.1333, 0.06285] # wrist 2 - wrist_2_link from wrist_1
    q0[:, 5] = [0.817, 0.233, 0.06285] # wrist 3 - wrist_3_link from wrist_2

    w0[:, 0] = [0., 0., 1] # shoulder pan joint
    w0[:, 1] = [0, 1., 0] # shoulder lift joint
    w0[:, 2] = [0., 1., 0] # elbow_joint
    w0[:, 3] = [0., 1., 0] # wrist 1
    w0[:, 4] = [0., 0., -1] # wrist 2 
    w0[:, 5] = [0., 1., 0] # wrist 3

    # Rotation matrix from base_link to wrist_3_link in zero config
    R = np.array([[-1., 0., 0.],
                  [0., 0., 1.], 
                  [0., 1., 0.]])

    # YOUR CODE HERE (Task 1)
    T = np.eye(4)

    for i in range(6):
        # Extract the joint axis and joint displacement
        omega = w0[:, i]
        q = q0[:, i]
        theta = joint_angles[i]  # Joint angle

        xi = np.concatenate([q, omega])  # Twist representation
        T_joint = kfs.homog_3d(xi, theta)  # Compute the joint's transformation matrix
        
        # Update the overall transformation matrix by multiplying with the current joint's transformation
        T = T @ T_joint

    # Apply the final rotation (if needed, depending on the robot's configuration)
    T[0:3, 0:3] = R @ T[0:3, 0:3]

    return T

def ur7e_forward_kinematics_from_joint_state(joint_state):
    """
    Computes the orientation of the ur7e's end-effector given the joint
    state.

    Parameters
    ----------
    joint_state (sensor_msgs.JointState): JointState of ur7e robot

    Returns
    -------
    (4x4) np.ndarray: homogenous transformation matrix
    """
    # Extract joint angles from joint_state object 
    j1, j2, j3, j4, j5, j6 = joint_state.position  # joint_state.position should be a list of angles
    
    # Convert to a numpy array
    angles = np.array([j1, j2, j3, j4, j5, j6])
    
    T = ur7e_foward_kinematics_from_angles(angles)  # Call the function you implemented in Task 1
    
    return T
