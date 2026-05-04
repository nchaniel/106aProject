#!usr/bin/env python
import numpy as np
import scipy as sp
<<<<<<< HEAD
import forward_kinematics.kin_func_skeleton as kfs
=======
import forward_kinematics.kin_func_skeleton as kfs 
from sensor_msgs.msg import JointState
>>>>>>> 0445bcc (Complete lab3)

def ur7e_forward_kinematics_from_angles(joint_angles):
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

<<<<<<< HEAD
    # define gst(0)
    # this represents the tool frame relative to base when all angles are 0
    # use the point on the last joint (wrist_3) for the translation
    q_tool = q0[:, 5] 
    gst0 = np.eye(4)
    gst0[:3, :3] = R      
    gst0[:3, 3] = q_tool  #initial tool position
=======
    # YOUR CODE HERE (Task 1)
    q = q0[:, 5].reshape(-1,1)
    gst = np.concatenate((np.concatenate((R, q), axis=1), np.array([[0,0,0,1]])), axis=0)

    xi0 = np.zeros((6,6))
    for i in range(6):
        w = w0[:, i]
        v0 = -1 * np.cross(w, q0[:, i])
        xi = np.concatenate((v0, w), axis=0)
        xi0[:,i] = xi

    G = kfs.prod_exp(xi0, joint_angles) @ gst
    # print(G)
    return G
>>>>>>> 0445bcc (Complete lab3)

    # compute PoE
    prod_exp = np.eye(4)
    for i in range(6):
        omega = w0[:, i]
        q = q0[:, i]
        theta = joint_angles[i]

        # calculate the linear velocity component of the twist
        # v = -omega x q
        v = np.cross(-omega, q)
        xi = np.concatenate([v, omega]) #twist
        
        # calculate e^(xi * theta)
        exp_xi_theta = kfs.homog_3d(xi, theta)
        
        # multiply them in order
        prod_exp = prod_exp @ exp_xi_theta 

    # apply the product of exponentials to the initial condition to get gst(theta)
    return prod_exp @ gst0

def ur7e_forward_kinematics_from_joint_state(joint_state): #not necessary as it was sorted in the node
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
    
<<<<<<< HEAD
    # Convert to a numpy array
    angles = np.array([j1, j2, j3, j4, j5, j6])
    
    T = ur7e_foward_kinematics_from_angles(angles)  # Call the function you implemented in Task 1
    
    return T
=======
    angles = np.zeros(6)
    # YOUR CODE HERE (Task 2)
    names, pos = joint_state.name, joint_state.position
    for i in range(len(names)):
        n = names[i]
        if n == "shoulder_pan_joint":
            angles[0] = pos[i]
        if n == "shoulder_lift_joint":
            angles[1] = pos[i]
        if n == "elbow_joint":
            angles[2] = pos[i]
        if n == "wrist_1_joint":
            angles[3] = pos[i]
        if n == "wrist_2_joint":
            angles[4] = pos[i]
        if n == "wrist_3_joint":
            angles[5] = pos[i]
            
    return ur7e_forward_kinematics_from_angles(angles)
    # END YOUR CODE HERE

>>>>>>> 0445bcc (Complete lab3)
