#!/usr/bin/env python3
"""
Trajectory classes for Lab 7
Defines Linear, Circular, and ScanOrbit trajectories for the UR7e end effector

Author: EECS 106B Course Staff, Spring 2026
         ScanOrbitTrajectory added for dish-scanning project
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


class Trajectory:
    """Base trajectory class"""

    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
            Duration of the trajectory in seconds
        """

        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns desired end-effector pose at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            7D vector [x, y, z, qx, qy, qz, qw] where:
            - (x, y, z) is position in meters
            - (qx, qy, qz, qw) is orientation as quaternion
            - Gripper pointing down corresponds to quaternion [0, 1, 0, 0]
        """

        raise NotImplementedError

    def target_velocity(self, time):
        """
        Returns desired end-effector velocity at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            6D twist [vx, vy, vz, wx, wy, wz] where:
            - (vx, vy, vz) is linear velocity in m/s
            - (wx, wy, wz) is angular velocity in rad/s
        """

        raise NotImplementedError

    def display_trajectory(self, num_waypoints=100, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.

        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animation : bool
            if True, saves a gif of the animated trajectory
        """

        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])

        fig = plt.figure(figsize=(12, 10))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Row 1: time series (position and translational velocity vs time)
        ax_ts_pos = fig.add_subplot(2, 2, 1)
        ax_ts_pos.plot(times, target_positions[:, 0], label='x')
        ax_ts_pos.plot(times, target_positions[:, 1], label='y')
        ax_ts_pos.plot(times, target_positions[:, 2], label='z')
        ax_ts_pos.set_ylabel('Position')
        ax_ts_pos.set_title('Target Position')
        ax_ts_pos.legend()
        ax_ts_pos.grid(True)

        ax_ts_vel = fig.add_subplot(2, 2, 2, sharex=ax_ts_pos)
        ax_ts_vel.plot(times, target_velocities[:, 0], label='vx')
        ax_ts_vel.plot(times, target_velocities[:, 1], label='vy')
        ax_ts_vel.plot(times, target_velocities[:, 2], label='vz')
        ax_ts_vel.set_xlabel('Time')
        ax_ts_vel.set_ylabel('Velocity')
        ax_ts_vel.set_title('Target Velocity')
        ax_ts_vel.legend()
        ax_ts_vel.grid(True)

        # Row 2: 3D evolution of position and body-frame velocity
        ax0 = fig.add_subplot(2, 2, 3, projection='3d')
        pos_padding = [[-0.1, 0.1],
                       [-0.1, 0.1],
                       [-0.1, 0.1]]
        ax0.set_xlim3d([min(target_positions[:, 0]) + pos_padding[0][0],
                        max(target_positions[:, 0]) + pos_padding[0][1]])
        ax0.set_ylim3d([min(target_positions[:, 1]) + pos_padding[1][0],
                        max(target_positions[:, 1]) + pos_padding[1][1]])
        ax0.set_zlim3d([min(target_positions[:, 2]) + pos_padding[2][0],
                        max(target_positions[:, 2]) + pos_padding[2][1]])
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        ax0.set_title(f'{trajectory_name} evolution of end-effector\'s position.')
        line0 = ax0.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], c=colormap, s=2)

        ax1 = fig.add_subplot(2, 2, 4, projection='3d')
        vel_padding = [[-0.1, 0.1],
                       [-0.1, 0.1],
                       [-0.1, 0.1]]
        ax1.set_xlim3d([min(target_velocities[:, 0]) + vel_padding[0][0],
                        max(target_velocities[:, 0]) + vel_padding[0][1]])
        ax1.set_ylim3d([min(target_velocities[:, 1]) + vel_padding[1][0],
                        max(target_velocities[:, 1]) + vel_padding[1][1]])
        ax1.set_zlim3d([min(target_velocities[:, 2]) + vel_padding[2][0],
                        max(target_velocities[:, 2]) + vel_padding[2][1]])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'{trajectory_name} evolution of end-effector\'s translational body-frame velocity.')
        line1 = ax1.scatter(target_velocities[:, 0], target_velocities[:, 1], target_velocities[:, 2], c=colormap, s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints,
                                                          fargs=([line0, line1],),
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))),
                                                          blit=False)
        fig.tight_layout()
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)


class LinearTrajectory(Trajectory):
    """
    Straight line trajectory from start to goal position.
    Uses trapezoidal velocity profile (constant acceleration, constant velocity, constant deceleration).
    """

    def __init__(self, start_position, goal_position, total_time):
        """
        Parameters
        ----------
        start_position : np.ndarray
            3D starting position [x, y, z] in meters
        goal_position : np.ndarray
            3D goal position [x, y, z] in meters
        total_time : float
            Total duration of trajectory in seconds
        """

        super().__init__(total_time)

        self.start_position = np.array(start_position)
        self.goal_position = np.array(goal_position)

        # Calculate trajectory parameters
        self.direction = self.goal_position - self.start_position
        self.distance = np.linalg.norm(self.direction)

        if self.distance > 0:
            self.unit_direction = self.direction / self.distance
        else:
            self.unit_direction = np.zeros(3)

        # Trapezoidal velocity profile parameters
        # Accelerate for first half, decelerate for second half
        self.t_half = total_time / 2.0
        self.v_max = 2.0 * self.distance / total_time  # Peak velocity
        self.acceleration = self.v_max / self.t_half

        # Gripper pointing down quaternion
        self.orientation = np.array([0.0, 1.0, 0.0, 0.0])  # [qx, qy, qz, qw]

    def target_pose(self, time):
        """
        Returns desired end-effector pose at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            7D vector [x, y, z, qx, qy, qz, qw] where:
            - (x, y, z) is position in meters
            - (qx, qy, qz, qw) is orientation as quaternion
            - Gripper pointing down corresponds to quaternion [0, 1, 0, 0]
        """

        # Clamp time
        t = np.clip(time, 0, self.total_time)

        if t <= self.t_half:
            s = 0.5 * self.acceleration * t**2
        else:
            time_remaining = self.total_time - t
            s = self.distance - 0.5 * self.acceleration * time_remaining**2

        # Combine position and orientation
        pos = self.start_position + s * self.unit_direction
        return np.concatenate([pos, self.orientation])

    def target_velocity(self, time):
        """
        Returns desired end-effector velocity at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            6D twist [vx, vy, vz, wx, wy, wz] where:
            - (vx, vy, vz) is linear velocity in m/s
            - (wx, wy, wz) is angular velocity in rad/s
        """

        # Clamp time
        t = np.clip(time, 0, self.total_time)

        if t <= self.t_half:
            speed = self.acceleration * t
        else:
            time_remaining = self.total_time - t
            speed = self.acceleration * time_remaining

        vel = speed * self.unit_direction

        return np.concatenate([vel, np.zeros(3)])


class CircularTrajectory(Trajectory):
    """
    Circular trajectory around a center point in a horizontal plane.
    Uses angular trapezoidal velocity profile.
    """

    def __init__(self, center_position, radius, total_time):
        """
        Parameters
        ----------
        center_position : np.ndarray
            3D center position [x, y, z] in meters
        radius : float
            Radius of circle in meters
        total_time : float
            Total duration of trajectory in seconds
        """
        super().__init__(total_time)

        self.center_position = np.array(center_position)
        self.radius = radius

        # Angular trapezoidal profile (complete circle = 2π radians)
        self.total_angle = 2 * np.pi
        self.t_half = total_time / 2.0
        self.angular_v_max = 2.0 * self.total_angle / total_time
        self.angular_acceleration = self.angular_v_max / self.t_half

        # Gripper pointing down quaternion
        self.orientation = np.array([0.0, 1.0, 0.0, 0.0])

    def target_pose(self, time):
        t = np.clip(time, 0, self.total_time)

        if t <= self.t_half:
            theta = 0.5 * self.angular_acceleration * t**2
        else:
            time_remaining = self.total_time - t
            theta = self.total_angle - 0.5 * self.angular_acceleration * time_remaining**2

        pos = self.center_position + self.radius * np.array([np.cos(theta), np.sin(theta), 0])

        return np.concatenate([pos, self.orientation])

    def target_velocity(self, time):
        t = np.clip(time, 0, self.total_time)

        if t <= self.t_half:
            theta_dot = self.angular_acceleration * t
            theta = 0.5 * self.angular_acceleration * t**2
        else:
            time_remaining = self.total_time - t
            theta_dot = self.angular_acceleration * time_remaining
            theta = self.total_angle - 0.5 * self.angular_acceleration * time_remaining**2

        vel = self.radius * theta_dot * np.array([-np.sin(theta), np.cos(theta), 0])

        return np.concatenate([vel, np.zeros(3)])


class ScanOrbitTrajectory(Trajectory):
    """
    Orbital scanning trajectory for 3D dish reconstruction.

    The end-effector circles horizontally at a fixed height above the dish,
    completing one full revolution. Unlike CircularTrajectory, the wrist
    orientation continuously rotates so the camera (mounted at +Y on wrist_3_link
    per static_tf_transform.py) always faces the dish center.

    Orientation derivation
    ----------------------
    The base "gripper down" quaternion is q_down = [0, 1, 0, 0] (qx, qy, qz, qw).
    At orbit angle θ we additionally rotate around the world Z-axis by θ:
        q_z(θ) = [0, 0, sin(θ/2), cos(θ/2)]

    Composing q_z(θ) * q_down via the Hamilton product gives the closed form:
        q(θ) = [-sin(θ/2), cos(θ/2), 0, 0]

    Sanity check:
      θ=0   → [0,  1, 0, 0]  — original pointing-down orientation
      θ=π/2 → [-√2/2, √2/2, 0, 0]  — rotated 90° around Z
      θ=2π  → [0,  1, 0, 0]  — full circle, back to start ✓

    NOTE: If the camera still doesn't face the dish after running,
    the camera may be mounted at a different offset than assumed.
    Try negating the sin term (use +sin(θ/2)) to flip the tracking direction.
    """

    def __init__(self, dish_position, radius, scan_height, total_time):
        """
        Parameters
        ----------
        dish_position : np.ndarray
            3D position of the dish / AR tag [x, y, z] in base_link frame
        radius : float
            Orbit radius in meters. Should be large enough to clear the dish
            and give the camera a good viewing angle (typically 0.15–0.25 m).
        scan_height : float
            Height above dish_position to orbit at, in meters.
            Higher values give a more top-down view; lower values give more
            side-on coverage (better for tall dishes / garnishes).
        total_time : float
            Total scan duration in seconds. Slower orbits (20–30 s) give
            the camera time to settle at each position.
        """
        super().__init__(total_time)

        self.dish_position = np.array(dish_position)
        self.radius = radius
        self.scan_height = scan_height

        # The orbit center is directly above the dish
        self.orbit_center = self.dish_position + np.array([0.0, 0.0, scan_height])

        # Trapezoidal angular velocity profile — same scheme as CircularTrajectory
        self.total_angle = 2.0 * np.pi
        self.t_half = total_time / 2.0
        self.angular_v_max = 2.0 * self.total_angle / total_time
        self.angular_acceleration = self.angular_v_max / self.t_half

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _theta(self, t):
        """Orbit angle [rad] at clamped time t (trapezoidal profile)."""
        t = np.clip(t, 0.0, self.total_time)
        if t <= self.t_half:
            return 0.5 * self.angular_acceleration * t**2
        else:
            tr = self.total_time - t
            return self.total_angle - 0.5 * self.angular_acceleration * tr**2

    def _theta_dot(self, t):
        """Angular velocity [rad/s] at clamped time t."""
        t = np.clip(t, 0.0, self.total_time)
        if t <= self.t_half:
            return self.angular_acceleration * t
        else:
            return self.angular_acceleration * (self.total_time - t)

    def _look_at_quat(self, orbit_pos):
        """
        Compute end-effector quaternion so the camera points at the dish.

        Constructs an orthonormal frame where:
        - tool Z → toward dish (camera optical axis)
        - tool X → tangential to orbit (keeps camera roll consistent)
        - tool Y → right-hand rule from Z and X

        This requires wrist_1 (pitch), wrist_2 (yaw), and wrist_3 (roll)
        to all move, unlike the previous pure-Z-rotation approach.
        """
        to_dish = self.dish_position - orbit_pos
        z_col = to_dish / np.linalg.norm(to_dish)  # tool Z = toward dish

        # Tangential direction in XY plane — sets camera roll/horizontal axis
        # Compute theta from orbit_pos rather than storing it separately
        dx = orbit_pos[0] - self.orbit_center[0]
        dy = orbit_pos[1] - self.orbit_center[1]
        theta = np.arctan2(dy, dx)
        x_col = np.array([-np.sin(theta), np.cos(theta), 0.0])

        # Orthogonalize x_col against z_col to keep the frame valid
        # (they won't be perfectly perpendicular in general)
        x_col = x_col - np.dot(x_col, z_col) * z_col
        x_col /= np.linalg.norm(x_col)

        # Right-hand rule: y = z × x
        y_col = np.cross(z_col, x_col)

        # Build rotation matrix — columns are tool axes expressed in world frame
        R_mat = np.column_stack([x_col, y_col, z_col])

        return Rot.from_matrix(R_mat).as_quat()  # [qx, qy, qz, qw]

    def capture_times(self, n_images):
        """
        Compute the exact times at which the robot has traversed each of
        n_images evenly-spaced angles around the orbit. Use these times to
        trigger image captures during execution so that the captured views
        are angularly uniform (important for photogrammetry quality).

        Inverts the trapezoidal profile:
          first-half  (θ ≤ π):  t = sqrt(2θ / α)
          second-half (θ > π):  t = T - sqrt(2(2π-θ) / α)

        Parameters
        ----------
        n_images : int
            Number of images to capture (evenly spaced in angle).

        Returns
        -------
        np.ndarray
            Array of capture times in seconds, shape (n_images,).
        """
        capture_angles = np.linspace(0.0, self.total_angle, n_images, endpoint=False)
        times = np.zeros(n_images)

        for i, theta in enumerate(capture_angles):
            if theta <= self.total_angle / 2.0:
                times[i] = np.sqrt(2.0 * theta / self.angular_acceleration)
            else:
                remaining_angle = self.total_angle - theta
                times[i] = self.total_time - np.sqrt(2.0 * remaining_angle / self.angular_acceleration)

        return times

    # ------------------------------------------------------------------
    # Trajectory interface
    # ------------------------------------------------------------------

    def target_pose(self, time):
        """
        Returns desired end-effector pose at time t.

        Returns
        -------
        np.ndarray
            7D vector [x, y, z, qx, qy, qz, qw]
        """
        theta = self._theta(time)
        pos = self.orbit_center + self.radius * np.array([
            np.cos(theta), np.sin(theta), 0.0
        ])

        # Replace the old closed-form quaternion with the look-at orientation
        quat = self._look_at_quat(pos)

        return np.concatenate([pos, quat])

    def target_velocity(self, time):
        """
        Returns desired end-effector velocity at time t.

        Returns
        -------
        np.ndarray
            6D twist [vx, vy, vz, wx, wy, wz]
            The angular velocity omega = [0, 0, theta_dot] reflects the
            wrist rotating around world Z at the same rate as the orbit.
        """
        theta = self._theta(time)
        theta_dot = self._theta_dot(time)

        # Linear velocity is unchanged
        lin_vel = self.radius * theta_dot * np.array([
            -np.sin(theta), np.cos(theta), 0.0
        ])

        # Angular velocity: the camera tilts toward the dish at a fixed elevation
        # angle (arctan(h/r)), and rotates around world Z at rate theta_dot.
        # Only the Z component changes with theta_dot; tilt is constant.
        ang_vel = np.array([0.0, 0.0, theta_dot])

        return np.concatenate([lin_vel, ang_vel])


# ---------------------------------------------------------------------------
# Helpers for define_trajectories / CLI testing
# ---------------------------------------------------------------------------

def define_trajectories(args):
    """Define each type of trajectory with the appropriate parameters."""

    trajectory = None
    if args.task == 'line':
        start = np.array([0.3, 0.2, 0.3])
        goal = np.array([0.5, 0.4, 0.4])
        total_time = 5.0
        trajectory = LinearTrajectory(start, goal, total_time)

    elif args.task == 'circle':
        center = np.array([0.4, 0.3, 0.3])
        radius = 0.1
        total_time = 8.0
        trajectory = CircularTrajectory(center, radius, total_time)

    elif args.task == 'scan':
        # Example: dish at robot's reachable forward position
        dish_pos = np.array([0.45, 0.0, 0.1])
        radius = 0.18
        scan_height = 0.28
        total_time = 20.0
        trajectory = ScanOrbitTrajectory(dish_pos, radius, scan_height, total_time)
        print(f"Capture times for 36 images: {trajectory.capture_times(36).round(2)}")

    return trajectory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default='line',
                        help='Options: line, circle, scan.  Default: line')
    parser.add_argument('--animate', action='store_true',
                        help='Show animated trajectory.')
    args = parser.parse_args()

    trajectory = define_trajectories(args)

    if trajectory:
        trajectory.display_trajectory(show_animation=args.animate)