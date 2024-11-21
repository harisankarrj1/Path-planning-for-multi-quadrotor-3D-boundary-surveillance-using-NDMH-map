from math import cos, sin
import numpy as np
from Quadrotor import Quadrotor
from TrajectoryGenerator import TrajectoryGenerator

show_animation = True

# Simulation parameters
g = 9.81
m = 0.2
Ixx = 1
Iyy = 1
Izz = 1
T = 5

# Proportional coefficients
Kp_x = 1
Kp_y = 1
Kp_z = 1
Kp_roll = 25
Kp_pitch = 25
Kp_yaw = 25

# Derivative coefficients
Kd_x = 10
Kd_y = 10
Kd_z = 1


def quad_sim(x_c1, y_c1, z_c1, x_c2, y_c2, z_c2):
    """
    Simulates two quadrotors simultaneously and plots their trajectories.
    """
    # Initial positions and velocities for Drone 1
    x_pos1, y_pos1, z_pos1 = -5, -5, 5
    x_vel1, y_vel1, z_vel1 = 0, 0, 0
    roll1, pitch1, yaw1 = 0, 0, 0
    roll_vel1, pitch_vel1, yaw_vel1 = 0, 0, 0

    # Initial positions and velocities for Drone 2
    x_pos2, y_pos2, z_pos2 = 5, 5, 5
    x_vel2, y_vel2, z_vel2 = 0, 0, 0
    roll2, pitch2, yaw2 = 0, 0, 0
    roll_vel2, pitch_vel2, yaw_vel2 = 0, 0, 0

    des_yaw = 0
    dt = 0.1
    t = 0

    # Create Quadrotor objects for both drones
    q1 = Quadrotor(x=x_pos1, y=y_pos1, z=z_pos1, roll=roll1,
                   pitch=pitch1, yaw=yaw1, size=1, show_animation=show_animation)
    q2 = Quadrotor(x=x_pos2, y=y_pos2, z=z_pos2, roll=roll2,
                   pitch=pitch2, yaw=yaw2, size=1, show_animation=show_animation)

    i = 0
    n_run = 8
    irun = 0

    while True:
        while t <= T:
            # Calculate desired positions and accelerations for Drone 1
            des_z_pos1 = calculate_position(z_c1[i], t)
            des_z_vel1 = calculate_velocity(z_c1[i], t)
            des_x_acc1 = calculate_acceleration(x_c1[i], t)
            des_y_acc1 = calculate_acceleration(y_c1[i], t)
            des_z_acc1 = calculate_acceleration(z_c1[i], t)

            # Calculate desired positions and accelerations for Drone 2
            des_z_pos2 = calculate_position(z_c2[i], t)
            des_z_vel2 = calculate_velocity(z_c2[i], t)
            des_x_acc2 = calculate_acceleration(x_c2[i], t)
            des_y_acc2 = calculate_acceleration(y_c2[i], t)
            des_z_acc2 = calculate_acceleration(z_c2[i], t)

            # Thrust and torques for Drone 1
            thrust1 = m * (g + des_z_acc1 + Kp_z * (des_z_pos1 - z_pos1) + Kd_z * (des_z_vel1 - z_vel1))
            roll_torque1 = Kp_roll * (((des_x_acc1 * sin(des_yaw) - des_y_acc1 * cos(des_yaw)) / g) - roll1)
            pitch_torque1 = Kp_pitch * (((des_x_acc1 * cos(des_yaw) - des_y_acc1 * sin(des_yaw)) / g) - pitch1)
            yaw_torque1 = Kp_yaw * (des_yaw - yaw1)

            # Thrust and torques for Drone 2
            thrust2 = m * (g + des_z_acc2 + Kp_z * (des_z_pos2 - z_pos2) + Kd_z * (des_z_vel2 - z_vel2))
            roll_torque2 = Kp_roll * (((des_x_acc2 * sin(des_yaw) - des_y_acc2 * cos(des_yaw)) / g) - roll2)
            pitch_torque2 = Kp_pitch * (((des_x_acc2 * cos(des_yaw) - des_y_acc2 * sin(des_yaw)) / g) - pitch2)
            yaw_torque2 = Kp_yaw * (des_yaw - yaw2)

            # Update angular velocities for both drones
            roll_vel1 += roll_torque1 * dt / Ixx
            pitch_vel1 += pitch_torque1 * dt / Iyy
            yaw_vel1 += yaw_torque1 * dt / Izz

            roll_vel2 += roll_torque2 * dt / Ixx
            pitch_vel2 += pitch_torque2 * dt / Iyy
            yaw_vel2 += yaw_torque2 * dt / Izz

            # Update orientations for both drones
            roll1 += roll_vel1 * dt
            pitch1 += pitch_vel1 * dt
            yaw1 += yaw_vel1 * dt

            roll2 += roll_vel2 * dt
            pitch2 += pitch_vel2 * dt
            yaw2 += yaw_vel2 * dt

            # Rotation matrices
            R1 = rotation_matrix(roll1, pitch1, yaw1)
            R2 = rotation_matrix(roll2, pitch2, yaw2)

            # Accelerations for both drones
            acc1 = (np.matmul(R1, np.array([0, 0, thrust1.item()]).T) - np.array([0, 0, m * g]).T) / m
            acc2 = (np.matmul(R2, np.array([0, 0, thrust2.item()]).T) - np.array([0, 0, m * g]).T) / m

            x_acc1, y_acc1, z_acc1 = acc1
            x_acc2, y_acc2, z_acc2 = acc2

            # Update velocities for both drones
            x_vel1 += x_acc1 * dt
            y_vel1 += y_acc1 * dt
            z_vel1 += z_acc1 * dt

            x_vel2 += x_acc2 * dt
            y_vel2 += y_acc2 * dt
            z_vel2 += z_acc2 * dt

            # Update positions for both drones
            x_pos1 += x_vel1 * dt
            y_pos1 += y_vel1 * dt
            z_pos1 += z_vel1 * dt

            x_pos2 += x_vel2 * dt
            y_pos2 += y_vel2 * dt
            z_pos2 += z_vel2 * dt

            # Update quadrotor poses
            q1.update_pose(x_pos1, y_pos1, z_pos1, roll1, pitch1, yaw1)
            q2.update_pose(x_pos2, y_pos2, z_pos2, roll2, pitch2, yaw2)

            t += dt

        t = 0
        i = (i + 1) % 4
        irun += 1
        if irun >= n_run:
            break

    print("Simulation Complete")


def calculate_position(c, t):
    """
    Calculates a position given a set of quintic coefficients and a time.
    """
    return c[0] * t**5 + c[1] * t**4 + c[2] * t**3 + c[3] * t**2 + c[4] * t + c[5]


def calculate_velocity(c, t):
    """
    Calculates a velocity given a set of quintic coefficients and a time.
    """
    return 5 * c[0] * t**4 + 4 * c[1] * t**3 + 3 * c[2] * t**2 + 2 * c[3] * t + c[4]


def calculate_acceleration(c, t):
    """
    Calculates an acceleration given a set of quintic coefficients and a time.
    """
    return 20 * c[0] * t**3 + 12 * c[1] * t**2 + 6 * c[2] * t + 2 * c[3]


def rotation_matrix(roll, pitch, yaw):
    """
    Calculates the ZYX rotation matrix.
    """
    return np.array(
                [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll)],
         [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll)],
         [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw)]
         ])


def main():
    """
    Calculates the x, y, z coefficients for the four segments
    of the trajectory
    """
    x_coeffs1 = [[], [], [], []]
    y_coeffs1 = [[], [], [], []]
    z_coeffs1 = [[], [], [], []]

    x_coeffs2 = [[], [], [], []]
    y_coeffs2 = [[], [], [], []]
    z_coeffs2 = [[], [], [], []]

    # Initial waypoints for both drones
    initial_waypoints = [[-5, -5, 5], [5, 0, 5]]
    radius1 = 5
    radius2 =8
    center = [0, 0, 5]
    num_points = 10

    # Generate circular waypoints
    theta = np.linspace(0, 2 * np.pi, num_points)
    circular_waypoints = [[center[0] + radius1 * np.cos(t), center[1] + radius1 * np.sin(t), center[2]] for t in theta]
    circular_waypoints2 = [[center[0] + radius2 * np.cos(t), center[1] + radius2 * np.sin(t), center[2]] for t in theta]

    # Assign waypoints for the trajectories
    waypoints1 = circular_waypoints
    waypoints2 = circular_waypoints

    for i in range(4):
        traj1 = TrajectoryGenerator(waypoints1[i], waypoints1[(i + 1) % 4], T)
        traj1.solve()
        x_coeffs1[i] = traj1.x_c
        y_coeffs1[i] = traj1.y_c
        z_coeffs1[i] = traj1.z_c

        traj2 = TrajectoryGenerator(waypoints2[i], waypoints2[(i + 1) % 4], T)
        traj2.solve()
        x_coeffs2[i] = traj2.x_c
        y_coeffs2[i] = traj2.y_c
        z_coeffs2[i] = traj2.z_c

    quad_sim(x_coeffs1, y_coeffs1, z_coeffs1, x_coeffs2, y_coeffs2, z_coeffs2)


if __name__ == "__main__":
    main()
