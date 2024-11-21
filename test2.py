import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi

# Simulation parameters
g = 9.81  # gravitational acceleration (m/s^2)
m = 0.2  # mass of the quadrotor (kg)
Ixx = 1  # moment of inertia about the x-axis
Iyy = 1  # moment of inertia about the y-axis
Izz = 1  # moment of inertia about the z-axis
T = 15  # duration of each trajectory segment (s)

# Proportional coefficients
Kp_x = 1  # proportional gain for x-axis
Kp_y = 1  # proportional gain for y-axis
Kp_z = 1  # proportional gain for z-axis
Kp_roll = 25  # proportional gain for roll
Kp_pitch = 25  # proportional gain for pitch
Kp_yaw = 25  # proportional gain for yaw

# Derivative coefficients
Kd_x = 10  # derivative gain for x-axis
Kd_y = 10  # derivative gain for y-axis
Kd_z = 1  # derivative gain for z-axis

# Circular trajectory parameters
r = 2  # radius of the circle
omega = 2 * pi / T  # angular velocity
h = 3  # constant height

# Animation flag
show_animation = True

class Quadrotor:
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=0.25, show_animation=True):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.show_animation = show_animation

        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            fig.canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            self.ax = fig.add_subplot(111, projection='3d')

        self.update_pose(x, y, z, roll, pitch, yaw)

    def update_pose(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)

        if self.show_animation:
            self.plot()

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll), z]
             ])

    def plot(self):
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        plt.cla()

        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')

        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')

        self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:')

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        self.ax.set_zlim(0, 10)

        plt.pause(0.001)

def quad_sim(r, omega, h, T):
    x_pos = r
    y_pos = 0
    z_pos = h
    x_vel = 0
    y_vel = 0
    z_vel = 0
    x_acc = 0
    y_acc = 0
    z_acc = 0
    roll = 0
    pitch = 0
    yaw = 0
    roll_vel = 0
    pitch_vel = 0
    yaw_vel = 0

    des_yaw = 0

    dt = 0.1

    quad = Quadrotor(x_pos, y_pos, z_pos, roll, pitch, yaw, show_animation=show_animation)

    time = 0.0
    while time <= T:
        des_x_pos = r * cos(omega * time)
        des_y_pos = r * sin(omega * time)
        des_z_pos = h

        des_x_vel = -r * omega * sin(omega * time)
        des_y_vel = r * omega * cos(omega * time)
        des_z_vel = 0

        des_x_acc = -r * omega**2 * cos(omega * time)
        des_y_acc = -r * omega**2 * sin(omega * time)
        des_z_acc = 0

        x_acc = Kp_x * (des_x_pos - x_pos) + Kd_x * (des_x_vel - x_vel)
        y_acc = Kp_y * (des_y_pos - y_pos) + Kd_y * (des_y_vel - y_vel)
        z_acc = Kp_z * (des_z_pos - z_pos) + Kd_z * (des_z_vel - z_vel)

        x_vel += x_acc * dt
        y_vel += y_acc * dt
        z_vel += z_acc * dt

        x_pos += x_vel * dt + (0.5 * x_acc * dt**2)
        y_pos += y_vel * dt + (0.5 * y_acc * dt**2)
        z_pos += z_vel * dt + (0.5 * z_acc * dt**2)

        pitch = cos(yaw) * (x_acc + g * sin(pitch)) - sin(yaw) * (y_acc + g * sin(roll)) / g
        roll = sin(yaw) * (x_acc + g * sin(pitch)) + cos(yaw) * (y_acc + g * sin(roll)) / g

        pitch_vel = Kp_pitch * (pitch - pitch_vel)
        roll_vel = Kp_roll * (roll - roll_vel)
        yaw_vel = Kp_yaw * (des_yaw - yaw_vel)

        yaw += yaw_vel * dt

        quad.update_pose(x_pos, y_pos, z_pos, roll, pitch, yaw)

        time += dt

    return [quad.x_data, quad.y_data, quad.z_data]

if __name__ == '__main__':
    [x_data, y_data, z_data] = quad_sim(r, omega, h, T)

    if show_animation:
        plt.plot(x_data, y_data, z_data, 'b')
        plt.grid(True)
        plt.show()
