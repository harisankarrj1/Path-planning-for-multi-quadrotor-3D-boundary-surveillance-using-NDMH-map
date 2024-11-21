import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev

from math import cos, sin
import numpy as np


class TrajectoryGenerator():
    def __init__(self, start_pos, des_pos, T, start_vel=[0,0,0], des_vel=[0,0,0], start_acc=[0,0,0], des_acc=[0,0,0]):
        self.start_x = start_pos[0]
        self.start_y = start_pos[1]
        self.start_z = start_pos[2]

        self.des_x = des_pos[0]
        self.des_y = des_pos[1]
        self.des_z = des_pos[2]

        self.start_x_vel = start_vel[0]
        self.start_y_vel = start_vel[1]
        self.start_z_vel = start_vel[2]

        self.des_x_vel = des_vel[0]
        self.des_y_vel = des_vel[1]
        self.des_z_vel = des_vel[2]

        self.start_x_acc = start_acc[0]
        self.start_y_acc = start_acc[1]
        self.start_z_acc = start_acc[2]

        self.des_x_acc = des_acc[0]
        self.des_y_acc = des_acc[1]
        self.des_z_acc = des_acc[2]

        self.T = T

    def solve(self):
        A = np.array(
            [[0, 0, 0, 0, 0, 1],
             [self.T**5, self.T**4, self.T**3, self.T**2, self.T, 1],
             [0, 0, 0, 0, 1, 0],
             [5*self.T**4, 4*self.T**3, 3*self.T**2, 2*self.T, 1, 0],
             [0, 0, 0, 2, 0, 0],
             [20*self.T**3, 12*self.T**2, 6*self.T, 2, 0, 0]
            ])

        b_x = np.array(
            [[self.start_x],
             [self.des_x],
             [self.start_x_vel],
             [self.des_x_vel],
             [self.start_x_acc],
             [self.des_x_acc]
            ])

        b_y = np.array(
            [[self.start_y],
             [self.des_y],
             [self.start_y_vel],
             [self.des_y_vel],
             [self.start_y_acc],
             [self.des_y_acc]
            ])

        b_z = np.array(
            [[self.start_z],
             [self.des_z],
             [self.start_z_vel],
             [self.des_z_vel],
             [self.start_z_acc],
             [self.des_z_acc]
            ])

        self.x_c = np.linalg.solve(A, b_x)
        self.y_c = np.linalg.solve(A, b_y)
        self.z_c = np.linalg.solve(A, b_z)


class Quadrotor():
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
            fig.tight_layout()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

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
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll), z]
             ])

    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        plt.cla()

    # Plot the quadrotor
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                 [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                 [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')

        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                 [p1_t[2], p2_t[2]], 'r-')
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                 [p3_t[2], p4_t[2]], 'r-')

        self.ax.plot(self.x_data, self.y_data, self.z_data, 'g:',linewidth=3)

    # Plot contour
        self.plot_contour(contourPoints,randomHenonPoints,building_faces)

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        self.ax.set_zlim(-0.5,11)
     
        plt.pause(0.001)
    def plot_contour(self, contour_points,random_HenonPoints,building_faces):
        # Unpack contour points
        # Unpack contour points
        x_contour, y_contour, z_contour = contour_points[:,0], contour_points[:,1], contour_points[:,2]

        x_random, y_random, z_random = random_HenonPoints[:,0], random_HenonPoints[:,1],random_HenonPoints[:,2]

        # Plot the contour
        # Plot the contour
        self.ax.plot(x_contour, y_contour, z_contour, 'b-', linewidth=2, label='Contour')
        # Plot the building
        building_poly = Poly3DCollection(building_faces, facecolors='gray', linewidths=1, edgecolors='lightgrey', alpha=0.35)
        self.ax.add_collection3d(building_poly)
        self.ax.plot([],[], 'g--', label='Quadrotor Trajectory')
        #self.ax.scatter(x_random, y_random, z_random, color='r',s=10)
        self.ax.plot(x_random, y_random, z_random, 'r-',linewidth=0.6, label='trajectory')
        self.ax.grid(False)
        self.ax.tick_params(axis='both', which='major', labelsize=19)
        # Optionally, you can add a legend and adjust limits if needed
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=True,fontsize="14",scatterpoints=1, markerscale=2, handletextpad=0.5,)
show_animation = True

# Simulation parameters
g = 9.81
m = 0.2
Ixx = 1
Iyy = 1
Izz = 1
T = 60

# Proportional coefficients
Kp_x = 2
Kp_y = 2
Kp_z = 2
Kp_roll = 40
Kp_pitch = 40
Kp_yaw = 40


# Derivative coefficients
Kd_x = 20
Kd_y = 20
Kd_z = 10


def quad_sim(x_coeffs, y_coeffs, z_coeffs):
    """
    Simulates the quadrotor's motion given the trajectory coefficients
    for x, y, and z axes.
    """
    x_pos = randomHenonPoints[0,0]
    y_pos = randomHenonPoints[0,1]
    z_pos = randomHenonPoints[0,2]
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
    t = 0

    q = Quadrotor(x=x_pos, y=y_pos, z=z_pos, roll=roll,
                  pitch=pitch, yaw=yaw, size=1, show_animation=show_animation)

    i = 0
    n_run = len(x_coeffs)
    irun = 0

    while True:
        while t <= T:
            des_x_pos = calculate_position(x_coeffs[i], t)
            des_y_pos = calculate_position(y_coeffs[i], t)
            des_z_pos = calculate_position(z_coeffs[i], t)
            des_x_vel = calculate_velocity(x_coeffs[i], t)
            des_y_vel = calculate_velocity(y_coeffs[i], t)
            des_z_vel = calculate_velocity(z_coeffs[i], t)
            des_x_acc = calculate_acceleration(x_coeffs[i], t)
            des_y_acc = calculate_acceleration(y_coeffs[i], t)
            des_z_acc = calculate_acceleration(z_coeffs[i], t)

            thrust = m * (g + des_z_acc + Kp_z * (des_z_pos -
                                                  z_pos) + Kd_z * (des_z_vel - z_vel))

            roll_torque = Kp_roll * \
                (((des_x_acc * sin(des_yaw) - des_y_acc * cos(des_yaw)) / g) - roll)
            pitch_torque = Kp_pitch * \
                (((des_x_acc * cos(des_yaw) - des_y_acc * sin(des_yaw)) / g) - pitch)
            yaw_torque = Kp_yaw * (des_yaw - yaw)

            roll_vel += roll_torque * dt / Ixx
            pitch_vel += pitch_torque * dt / Iyy
            yaw_vel += yaw_torque * dt / Izz

            roll += roll_vel * dt
            pitch += pitch_vel * dt
            yaw += yaw_vel * dt

            R = rotation_matrix(roll, pitch, yaw)
            acc = (np.matmul(R, np.array(
                [0, 0, thrust.item()]).T) - np.array([0, 0, m * g]).T) / m
            x_acc = acc[0]
            y_acc = acc[1]
            z_acc = acc[2]
            x_vel += x_acc * dt
            y_vel += y_acc * dt
            z_vel += z_acc * dt
            x_pos += x_vel * dt
            y_pos += y_vel * dt
            z_pos += z_vel * dt

            q.update_pose(x_pos, y_pos, z_pos, roll, pitch, yaw)

            t += dt

        t = 0
        i = (i + 1) % n_run
        irun += 1
        if irun >= n_run:
            break

    print("Done")


def calculate_position(c, t):
    """
    Calculates a position given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the position

    Returns
        Position
    """
    return c[0] * t**5 + c[1] * t**4 + c[2] * t**3 + c[3] * t**2 + c[4] * t + c[5]


def calculate_velocity(c, t):
    """
    Calculates a velocity given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the velocity

    Returns
        Velocity
    """
    return 5 * c[0] * t**4 + 4 * c[1] * t**3 + 3 * c[2] * t**2 + 2 * c[3] * t + c[4]


def calculate_acceleration(c, t):
    """
    Calculates an acceleration given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the acceleration

    Returns
        Acceleration
    """
    return 20 * c[0] * t**3 + 12 * c[1] * t**2 + 6 * c[2] * t + 2 * c[3]


def rotation_matrix(roll, pitch, yaw):
    """
    Calculates the ZYX rotation matrix.

    Args
        Roll: Angular position about the x-axis in radians.
        Pitch: Angular position about the y-axis in radians.
        Yaw: Angular position about the z-axis in radians.

    Returns
        3x3 rotation matrix as NumPy array
    """
    return np.array(
        [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll)],
         [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll)],
         [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw)]
         ])

# Parameters for the circular helix with increasing radius
# Parameters for the circular helix with increasing radius
initial_radius = 6  # Initial radius of the helix
radius_growth = 0.0  # Growth rate of the radius
pitch = 3          # Distance between successive turns

# Define the parameter range
t = np.linspace(0, 6 * np.pi, 200)  # Increased number of points for a smoother curve

# Simulation parameters
steps = 60
step_size = len(t) // steps 

trans_list=[]
# Parametric equations of the circular helix with increasing radius
def xC(t):
    return (initial_radius + radius_growth * t) * np.cos(t)

def yC(t):
    return (initial_radius + radius_growth * t) * np.sin(t)

def zC(t):
    return pitch * t / (2 * np.pi)

# Tangential vector of the circular helix with increasing radius
def tX(t):
    return -(initial_radius + radius_growth * t) * np.sin(t) + radius_growth * np.cos(t) * t

def tY(t):
    return (initial_radius + radius_growth * t) * np.cos(t) + radius_growth * np.sin(t) * t

def tZ(t):
    return pitch / (2 * np.pi)

# Plane parameters
planeLength = 4.7
planeWidth =  2.9

# Henon map parameters

# Henon map parameters
c = 0.56
u = 0.8
k = 1
r = 0.05
d = -0.8
x0 = 0.1
q0 = 0.1
iterations = 100000
index_list = []
rad_list =[]
def ndmh_map(c, u, k, r, d, x0, q0, iterations):
    """
    Simulates the NDMH map.

    Parameters:
    c, u, k, r, d: control parameters of the NDMH map
    x0, q0: initial conditions for x and q
    iterations: number of iterations to simulate

    Returns:
    x, q: arrays containing the simulated values of x and q
    """
    # Initialize arrays
    x = np.zeros(iterations)
    q = np.zeros(iterations)
    
    # Set initial conditions
    x[0] = x0
    q[0] = q0

    # Iterate the map
    for n in range(iterations - 1):
        x[n + 1] = u * x[n] * (q[n]**2 - 1 + c * np.sin(r * n + r)) + d * np.sin(np.pi * x[n])
        q[n + 1] = q[n] + k * x[n]

    return x, q

# Function to plot the plane at each position with normal parallel to tangential vector
def plotPlane(position, tangentialVector, length, width, initialHenonPoint,scale,index):
    # Normalize tangential vector
    tangentialVector = tangentialVector / np.linalg.norm(tangentialVector)
    
   # Define the point (0, 0, |z|) based on the z-component of the position vector
    target_point = np.array([0, 0, abs(position[2])])

# Calculate the vector from the current position to the target point
    referenceVector = target_point - position
    trans_list.append(tangentialVector)
# Normalize the reference vector
    referenceVector = referenceVector / np.linalg.norm(referenceVector)
    rad_list.append(referenceVector)
    # Calculate the normal vector (perpendicular to tangential vector and reference vector)
    normalVector = np.cross(tangentialVector, referenceVector)
    normalVector = normalVector / np.linalg.norm(normalVector)
    
    # Define vertices of the plane based on position and orientation
    vertices = [
        position + (length / 2) * referenceVector + (width / 2) * np.cross(referenceVector, tangentialVector),
        position + (length / 2) * referenceVector - (width / 2) * np.cross(referenceVector, tangentialVector),
        position - (length / 2) * referenceVector - (width / 2) * np.cross(referenceVector, tangentialVector),
        position - (length / 2) * referenceVector + (width / 2) * np.cross(referenceVector, tangentialVector)
    ]
    
    # Generate Henon map data
    x, y = ndmh_map(c, u, k, r, d, x0, q0, iterations)

    # Extract points from iteration 49,500 to 50,000
    start_iter = 49000
    end_iter = 50000
    x_subset = x[start_iter:end_iter]
    y_subset = y[start_iter:end_iter]

    # Create an array of (x, y) points
    henonPoints = np.column_stack((x_subset, y_subset))
    
    # Scale Henon points
    scaledHenonPoints = henonPoints * scale

    index_list.append(index)
    # Ensure the random iteration is within the bounds of henonPoints
    randomIteration = np.random.randint(0, len(scaledHenonPoints))
    randomHenonPoint = scaledHenonPoints[len(index_list)]
    
    # Ensure the random iteration is within the bounds of henonPoints
    #randomIteration = np.random.randint(0, len(scaledHenonPoints))
    #randomHenonPoint = henonPoints[randomIteration]
    
    # Transform Henon points to the plane's local coordinate system
    # Transform Henon points to the plane's local coordinate system
    localHenonPoints = [position + point[0] * referenceVector + point[1] * np.cross(referenceVector, tangentialVector) for point in scaledHenonPoints]
    localRandomHenonPoint = position + randomHenonPoint[0] * referenceVector + randomHenonPoint[1] * np.cross(referenceVector, tangentialVector)
    
    return vertices, localHenonPoints, localRandomHenonPoint, target_point

# Generate points for the contour and tangential vectors
contourPoints = np.array([[xC(ti), yC(ti), zC(ti)] for ti in t])
tangentialVectors = np.array([[tX(ti), tY(ti), tZ(ti)] for ti in t])
scale =1
# Generate planes and final Henon points along the contour
planesAndHenonPoints = [plotPlane(contourPoints[i], tangentialVectors[i], planeLength, planeWidth, [0.1, 0.1],scale,i) for i in range(0, len(contourPoints), step_size)]

# Extract planes, Henon points, and random Henon points separately
planes = [p[0] for p in planesAndHenonPoints]
henonPoints = [p[1] for p in planesAndHenonPoints]
randomHenonPoints = np.array([p[2] for p in planesAndHenonPoints])
tp = [p[3] for p in planesAndHenonPoints]


# Calculate the intersection points with the contour
intersection_points = np.array([np.mean(np.array(plane), axis=0) for plane in planes])


# Calculate the z0 unit vector
first_point = intersection_points[0]
z0_vector = first_point / np.linalg.norm(first_point)
z0_vectors = z0_vector
# Calculate zn unit vectors and check the cross product with z0
zn_vectors = trans_list
yn_vectors =[]
xn_vectors =rad_list
for i in range(len(intersection_points)):
    
    point_n = intersection_points[i]
    point_np1 = intersection_points[(i + 1) % len(intersection_points)]
    zn = zn_vectors[i]
    xn = xn_vectors[i]
    # Define the threshold
    angle_threshold = np.pi / 18  # pi/180 radians

    # Check if the magnitude of the cross product is less than the threshold
    if np.linalg.norm(np.cross(z0_vector, zn)) < angle_threshold:
        if i > 0:
           z0_vector = zn_vectors[-1]  # Use the previous zn if cross product is zero # Use the previous zn if cross product is zero
            
          
    #zn_vectors.append(zn)
    # Calculate yn unit vector
    yn = np.cross( zn,xn)
    yn = yn / np.linalg.norm(yn)
    yn_vectors.append(yn)
    #xn = np.cross(yn,zn)
    #xn = xn / np.linalg.norm(xn)
    #xn_vectors.append(xn)
yn_vectors = np.array(yn_vectors)
zn_vectors = np.array(zn_vectors)
xn_vectors = np.array(xn_vectors)

# Define the building dimensions and vertices
building_height = 11
building_base_center = np.array([0, 0, 0])
building_size = 6.8

building_vertices = np.array([
    [building_base_center[0] - building_size / 2, building_base_center[1] - building_size / 2, building_base_center[2]],
    [building_base_center[0] + building_size / 2, building_base_center[1] - building_size / 2, building_base_center[2]],
    [building_base_center[0] + building_size / 2, building_base_center[1] + building_size / 2, building_base_center[2]],
    [building_base_center[0] - building_size / 2, building_base_center[1] + building_size / 2, building_base_center[2]],
    [building_base_center[0] - building_size / 2, building_base_center[1] - building_size / 2, building_base_center[2] + building_height],
    [building_base_center[0] + building_size / 2, building_base_center[1] - building_size / 2, building_base_center[2] + building_height],
    [building_base_center[0] + building_size / 2, building_base_center[1] + building_size / 2, building_base_center[2] + building_height],
    [building_base_center[0] - building_size / 2, building_base_center[1] + building_size / 2, building_base_center[2] + building_height]
])

building_faces = [
    [building_vertices[0], building_vertices[1], building_vertices[5], building_vertices[4]],
    [building_vertices[1], building_vertices[2], building_vertices[6], building_vertices[5]],
    [building_vertices[2], building_vertices[3], building_vertices[7], building_vertices[6]],
    [building_vertices[3], building_vertices[0], building_vertices[4], building_vertices[7]],
    [building_vertices[4], building_vertices[5], building_vertices[6], building_vertices[7]],
    [building_vertices[0], building_vertices[1], building_vertices[2], building_vertices[3]]
]





def main():
    """
    Calculates the x, y, z coefficients for the trajectory
    using the specified waypoints.
    """
    x_coeffs = []
    y_coeffs = []
    z_coeffs = []

    # Initial waypoints
    initial_waypoints = [[-5, -5, 5], [5, 0, 5]]
    
    
    # Combine initial waypoints with circular waypoints
    waypoints = randomHenonPoints

    for i in range(len(waypoints) - 1):
        traj = TrajectoryGenerator(waypoints[i], waypoints[i + 1], T)
        traj.solve()
        x_coeffs.append(traj.x_c)
        y_coeffs.append(traj.y_c)
        z_coeffs.append(traj.z_c)

    quad_sim(x_coeffs, y_coeffs, z_coeffs)


if __name__ == "__main__":
    main()

