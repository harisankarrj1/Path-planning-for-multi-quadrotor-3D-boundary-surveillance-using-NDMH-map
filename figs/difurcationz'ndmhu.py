import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parameters for the circular helix with increasing radius
initial_radius = 3
radius_growth = 0.0
pitch = 0

# Define the parameter range
t = np.linspace(0, 2 * np.pi, 200)

# Simulation parameters
steps = 1
step_size = len(t) // steps

trans_list = []
rad_list = []

def xC(t):
    return (initial_radius + radius_growth * t) * np.cos(t)

def yC(t):
    return (initial_radius + radius_growth * t) * np.sin(t)

def zC(t):
    return pitch * t / (2 * np.pi)

def tX(t):
    return -(initial_radius + radius_growth * t) * np.sin(t) + radius_growth * np.cos(t) * t

def tY(t):
    return (initial_radius + radius_growth * t) * np.cos(t) + radius_growth * np.sin(t) * t

def tZ(t):
    return pitch / (2 * np.pi)

planeLength = 3.5
planeWidth = 2.9

c = 0.56
k = 1
r = 0.05
d = -0.8
x0 = 0.1
q0 = 0.1
iterations = 1000

def ndmh_map(c, u, k, r, d, x0, q0, iterations):
    x = np.zeros(iterations)
    q = np.zeros(iterations)
    x[0] = x0
    q[0] = q0

    for n in range(iterations - 1):
        x[n + 1] = u * x[n] * (q[n]**2 - 1 + c * np.sin(r * n + r)) + d * np.sin(np.pi * x[n])
        q[n + 1] = q[n] + k * x[n]

        if np.isnan(x[n + 1]) or np.isinf(x[n + 1]):
            print(f"NaN or Inf encountered in x at iteration {n+1}: x = {x[n+1]}, q = {q[n]}")
        if np.isnan(q[n + 1]) or np.isinf(q[n + 1]):
            print(f"NaN or Inf encountered in q at iteration {n+1}: x = {x[n]}, q = {q[n]}")
    
    return x, q

def plotPlane(position, tangentialVector, length, width, scale, index, u):
    tangentialVector = tangentialVector / np.linalg.norm(tangentialVector) if np.linalg.norm(tangentialVector) != 0 else tangentialVector
    target_point = np.array([0, 0, abs(position[2])])
    referenceVector = target_point - position
    trans_list.append(tangentialVector)
    referenceVector_norm = np.linalg.norm(referenceVector)
    if referenceVector_norm != 0:
        referenceVector = referenceVector / referenceVector_norm
    else:
        referenceVector = np.array([1, 0, 0])
    
    rad_list.append(referenceVector)
    normalVector = np.cross(tangentialVector, referenceVector)
    normalVector_norm = np.linalg.norm(normalVector)
    normalVector = normalVector / normalVector_norm if normalVector_norm != 0 else normalVector
    
    vertices = [
        position + (length / 2) * referenceVector + (width / 2) * np.cross(referenceVector, tangentialVector),
        position + (length / 2) * referenceVector - (width / 2) * np.cross(referenceVector, tangentialVector),
        position - (length / 2) * referenceVector - (width / 2) * np.cross(referenceVector, tangentialVector),
        position - (length / 2) * referenceVector + (width / 2) * np.cross(referenceVector, tangentialVector)
    ]

    x, y = ndmh_map(c, u, k, r, d, x0, q0, iterations)
    start_iter = 800
    end_iter = 1000
    x_subset = x[start_iter:end_iter]
    y_subset = y[start_iter:end_iter]

    henonPoints = np.column_stack((x_subset, y_subset))
    scaledHenonPoints = henonPoints * scale

    localHenonPoints = []
    for point in scaledHenonPoints:
        local_point = position + point[0] * referenceVector + point[1] * np.cross(referenceVector, tangentialVector)
        if not np.any(np.isnan(local_point)) and not np.any(np.isinf(local_point)):
            localHenonPoints.append(local_point)
    
    return vertices, localHenonPoints, target_point

contourPoints = np.array([[xC(ti), yC(ti), zC(ti)] for ti in t])
tangentialVectors = np.array([[tX(ti), tY(ti), tZ(ti)] for ti in t])
scale = 1

def bifurcation_diagram(c, u_min, u_max, k, r, d, x0, q0, iterations, num_points):
    u_values = np.linspace(u_min, u_max, num_points)
    x_values = []
    u_plot_values = []

    for u in u_values:
        planesAndHenonPoints = [plotPlane(contourPoints[i], tangentialVectors[i], planeLength, planeWidth, scale, i, u) for i in range(0, len(contourPoints), step_size)]
        henonPoints = [p[1] for p in planesAndHenonPoints]
        z_coordinates = []

        for points in henonPoints:
            z_coords_for_plane = [point[2] for point in points]
            z_coordinates.extend(z_coords_for_plane)

        x_values.extend(z_coordinates[-100:])
        u_plot_values.extend([u] * 100)

    plt.figure(figsize=(10, 7))
    plt.plot(u_plot_values, x_values, ',k', markersize=1, alpha=0.75)
    plt.xlabel('$u$', fontsize="25")
    plt.ylabel('$z\'$', fontsize="25")
    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=15)
    
    plt.show()

num_points = 1000
u_min = 0.0
u_max = 1.0

bifurcation_diagram(c, u_min, u_max, k, r, d, x0, q0, iterations, num_points)
