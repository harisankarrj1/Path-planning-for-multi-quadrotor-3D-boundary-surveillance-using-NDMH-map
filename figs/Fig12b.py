import numpy as np
import matplotlib.pyplot as plt

# Define the position, reference, and tangential vectors
position = np.array([3., 0., 0.])
referenceVector = np.array([-1., 0., 0.])
tangentialVector = np.array([0., 1., 0.])

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

def transform_to_global(x, q, position, referenceVector, tangentialVector):
    """
    Transforms x and q values to global coordinates and extracts the z coordinate.

    Parameters:
    x, q: arrays of x and q values
    position, referenceVector, tangentialVector: vectors for transformation

    Returns:
    z_coordinates: array of z coordinates from global points
    """
    z_coordinates = []
    for xi, qi in zip(x, q):
        global_point = position + xi * referenceVector + qi * np.cross(referenceVector, tangentialVector)
        z_coordinates.append(global_point[2])
    
    return np.array(z_coordinates)

def bifurcation_diagram(c, u_min, u_max, k, r, d, x0, q0, iterations, num_points):
    u_values = np.linspace(u_min, u_max, num_points)
    z_values = []
    u_plot_values = []

    for u in u_values:
        x, q = ndmh_map(c, u, k, r, d, x0, q0, iterations)
        z_coordinates = transform_to_global(x, q, position, referenceVector, tangentialVector)
        # Collect the last few z coordinates to plot
        z_values.extend(z_coordinates[-100:])
        u_plot_values.extend([u] * 100)
    
    plt.figure(figsize=(10, 7))
    plt.plot(u_plot_values, z_values, ',k', alpha=0.25)  # Comma marker for a fine plot
    plt.title('Bifurcation Diagram of the NDMH Map (z Coordinate)')
    plt.xlabel('$u$')
    plt.ylabel('$z$')
    plt.grid(True)
    plt.show()

# Parameters for the NDMH map
c = 0.56
k = 1
r = 0.05
d = -0.8
x0 = 0.1
q0 = 0.1
iterations = 5000
num_points = 1000

# Define the range of u to explore
u_min = 0.0
u_max = 1.0

bifurcation_diagram(c, u_min, u_max, k, r, d, x0, q0, iterations, num_points)
