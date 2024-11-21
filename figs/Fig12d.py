import numpy as np
import matplotlib.pyplot as plt
#correct
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

def bifurcation_diagram(c, u_min, u_max, k, r, d, x0, q0, iterations, num_points):
    u_values = np.linspace(u_min, u_max, num_points)
    x_values = []
    u_plot_values = []

    for u in u_values:
        x, q = ndmh_map(c, u, k, r, d, x0, q0, iterations)
        # Collect the last few points to plot
        x_values.extend([0.5 * xi for xi in x[-2000:]])

        u_plot_values.extend([u] * 2000)
    
    plt.figure(figsize=(10, 7))
    plt.plot(u_plot_values, x_values, ',k' ,alpha=0.25)  # Comma marker for a fine plot
    #plt.title('Bifurcation diagram of the NDMH map')
    plt.xlabel('$u$',fontsize="30")
    plt.ylabel(r"$z'$", fontsize="30")

    plt.tick_params(axis='both', which='major', labelsize=19)
    plt.grid(False)
    plt.show()

# Parameters for the NDMH map
c = 0.56
k = 1
r = 0.05
d = -0.8
x0 = 0.1
q0 = 0.1
iterations = 25000
num_points = 1000

# Define the range of u to explore
u_min = 0.0
u_max = 1.0

bifurcation_diagram(c, u_min, u_max, k, r, d, x0, q0, iterations, num_points)
