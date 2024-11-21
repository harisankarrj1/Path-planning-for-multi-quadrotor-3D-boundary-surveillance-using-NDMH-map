import numpy as np
import matplotlib.pyplot as plt

def henon_map(a, b, x0, y0, iterations, transient=100):
    """
    Simulates the Henon map.

    Parameters:
    a, b: control parameters of the Henon map
    x0, y0: initial conditions
    iterations: number of iterations to simulate
    transient: number of initial iterations to discard (transient behavior)

    Returns:
    x, y: arrays containing the simulated values of x and y
    """
    x = np.zeros(iterations)
    y = np.zeros(iterations)
    
    x[0] = x0
    y[0] = y0
    
    for n in range(iterations - 1):
        x[n + 1] = 1 - a * x[n]**2 + y[n]
        y[n + 1] = b * x[n]
    
    return x[transient:], y[transient:]

def bifurcation_diagram(a_min, a_max, b, x0, y0, iterations, num_points):
    a_values = np.linspace(a_min, a_max, num_points)
    x_values = []
    a_plot_values = []

    for a in a_values:
        x, y = henon_map(a, b, x0, y0, iterations)
        # Collect the last few points to plot
        x_values.extend(x[-100:])
        a_plot_values.extend([a] * 100)
    
    plt.figure(figsize=(10, 7))
    plt.plot(a_plot_values, x_values, ',k', alpha=0.25)  # Comma marker for a fine plot
    plt.title('Bifurcation diagram of the Henon map')
    plt.xlabel('a')
    plt.ylabel('x')
    plt.grid(True)
    plt.show()

# Parameters for the Henon map
a_min = 1.0
a_max = 1.4
b = 0.3
x0 = 0.1
y0 = 0.1
iterations = 30000
num_points = 1000

bifurcation_diagram(a_min, a_max, b, x0, y0, iterations, num_points)
