import numpy as np
import matplotlib.pyplot as plt #in this code i added radis vector properly ie target vector.,zn= tanget vectors ,need to add sequence instead of random ie add index.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parameters for the circular helix with increasing radius
initial_radius = 6  # Initial radius of the helix
radius_growth = 0.0  # Growth rate of the radius
pitch = 3           # Distance between successive turns

# Define the parameter range
t = np.linspace(0, 6 * np.pi, 200)  # Increased number of points for a smoother curve

# Simulation parameters
steps = 10
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
planeLength = 3.5
planeWidth =  2.9

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
# Generate planes and final Henon points along the contour
planesAndHenonPoints2 = [plotPlane(contourPoints[i], tangentialVectors[i], planeLength, planeWidth, [0.11, 0.11],scale,i) for i in range(0, len(contourPoints), step_size)]

# Extract planes, Henon points, and random Henon points separately
planes = [p[0] for p in planesAndHenonPoints]
henonPoints = [p[1] for p in planesAndHenonPoints]
randomHenonPoints = np.array([p[2] for p in planesAndHenonPoints])
tp = [p[3] for p in planesAndHenonPoints]

randomHenonPoints2 = np.array([p[2] for p in planesAndHenonPoints2])
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
# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the contour
contour = contourPoints.T
ax.plot(contour[0], contour[1], contour[2], 'b')

# Plot the planes
#plane=planes[6]
#ax.add_collection3d(Poly3DCollection([plane], color='g', alpha=0.3))

# Plot the Henon points on each plane
#points = henonPoints[6]
#points = np.array(points)
#ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='orange', s=1,label ="Hyperchaotic points")
# Plot the planes
for plane in planes:
    ax.add_collection3d(Poly3DCollection([plane], color='g', alpha=0.2))

# Plot the Henon points on each plane
# Plot the Henon points on each plane
for points in henonPoints:
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='orange', s=1)

# Plot the randomly selected Henon points
ax.scatter(randomHenonPoints[:, 0], randomHenonPoints[:, 1], randomHenonPoints[:, 2], color='purple', s=10,label ="Sequence of NDMH points distributed across planes")

# Plot the z0 unit vector
#ax.quiver(0, 0, 0, z0_vectors[0], z0_vectors[1], z0_vectors[2], color='black', length=np.linalg.norm(first_point)/3, normalize=True, label='z0 Unit Vector')
unit_length = 8

# Plot the global x, y, z axes
max_range = np.array([contourPoints[:, 0].max(), contourPoints[:, 1].max(), contourPoints[:, 2].max()]).max()
ax.quiver(0, 0, 0, max_range/2, 0, 0, color='purple', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, max_range/2, 0, color='green', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, unit_length, color='red', arrow_length_ratio=0.05)

# Plot the first zn unit vector
#point = intersection_points[6]
#zn = zn_vectors[6]
#ax.quiver(point[0], point[1], point[2], zn[0], zn[1], zn[2], color='red', length=2, normalize=True,label= "Tangential unit vector or $\hat{z}_n$")

# Plot the first yn unit vector
#yn = yn_vectors[6]
#ax.quiver(point[0], point[1], point[2], yn[0], yn[1], yn[2], color='green', length=1.5, normalize=True,label= "Normal unit vector or $\hat{y}_n$")

# Plot the first xn unit vector
#xn = xn_vectors[6]
#ax.quiver(point[0], point[1], point[2], xn[0], xn[1], xn[2], color='purple', length=2, normalize=True,label= "$\hat{x}_n$")

# Plot lines connecting the random Henon points
#ax.plot(randomHenonPoints[:, 0], randomHenonPoints[:, 1], randomHenonPoints[:, 2], color='purple', linewidth=2)

# Plot arrows from intersection points to target points
# Plot arrows from intersection point at index 7 to target point at index 7
# Plot arrows from target point to intersection point at index 7
index = 6
intersection = intersection_points[index]
target_point = tp[index]
#ax.quiver(target_point[0], target_point[1], target_point[2],
       # intersection[0] - target_point[0], intersection[1] - target_point[1], intersection[2] - target_point[2],
       # color='brown', arrow_length_ratio=0.05,label="Radius vector")

# Define circle parameters
radius = 6
center_point = target_point # Center of the circle (as target_point[6])

# Generate circle points in the xy plane
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = center_point[0] + radius * np.cos(theta)
circle_y = center_point[1] + radius * np.sin(theta)
circle_z = np.full_like(theta, center_point[2])  # Same z for all points


# Plot the circle
#ax.plot(circle_x, circle_y, circle_z, 'k--', label='Circle of Radius ')


ax.grid(False)

# Set labels and view
ax.set_xlabel(r"$x'$",fontsize ="35")
ax.set_ylabel(r"$y'$",fontsize ="35")
ax.set_zlabel(r"$z'$",fontsize ="35")
ax.set_xlim(-8,8)
ax.set_ylim(-8,8)
ax.set_zlim(-0.5,10)
#ax.legend()
ax.tick_params(axis='both', which='major', labelsize=19)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.89), ncol=2, frameon=True,fontsize="13",scatterpoints=1, markerscale=2, handletextpad=0.5,)
#ax.legend(scatterpoints=1, markerscale=5, handletextpad=0.5, loc='best')

plt.show()

