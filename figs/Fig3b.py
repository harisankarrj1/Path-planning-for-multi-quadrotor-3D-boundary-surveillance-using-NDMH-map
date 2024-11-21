import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev

# Parameters for the elliptical contour
a = 5  # Semi-major axis along x-axis
b = 5  # Semi-minor axis along y-axis
c = 1  # Amplitude along z-axis

# Define the parameter range
t = np.arange(0, 2 * np.pi, 0.1)

# Simulation parameters
steps = 20
step_size = len(t) // steps

# Parametric equations of the contour
def xC(t):
    return a * np.cos(t)

def yC(t):
    return b * np.sin(t)

def zC(t):
    return c * np.cos(t)

# Tangential vector
def tX(t):
    return -a * np.sin(t)

def tY(t):
    return b * np.cos(t)

def tZ(t):
    return -c * np.sin(t)

# Plane parameters
planeLength = 2.6
planeWidth = 1

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
trans_list =[]
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

# 3D Plot including everything
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot the contour
contour = contourPoints
ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], 'b')


# Plot the Henon points on the first plane
henonPoints1 = np.array(henonPoints)
ax.scatter(henonPoints1[:, 0], henonPoints1[:, 1], henonPoints1[:, 2], color='orange', s=1,alpha=0.3)


# Plot the planes
#for plane in planes:
 #   ax.add_collection3d(Poly3DCollection([plane], color='g', alpha=0.3))

# Plot the Henon points on each plane
#for points in henonPoints:
  #  points = np.array(points)
  #  ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='orange', s=1)

# Plot the randomly selected Henon points
#ax.scatter(randomHenonPoints[:, 0], randomHenonPoints[:, 1], randomHenonPoints[:, 2], color='r', s=10)
# Calculate the length of first_point
length = np.linalg.norm(first_point)


unit_length = 0.5
# Scale the unit vector by the length of first_point
scaled_vector = z0_vectors * length
# Plot the z0 unit vector
#ax.quiver(0, 0, 0, z0_vectors[0], z0_vectors[1], z0_vectors[2], color='black', normalize=True, label='z0 Unit Vector')
ax.plot([0, scaled_vector[0]], [0, scaled_vector[1]], [0, scaled_vector[2]], linestyle=':', color='black', label='Translational vector')
# Plot the global x, y, z axes
# Plot the global x, y, z axes as small unit vectors
max_range = np.array([contourPoints[:, 0].max(), contourPoints[:, 1].max(), contourPoints[:, 2].max()]).max()
ax.quiver(0, 0, 0, max_range/2, 0, 0, color='purple', arrow_length_ratio=0.1, label='x')
ax.quiver(0, 0, 0, 0, max_range/2, 0, color='green', arrow_length_ratio=0.1, label='y')
ax.quiver(0, 0, 0, 0, 0, unit_length, color='red', arrow_length_ratio=0.1, label='z')

# Plot the first zn unit vector
point = intersection_points[0]
zn = zn_vectors[0]
ax.quiver(point[0], point[1], point[2], zn[0], zn[1], zn[2], color='red', length=0.7, normalize=True,label= "z'")

# Plot the first yn unit vector
yn = yn_vectors[0]
ax.quiver(point[0], point[1], point[2], yn[0], yn[1], yn[2], color='green', length=0.3, normalize=True,label= "y'")

# Plot the first xn unit vector
xn = xn_vectors[0]
ax.quiver(point[0], point[1], point[2], xn[0], xn[1], xn[2], color='purple', length=0.7, normalize=True,label= "x'")

#ax.quiver(0, 0, 0, z0_vectors[0], z0_vectors[1], z0_vectors[2], color='black', normalize=True, label='z0 Unit Vector')
# Plot the smooth curve through the random Henon points
#ax.plot(out[0], out[1], out[2], color='purple', linewidth=2)
ax.legend(loc='upper center', bbox_to_anchor=(0.75, 0.9), ncol=2, frameon=True,fontsize="10")

# Set labels and view
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.set_title('Elliptical Contour with Tangent Planes and Smooth Chaotic Trajectory')

ax.view_init(elev=155., azim=6)
ax.grid(False)
# Set plot limits (optional, adjust as needed)
ax.set_zlim([-1.5, 1.5])

plt.show()