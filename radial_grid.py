import matplotlib.pyplot as plt
import numpy as np


def calculate_grid_centers(radius, theta1, theta2, num_slices_radial, num_slices_angular, rotation_angle):
    # Function to calculate midpoints in polar coordinates
    def midpoint_polar(r1, r2, theta1, theta2):
        r_mid = (r1 + r2) / 2
        theta_mid = (theta1 + theta2) / 2
        return r_mid, theta_mid

    # Rotate a point in polar coordinates
    def rotate_point_polar(r, theta, angle_deg):
        theta_rad = np.radians(angle_deg)
        theta_rot = theta + theta_rad
        return r, theta_rot

    grid_centers = np.zeros((10,9,2), dtype=np.float32)
    # grid_centers=[]
    radial_intervals = np.linspace(0, radius, num_slices_radial + 1)
    angular_intervals = np.linspace(np.radians(theta1), np.radians(theta2), num_slices_angular + 1)

    # Calculate midpoints for each grid cell
    for i in range(num_slices_radial):
        for j in reversed(range(num_slices_angular)):
            r_mid, theta_mid = midpoint_polar(radial_intervals[i], radial_intervals[i+1],
                                              angular_intervals[j], angular_intervals[j+1])
            # Apply rotation
            r_rot, theta_rot = rotate_point_polar(r_mid, theta_mid, rotation_angle)
            # Convert to Cartesian coordinates
            x_rot = r_rot * np.cos(theta_rot)
            y_rot = r_rot * np.sin(theta_rot)
            grid_centers[i, 8 - j] = [x_rot, y_rot]

    return grid_centers[1:,:,:]

# Define parameters for the circle slice
theta1, theta2 = 0, 180  # Degrees
radius = 3
num_slices_radial = 10
num_slices_angular = 9
rotation_angle = -90
# Generate the outer arc of the circle slice
theta_outer = np.linspace(np.radians(theta1), np.radians(theta2), 100)
x_outer = radius * np.cos(theta_outer)
y_outer = radius * np.sin(theta_outer)

# Initialize plot
# fig, ax = plt.subplots()

# Plot the outer arc
# ax.plot(x_outer, y_outer, 'k')

# # Plot the straight lines from center to the edge of the slice
# for theta in np.linspace(np.radians(theta1), np.radians(theta2), num_slices_angular + 1):
#     x = radius * np.cos(theta)
#     y = radius * np.sin(theta)
#     ax.plot([0, x], [0, y], 'blue')

# Plot the concentric arcs within the slice
# for r in np.linspace(0, radius, num_slices_radial + 1):
#     x_concentric = r * np.cos(theta_outer)
#     y_concentric = r * np.sin(theta_outer)
#     ax.plot(x_concentric, y_concentric, 'red')

# Function to rotate points around the origin
def rotate_points(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot

# Rotate the outer arc
x_outer_rot, y_outer_rot = rotate_points(x_outer, y_outer, rotation_angle)

# Re-initialize plot for rotated slice
fig, ax = plt.subplots()

# Plot the rotated outer arc
ax.plot(x_outer_rot, y_outer_rot, 'k')

# Rotate and plot the straight lines
for theta in np.linspace(np.radians(theta1), np.radians(theta2), num_slices_angular + 1):
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    x_rot, y_rot = rotate_points(x, y, rotation_angle)
    ax.plot([0, x_rot], [0, y_rot], 'blue')

# Rotate and plot the concentric arcs
for r in np.linspace(0, radius, num_slices_radial + 1):
    x_concentric = r * np.cos(theta_outer)
    y_concentric = r * np.sin(theta_outer)
    x_concentric_rot, y_concentric_rot = rotate_points(x_concentric, y_concentric, rotation_angle)
    ax.plot(x_concentric_rot, y_concentric_rot, 'red')
grid_centers = calculate_grid_centers(radius, theta1, theta2, num_slices_radial, num_slices_angular, rotation_angle)
print(grid_centers)
ax.scatter(grid_centers[0,0,0], grid_centers[0,0,1], c='yellow', label='centers')
# Adjust the plot for the rotated slice
ax.set_aspect('equal')
plt.xlim(0, radius+0.1)
plt.ylim(-radius-0.1, radius+0.1)
plt.show()
