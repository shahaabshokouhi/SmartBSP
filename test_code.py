
import numpy as np

# Placeholder definitions
# Assume these are the polar coordinates defining the grid cell boundaries
r1, r2 = 1, 2  # Radial distances
theta1, theta2 = np.radians(0), np.radians(10)  # Angular boundaries, converted to radians

# Example path - replace this with your actual curve points
path = np.array([
    [1.5, 0.5],  # A point that would be inside the cell
    [2.5, 0.5],  # A point outside due to radial distance
    [1.5, 1.5],  # A point outside due to angle
])

# Convert each point to polar coordinates to check if it falls within the grid cell
def is_point_in_cell(x, y, r1, r2, theta1, theta2):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # Adjust theta for negative angles
    if theta < 0:
        theta += 2*np.pi
    return r1 <= r <= r2 and theta1 <= theta <= theta2

# Check each point
inside_points = []
for x, y in path:
    if is_point_in_cell(x, y, r1, r2, theta1, theta2):
        inside_points.append((x, y))

print(inside_points)
