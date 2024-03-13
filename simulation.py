import numpy as np
import matplotlib.pyplot as plt

# Robot parameters
wheel_radius = 0.05  # meters
wheel_base = 0.15  # meters, distance between two wheels

# Generate a point cloud for obstacles
np.random.seed(42)  # For reproducible results
point_cloud = np.random.uniform(-10, 10, (100, 2))  # 100 random points in a 2D space

# Initial robot state [x, y, theta]
state = np.array([0, 0, np.pi / 4])

def update_state(state, left_wheel_velocity, right_wheel_velocity, dt):
    """
    Update the robot's state using differential drive kinematics.
    """
    x, y, theta = state
    v_l, v_r = left_wheel_velocity, right_wheel_velocity

    # Linear and angular velocities
    v = wheel_radius * (v_r + v_l) / 2.0
    omega = wheel_radius * (v_r - v_l) / wheel_base

    # New state
    dx = v * np.cos(theta) * dt
    dy = v * np.sin(theta) * dt
    dtheta = omega * dt

    new_state = np.array([x + dx, y + dy, theta + dtheta])
    return new_state

def simulate_movement(state, steps, left_wheel_velocity, right_wheel_velocity, dt=0.1):
    """
    Simulate the robot's movement for a given number of steps.
    """
    path = [state]
    for _ in range(steps):
        new_state = update_state(state, left_wheel_velocity, right_wheel_velocity, dt)
        # Simple check to avoid moving into obstacles (very rudimentary)
        if not any(np.linalg.norm(new_state[:2] - point) < 0.05 for point in point_cloud):
            state = new_state
        path.append(state)
    return np.array(path)

def filter_front_points(state, point_cloud, length=5.5, width=5):
    """
    Filter points that are within the specified rectangle in front of the robot.
    """
    x, y, theta = state
    rect_points = []
    for point in point_cloud:
        # Transform point to robot coordinates
        px, py = point[0] - x, point[1] - y
        px_rot, py_rot = np.cos(-theta) * px - np.sin(-theta) * py, np.sin(-theta) * px + np.cos(-theta) * py

        # Check if the point is within the rectangle
        if -width / 2 <= px_rot <= width / 2 and 0 <= py_rot <= length:
            rect_points.append(point)

    return np.array(rect_points)

# Simulate robot movement
path = simulate_movement(state, 200, 1.0, 0.5)

# Filter points directly in front of the robot
front_points = filter_front_points(state, point_cloud)

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c='red', label='Obstacles')
plt.scatter(front_points[:, 0], front_points[:, 1], c='yellow', label='Front Points')
plt.plot(path[:, 0], path[:, 1], 'b.-', label='Robot Path')
plt.legend()
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Differential Wheel Drive Robot Simulation in Point Cloud')
plt.show()
