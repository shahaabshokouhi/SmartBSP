import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches




# Generate a point cloud for obstacles
grid_size = 5
np.random.seed(42)  # For reproducible results
point_cloud = np.random.uniform(-10, 10, (50, 2))  # 100 random points in a 2D space

# Initial robot state [x, y, theta]
state = np.array([0, 0, np.pi / 2])

steps = 200
left_wheel_velocity = 0.95
right_wheel_velocity = 1
dt = 10
length = 5
width = 5

class Robot:
    def __init__(self, initial_state, wheel_radius=0.05, wheel_base=0.15):
        self.state = np.array(initial_state)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base

    def update_state(self, left_wheel_velocity, right_wheel_velocity, dt):
        x, y, theta = self.state
        v_l, v_r = left_wheel_velocity, right_wheel_velocity
        v = self.wheel_radius * (v_r + v_l) / 2.0
        omega = self.wheel_radius * (v_r - v_l) / self.wheel_base
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = omega * dt
        self.state = np.array([x + dx, y + dy, theta + dtheta])
        return self.state

class ObstacleGrid:
    def __init__(self, point_cloud, grid_size=(5, 5), area_size=(5, 5)):
        self.point_cloud = point_cloud
        self.grid_size = grid_size
        self.area_size = area_size
        self.width = area_size[1]
        self.length = area_size[0]

    def filter_front_points(self, robot_state):
        x, y, theta = robot_state
        inertial_points = []
        body_points = []

        for point in self.point_cloud:
            px, py = point[0] - x, point[1] - y
            px_rot, py_rot = np.cos(-theta) * px - np.sin(-theta) * py, np.sin(-theta) * px + np.cos(-theta) * py
            if 0.5 <= px_rot <= (self.area_size[0] + 0.5) and -self.area_size[1] / 2 <= py_rot <= self.area_size[1] / 2:
                inertial_points.append(point)
                body_points.append([px_rot, py_rot])

        return np.array(inertial_points), np.array(body_points)

    def create_grid(self, robot_state):
        grid = np.ones(self.grid_size)
        _, body_points = self.filter_front_points(robot_state)
        cell_length = self.area_size[0] / self.grid_size[0]
        cell_width = self.area_size[1] / self.grid_size[1]
        x, y, theta = robot_state
        for point in body_points:
            px_rot, py_rot = point[0], point[1]
            cell_x = int((px_rot - 0.5) // cell_width)
            cell_y = int((self.area_size[0] / 2 - py_rot) // cell_length)
            grid[cell_y, cell_x] = 0
        return grid

    def get_rectangle_corners(self, robot_state):
        """
        Get the corners of the rectangle in front of the robot in the global frame.
        """
        x, y, theta = robot_state
        # Corners in robot coordinate frame
        corners_local = np.array([
            [0.5, -self.width / 2],  # Bottom left
            [self.length + 0.5, -self.width / 2],  # Bottom right
            [self.length + 0.5, self.width / 2],  # Top right
            [0.5, self.width / 2]  # Top left
        ])

        # Rotate and translate corners to global frame
        corners_global = []
        for corner in corners_local:
            px_rot, py_rot = (
                np.cos(theta) * corner[0] - np.sin(theta) * corner[1] + x,
                np.sin(theta) * corner[0] + np.cos(theta) * corner[1] + y
            )
            corners_global.append([px_rot, py_rot])

        return np.array(corners_global)


def filter_front_points(state, point_cloud, length=length, width=width):
    """
    Filter points that are within the specified rectangle in front of the robot.
    """
    x, y, theta = state
    inertial_points = []
    body_points = []
    for point in point_cloud:
        # Transform point to robot coordinates
        px, py = point[0] - x, point[1] - y
        px_rot, py_rot = np.cos(-theta) * px - np.sin(-theta) * py, np.sin(-theta) * px + np.cos(-theta) * py

        # Check if the point is within the rectangle
        if 0.5 <= px_rot <= (length + 0.5) and -width / 2 <= py_rot <= width / 2:
            inertial_points.append(point)
            body_points.append([px_rot, py_rot])

    return np.array(inertial_points), np.array(body_points)

# Simulate robot movement
# path = simulate_movement(state, 200, 1.0, 0.5)

def create_obstacle_grid(state, point_cloud, grid_size=(5, 5), area_size=(5, 5)):
    """
    Create a 5x5 grid of the area in front of the robot, marking cells with obstacles as 0 and free cells as 1.

    Parameters:
    - state: The current state of the robot (x, y, theta).
    - point_cloud: The set of obstacle points.
    - grid_size: The dimensions of the grid (rows, columns).
    - area_size: The size of the area to cover with the grid (length, width).

    Returns:
    - A 5x5 numpy array representing the grid, with 0s for obstacles and 1s for free space.
    """
    x, y, theta = state
    grid = np.ones(grid_size)  # Initialize grid as all free space

    # Calculate the dimensions of each grid cell
    cell_length = area_size[0] / grid_size[0]
    cell_width = area_size[1] / grid_size[1]

    # Filter points within the rectangular area in front of the robot
    inertial_points, body_points = filter_front_points(state, point_cloud, *area_size)

    # Assign values to the grid based on the point cloud
    for point in inertial_points:
        # Transform point to robot coordinates
        px, py = point[0] - x, point[1] - y
        px_rot, py_rot = np.cos(-theta) * px - np.sin(-theta) * py, np.sin(-theta) * px + np.cos(-theta) * py

        # Determine the grid cell for the point
        cell_x = int((px_rot - 0.5) // cell_width)
        cell_y = int((2.5 - py_rot) // cell_length)

        # Check boundaries and assign value
        # if 0 <= cell_x < grid_size[1] and 0 <= cell_y < grid_size[0]:
        grid[cell_y, cell_x] = 0  # Mark as obstacle

    return grid


# Simulate the robot's movement for a given number of steps.
path = [state]
robot = Robot(state)
obstacle_to_grid = ObstacleGrid(point_cloud, grid_size=(grid_size, grid_size), area_size=(length, width))

for _ in range(steps):

    # Update the path
    state = robot.update_state(left_wheel_velocity, right_wheel_velocity, dt)
    path.append(state)
    path_np = np.array(path)

    # Filter points directly in front of the robot
    inertial_points, body_points = obstacle_to_grid.filter_front_points(state)
    print("Inertial points: ", inertial_points)
    print("Body points: ", body_points)

    # Get the final position of the robot to draw the rectangle
    rectangle_corners = obstacle_to_grid.get_rectangle_corners(state)

    # creating the grid
    grid = obstacle_to_grid.create_grid(state)
    print("extracted grid: ", grid)

    # Plot everything
    plt.figure(figsize=(8, 8))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c='red', label='Obstacles')
    if inertial_points.size > 0:
        plt.scatter(inertial_points[:, 0], inertial_points[:, 1], c='yellow', label='Front Points')
    plt.plot(path_np[:, 0], path_np[:, 1], 'b.-', label='Robot Path')
    # Draw the rectangle
    plt.plot(*zip(*np.append(rectangle_corners, [rectangle_corners[0]], axis=0)), 'g--', label='Viewing Area')

    plt.legend()
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Differential Wheel Drive Robot Simulation in Point Cloud')
    plt.grid()
    plt.show()


