import numpy as np
import matplotlib.pyplot as plt
import torch
from network import ConvNet
from bspfunctions import SmartPSB
from utils import *
from parameters import *




class Robot:
    def __init__(self, initial_state, robot_path=None, wheel_radius=0.05, wheel_base=0.15,
                 kp_linear=0.3, kd_linear=0.2, ki_linear=0.1,
                 kp_angular=0.3, kd_angular=0.2, ki_angular=0.1):
        self.state = np.array(initial_state)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.path = robot_path
        self.kp_linear = kp_linear
        self.kd_linear = kd_linear
        self.ki_linear = ki_linear

        self.kp_angular = kp_angular
        self.kd_angular = kd_angular
        self.ki_angular = ki_angular

        self.prev_error_position = 0
        self.prev_error_angle = 0

        self.prev_body_to_goal = 0
        self.prev_waypoint_idx = -1
        self.dt = 0.5
        self.trajectory = np.zeros((1, 3))

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

    def transform_path_to_global(self, path):
        x, y, theta = self.state
        # Create a rotation matrix
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

        # Transform each point
        path_global = np.dot(path, rotation_matrix.T) + [x, y]

        return path_global
    def trackPID(self, n = 10):
        trajectory = [self.state]
        for idx in range(n):
            target = self.path[idx + 1, :]
            error_position = get_distance(self.state[0], self.state[1], target[0], target[1])

            body_to_goal = get_angle(self.state[0], self.state[1], target[0], target[1])
            # body_to_nose = get_angle(x[0, 0], x[1, 0], nose[0], nose[1])

            # if self.prev_waypoint_idx == waypoint_idx and 350<(abs(self.prev_body_to_goal - body_to_goal)*180/np.pi):
            # 	print("HERE")
            # 	body_to_goal = self.prev_body_to_goal
            error_angle = body_to_goal - self.state[2]

            linear_velocity_control = self.kp_linear * error_position + self.kd_linear * (
                        error_position - self.prev_error_position)
            angular_velocity_control = self.kp_angular * error_angle + self.kd_angular * (
                        error_angle - self.prev_error_angle)

            self.prev_error_angle = error_angle
            self.prev_error_position = error_position

            self.prev_waypoint_idx = idx
            self.prev_body_to_goal = body_to_goal

            if linear_velocity_control > MAX_LINEAR_VELOCITY:
                linear_velocity_control = MAX_LINEAR_VELOCITY

            right_wheel_velocity, left_wheel_velocity = self.uniToDiff(linear_velocity_control, angular_velocity_control)
            state = robot.update_state(left_wheel_velocity, right_wheel_velocity, self.dt)
            trajectory.append(self.state)
        self.trajectory = np.vstack((self.trajectory, trajectory))
        return  np.array(trajectory)

    def getPath(self, path):
        self.path = path

    def uniToDiff(self, v, w):
        vR = (2 * v + w * self.wheel_base) / (2 * self.wheel_radius)
        vL = (2 * v - w * self.wheel_base) / (2 * self.wheel_radius)
        return vR, vL




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
            if 0.5 <= px_rot <= (self.area_size[0] + 0.49) and (-self.area_size[1] / 2 + 0.01) < py_rot <= self.area_size[1] / 2:
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
        return torch.tensor(grid, dtype=torch.float32)

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
    def distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def target_normalization(self, state, target):
        x, y, theta = state
        min_distance = float('inf')
        y_points = np.arange(-np.floor(self.grid_size[0] / 2), np.floor(self.grid_size[0] / 2) + 1)[::-1]
        x_points = self.grid_size[0] * np.ones_like(y_points)
        points = np.column_stack((x_points, y_points))

        # Rotate points
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        rotated_points = np.dot(points, rotation_matrix.T)  # Transpose to align dimensions
        # Translate points
        global_points = rotated_points + np.array([x, y])
        for idx in range(global_points.shape[0]):
            dist = self.distance(global_points[idx], target)
            if dist < min_distance:
                min_distance = dist
                normalized_target = global_points[idx]
                min_idx = idx + 1

        return min_idx, normalized_target

def generate_square_obstacle(center, size, points_per_edge):
    cx, cy = center
    half_size = size / 2

    # Corners of the square
    corners = np.array([
        [cx - half_size, cy - half_size],  # Bottom left
        [cx + half_size, cy - half_size],  # Bottom right
        [cx + half_size, cy + half_size],  # Top right
        [cx - half_size, cy + half_size],  # Top left
    ])

    # Generate points along the edges
    edge_points = []
    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]
        edge_points.extend(list(zip(np.linspace(start[0], end[0], points_per_edge, endpoint=False),
                                    np.linspace(start[1], end[1], points_per_edge, endpoint=False))))

    return np.array(edge_points)

def generate_multiple_squares(centers, size, points_per_edge):
    """
    Generate point clouds for multiple squares.

    Args:
    centers (list of tuples): List of (x, y) coordinates for the center of each square.
    size (float or list): The length of a side of the squares. If a single float is given,
                          all squares will have the same size. If a list is provided,
                          it must match the length of centers and specify the size for each square.
    points_per_edge (int): The number of points to generate per edge of the square.

    Returns:
    numpy.ndarray: An array of points representing all the obstacles.
    """
    all_points = []
    for i, center in enumerate(centers):
        current_size = size[i] if isinstance(size, list) else size
        square_points = generate_square_obstacle(center, current_size, points_per_edge)
        all_points.append(square_points)

    return np.vstack(all_points)  # Combine all points into one array


# Generate a point cloud for obstacles
grid_size = 5
# np.random.seed(3)  # For reproducible results
centers = np.random.uniform(-100, 100, (200, 2))  # Centers of the squares
size = 1  # Use the same size for all squares
# or use size = [5, 7] to specify different sizes for each square
points_per_edge = 25  # Number of points per edge

# point_cloud1 = generate_square_obstacle(center=center, size=size, points_per_edge=points_per_edge)
# point_cloud2 = generate_square_obstacle(center=(-2, 5), size=size, points_per_edge=points_per_edge)
point_cloud = generate_multiple_squares(centers, size, points_per_edge)
# point_cloud = np.random.uniform(-10, 10, (10, 2))  # 100 random points in a 2D space

# Initial robot state [x, y, theta]
state = np.array([0, 0, np.pi / 2])

# Parameter Initialization
steps = 200
left_wheel_velocity = 0.95
right_wheel_velocity = 1
dt = 10
length = 5
width = 5
x = np.arange(0, grid_size + 1)
final_target = np.array([-100, 100], dtype=np.float32)

path_planner = SmartPSB(num_y=grid_size)

# Simulate the robot's movement for a given number of steps.
# states = [state]
robot = Robot(state)
obstacle_to_grid = ObstacleGrid(point_cloud, grid_size=(grid_size, grid_size), area_size=(length, width))
# Preload all networks
actors = {
    1: ConvNet(grid_size=grid_size),
    2: ConvNet(grid_size=grid_size),
    3: ConvNet(grid_size=grid_size),
    4: ConvNet(grid_size=grid_size),
    5: ConvNet(grid_size=grid_size),
}

# Load state dicts
actors[1].load_state_dict(torch.load('ppo_actor_n5_ep3_10000_t1.pth'))
actors[2].load_state_dict(torch.load('ppo_actor_n5_ep3_10000_t2.pth'))
actors[3].load_state_dict(torch.load('ppo_actor_n5_ep3_10000_t3.pth'))
actors[4].load_state_dict(torch.load('ppo_actor_n5_ep3_10000_t4.pth'))
actors[5].load_state_dict(torch.load('ppo_actor_n5_ep3_10000_t5.pth'))

for _ in range(steps):

    whichNetwork, _ = obstacle_to_grid.target_normalization(robot.state, final_target)

    # Filter points directly in front of the robot
    inertial_points, body_points = obstacle_to_grid.filter_front_points(robot.state)
    # print("Inertial points: ", inertial_points)
    # print("Body points: ", body_points)

    # Get the final position of the robot to draw the rectangle
    rectangle_corners = obstacle_to_grid.get_rectangle_corners(robot.state)

    # creating the grid, and the path
    grid = obstacle_to_grid.create_grid(robot.state)
    grid = grid.view(1, grid_size, grid_size)
    print("extracted grid: ", grid)

    if whichNetwork in actors:
        actor_network = actors[whichNetwork]
        dist_map = actor_network(grid)  # Use the selected network
    else:
        print("Invalid network number. Skipping.")
        continue

    dist_map_numpy = dist_map.detach().numpy()
    actions = []
    for i in range(grid_size - 1):
        action = np.argmax(dist_map_numpy[0, :, i + 1], axis=0)
        actions.append(action)
    actions = np.array(actions)
    y = path_planner.action2point(actions)
    p = np.column_stack((x, y))
    path = path_planner.construct_sp(p)
    obs_col = path_planner.obstacle_check(grid)

    path_global = robot.transform_path_to_global(path)
    robot.getPath(path_global)
    trajectory = robot.trackPID(n=90)
    # Plot everything
    plt.figure(figsize=(8, 8))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c='red', label='Obstacles')
    plt.plot(path_global[:, 0], path_global[:, 1], c='red', label='Local path')
    if inertial_points.size > 0:
        plt.scatter(inertial_points[:, 0], inertial_points[:, 1], c='yellow', label='Front Points')
    # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', label='Robot Path')
    plt.plot(robot.trajectory[:, 0], robot.trajectory[:, 1], 'b.-', label='Robot Path')

    # Draw the rectangle
    plt.plot(*zip(*np.append(rectangle_corners, [rectangle_corners[0]], axis=0)), 'g--', label='Viewing Area')

    plt.legend()
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Differential Wheel Drive Robot Simulation in Point Cloud')
    plt.grid()
    plt.show()


