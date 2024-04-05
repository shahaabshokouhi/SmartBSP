import numpy as np
import matplotlib.pyplot as plt
import torch
from network import ConvNet
from bspfunctions import SmartPSB
from utils import *
from parameters import *
from environment import Environment
from rangesensor import RangeSensor, RangeSensorPolar
from differential_drive_robot import Robot


# Generate a point cloud for obstacles
grid_size = 5
np.random.seed(27)
# centers = np.array([[0, -25]])
centers = np.random.uniform(-100, 100, (400, 2))
size = 4  # Use the same size for all squares or use size = [5, 7] to specify different sizes for each square
points_per_edge = 20  # Number of points per edge
env = Environment(centers, size, points_per_edge)
point_cloud = env.generate_multiple_squares()

# Initial robot state [x, y, theta]
state = np.array([0, 0, np.pi / 2])

# Parameter Initialization
steps = 1000
left_wheel_velocity = 0.95
right_wheel_velocity = 1
dt = 10
length = 5
width = 5
render = False
first_render = True
failure = False

# Creating path planner object
path_planner = SmartPSB(num_y=grid_size)

# Define parameters for the circle slice
theta1, theta2 = 0, 100  # Degrees
radius = 3
num_slices_radial = 6
num_slices_angular = 5
rotation_angle = -theta2/2

grid_centers, grid_centers_polar = path_planner.calculate_grid_centers(radius, theta1, theta2, num_slices_radial, num_slices_angular, rotation_angle)
final_target = np.array([100, 100], dtype=np.float32)


# Simulate the robot's movement for a given number of steps.
robot = Robot(state)
obstacle_to_grid = RangeSensorPolar(grid_size, point_cloud, radius, theta1, theta2, num_slices_radial, num_slices_angular, rotation_angle)
# Preload all networks
actors = {
    1: ConvNet(grid_size=grid_size),
    2: ConvNet(grid_size=grid_size),
    3: ConvNet(grid_size=grid_size),
    4: ConvNet(grid_size=grid_size),
    5: ConvNet(grid_size=grid_size),
}

# Load state dicts
actors[1].load_state_dict(torch.load('ppo_actor_t1.pth'))
actors[2].load_state_dict(torch.load('ppo_actor_t2.pth'))
actors[3].load_state_dict(torch.load('ppo_actor_t3.pth'))
actors[4].load_state_dict(torch.load('ppo_actor_t4.pth'))
actors[5].load_state_dict(torch.load('ppo_actor_t5.pth'))

for _ in range(steps):


    whichNetwork, _ = obstacle_to_grid.target_normalization(robot.state, final_target, grid_centers[4])

    # Filter points directly in front of the robot
    inertial_points, body_points, body_points_polar = obstacle_to_grid.filter_front_points(robot.state)
    # print("Inertial points: ", inertial_points)
    # print("Body points: ", body_points)

    # Creating the grid, and the path
    grid = obstacle_to_grid.create_grid(robot.state)
    grid = grid.view(1, grid_size, grid_size)

    # print("extracted grid: ", grid)

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
    p = path_planner.action2point_polar(grid_centers, actions)
    path = path_planner.construct_sp(p)
    obs_col_first = path_planner.obstacle_check_polar(grid[0])
    obs_col = False
    if obs_col_first:
        print("The closest path is obstructed, changing the target temporarily")
        for idx in range(1, grid_size + 1):
            if idx == whichNetwork:
                continue
            actor_network = actors[idx]
            dist_map = actor_network(grid)  # Use the selected network

            dist_map_numpy = dist_map.detach().numpy()
            actions = []
            for i in range(grid_size - 1):
                action = np.argmax(dist_map_numpy[0, :, i + 1], axis=0)
                actions.append(action)
            actions = np.array(actions)
            p = path_planner.action2point_polar(grid_centers, actions)
            path = path_planner.construct_sp(p)
            obs_col = path_planner.obstacle_check_polar(grid[0])

            if not obs_col:
                break
    if obs_col:
        failure = True

    path_global = robot.transform_path_to_global(path)
    robot.getPath(path_global)
    initial_state = robot.state
    trajectory = robot.trackPID(n=10)
    if (obs_col_first and render) or first_render:

        fig, ax = plt.subplots()
        ax = obstacle_to_grid.create_polar_grid(initial_state, ax)
        # Plot everything


        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], c='red', label='Obstacles')
        ax.plot(path_global[:, 0], path_global[:, 1], c='red', label='Local path')
        ax.scatter(final_target[0], final_target[1],s=100, c='green', label='Target')

        if inertial_points.size > 0:
            ax.scatter(inertial_points[:, 0], inertial_points[:, 1], c='yellow', label='Front Points')
        # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', label='Robot Path')
        ax.plot(robot.trajectory[:, 0], robot.trajectory[:, 1], 'b.-', label='Robot Path')

        # Draw the rectangle
        # plt.plot(*zip(*np.append(rectangle_corners, [rectangle_corners[0]], axis=0)), 'g--', label='Viewing Area')

        ax.legend()
        ax.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Differential Wheel Drive Robot Simulation in Point Cloud')
        plt.grid()
        plt.show()
        first_render = False
    distance_check = path_planner.distance(robot.state[0:2], final_target)
    if distance_check < 3:
        print('Arrived at the target')
        break

fig, ax = plt.subplots()
ax = obstacle_to_grid.create_polar_grid(initial_state, ax)
# Plot everything


ax.scatter(point_cloud[:, 0], point_cloud[:, 1], c='red', label='Obstacles')
ax.plot(path_global[:, 0], path_global[:, 1], c='red', label='Local path')
ax.scatter(final_target[0], final_target[1],s=100, c='green', label='Target')

if inertial_points.size > 0:
    ax.scatter(inertial_points[:, 0], inertial_points[:, 1], c='yellow', label='Front Points')
# plt.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', label='Robot Path')
ax.plot(robot.trajectory[:, 0], robot.trajectory[:, 1], 'b.-', label='Robot Path')

# Draw the rectangle
# plt.plot(*zip(*np.append(rectangle_corners, [rectangle_corners[0]], axis=0)), 'g--', label='Viewing Area')

ax.legend()
ax.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Differential Wheel Drive Robot Simulation in Point Cloud')
plt.grid()
plt.show()

if failure:
    print("Path Planner failure reported at some point")

print("Done!")
