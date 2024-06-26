import numpy as np
import matplotlib.pyplot as plt
import torch
from network import ConvNet
from bspfunctions import SmartPSB
from utils import *
from parameters import *
from environment import Environment
from rangesensor import RangeSensor
from differential_drive_robot import Robot


# Generate a point cloud for obstacles
grid_size = 5
np.random.seed(8)
centers = np.random.uniform(-100, 100, (400, 2))
size = 2  # Use the same size for all squares or use size = [5, 7] to specify different sizes for each square
points_per_edge = 25  # Number of points per edge
env = Environment(centers, size, points_per_edge)
point_cloud = env.generate_multiple_squares()

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
final_target = np.array([100, 25], dtype=np.float32)

path_planner = SmartPSB(num_y=grid_size)

# Simulate the robot's movement for a given number of steps.
# states = [state]
robot = Robot(state)
obstacle_to_grid = RangeSensor(point_cloud, grid_size=(grid_size, grid_size), area_size=(length, width))
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
    obs_col = path_planner.obstacle_check(grid[0])
    print(obs_col)
    path_global = robot.transform_path_to_global(path)
    robot.getPath(path_global)
    trajectory = robot.trackPID(n=70)
    # Plot everything
    plt.figure(figsize=(8, 8))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c='red', label='Obstacles')
    plt.plot(path_global[:, 0], path_global[:, 1], c='red', label='Local path')
    plt.scatter(final_target[0], final_target[1],s=100, c='green', label='Target')

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
    obs_col = path_planner.obstacle_check(grid[0])

