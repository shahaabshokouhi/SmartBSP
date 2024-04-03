from bspfunctions import SmartPSB
import numpy as np
import matplotlib.pyplot as plt
import random
from network import ConvNet, ConvVal
import torch
import warnings
from datasetgenerator_polar import GridDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import time
warnings.filterwarnings('ignore')


n = 5  # N by N view of the robot, must be odd
path_planner = SmartPSB(num_y=n)
target = np.array([10, -10])
possible_actions = np.arange(0, n)
num_samples = 10000
griddataset = GridDataset(n, num_samples, test=False)
gridloader = DataLoader(griddataset, batch_size=8, shuffle=True)
actor = ConvNet(grid_size=n)
actor.load_state_dict(torch.load('ppo_actor_t5.pth'))
batch_rew_history = []
obs_cols = []
render = False

# Define parameters for the circle slice
theta1, theta2 = 0, 100  # Degrees
radius = 3
num_slices_radial = 6
num_slices_angular = 5
rotation_angle = -theta2/2

grid_centers, grid_centers_polar = path_planner.calculate_grid_centers(radius, theta1, theta2, num_slices_radial, num_slices_angular, rotation_angle)

for idx, batch_grids in enumerate(gridloader):
    batch_grids = batch_grids.view(-1, 1, n, n)
    dist_map = actor(batch_grids)
    dist_map_numpy = dist_map.detach().numpy()
    for j in range(dist_map_numpy.shape[0]):
        actions = []
        for i in range(n-1):
            action = np.argmax(dist_map_numpy[j, :, i+1], axis=0)
            actions.append(action)
        actions = np.array(actions)
        p = path_planner.action2point_polar(grid_centers, actions)
        path = path_planner.construct_sp(p)
        obs_col = path_planner.obstacle_check_polar(batch_grids[j, 0])
        obs_cols.append(obs_col)
        if render:
            fig, ax = plt.subplots()
            grid_example_numpy = batch_grids[j].detach().numpy().reshape(n, n)
            obstacles, _ = path_planner.obstacle_from_grid_polar(grid_example_numpy)
            dist_map_numpy_grid = dist_map_numpy[j]
            print(obs_col)
            # Plot grid
            ax.scatter(obstacles[:,0], obstacles[:,1], c='yellow', alpha=1)
            ax = path_planner.create_polar_grid(radius, theta1, theta2, num_slices_radial, num_slices_angular,
                                                rotation_angle, ax)
            # plt.imshow(np.concatenate((np.zeros((n, 1)), dist_map_numpy_grid), axis=1), cmap='gray', extent=[-0.5, n + 0.5, -n/2, n/2])
            ax.plot(path[:, 0], path[:, 1], 'r-', label='Spline Path')  # Adjust path plotting as needed
            ax.plot(p[:, 0], p[:, 1], 'o-', label='Control Points')  # Plot control points
            plt.tight_layout()
            plt.show()
obs_cols = np.array(obs_cols)
num_success = np.where(obs_cols == False)[0]  # Add [0] to access the first (and only) array in the tuple
success_rate = num_success.size / obs_cols.size

print(f'The overall success rate is: {success_rate:.3f}')
