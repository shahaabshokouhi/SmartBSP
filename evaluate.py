from bspfunctions import SmartPSB
import numpy as np
import matplotlib.pyplot as plt
import random
from network import ConvNet, ConvVal
import torch
import warnings
from datasetgenerator import GridDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import time
warnings.filterwarnings('ignore')


n = 5  # N by N view of the robot, must be odd
x = np.arange(0, n + 1)
path_planner = SmartPSB()
target = np.array([10, -10])
possible_actions = np.arange(0, 5)
num_samples = 10000
griddataset = GridDataset(n, num_samples, test=True)
gridloader = DataLoader(griddataset, batch_size=8, shuffle=True)
actor = ConvNet(grid_size=n)
actor.load_state_dict(torch.load('ppo_actor.pth'))
batch_rew_history = []
obs_cols = []
render = True

for idx, batch_grids in enumerate(gridloader):
    batch_grids = batch_grids.view(-1, 1, 5, 5)
    dist_map = actor(batch_grids)
    dist_map_numpy = dist_map.detach().numpy()
    for j in range(dist_map_numpy.shape[0]):
        actions = []
        for i in range(n-1):
            action = np.argmax(dist_map_numpy[j, :, i+1], axis=0)
            actions.append(action)
        actions = np.array(actions)
        y = path_planner.action2point(actions)
        p = np.column_stack((x, y))
        path = path_planner.construct_sp(p)
        obs_col = path_planner.obstacle_check(batch_grids[j, 0])
        obs_cols.append(obs_col)
        if obs_col and render:
            grid_example_numpy = batch_grids[j].detach().numpy().reshape(5, 5)
            obstacles = path_planner.obstacle_from_grid(grid_example_numpy)
            dist_map_numpy_grid = dist_map_numpy[j]

            # Plot grid
            plt.scatter(obstacles[:,0], obstacles[:,1] + 2, c='yellow', alpha=1)
            plt.imshow(np.concatenate((np.zeros((5, 1)), dist_map_numpy_grid), axis=1), cmap='gray', extent=[-0.5, 5.5, -0.5, 4.5])
            plt.plot(path[:, 0], path[:, 1] + 2, 'r-', label='Spline Path')  # Adjust path plotting as needed
            plt.plot(p[:, 0], p[:, 1] + 2, 'o-', label='Control Points')  # Plot control points
            plt.show()
obs_cols = np.array(obs_cols)
num_success = np.where(obs_cols == False)[0]  # Add [0] to access the first (and only) array in the tuple
success_rate = num_success.size / obs_cols.size

print(f'The overall success rate is: {success_rate:.3f}')
