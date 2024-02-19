from bspfunctions import SmartPSB
import numpy as np
import matplotlib.pyplot as plt
import random
from network import ConvNet
import torch
import warnings
from datasetgenerator import GridDataset
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')


n = 5  # N by N view of the robot, must be odd
x = np.arange(0, n + 1)
path_planner = SmartPSB()
target = np.array([10, 0])
possible_actions = np.arange(0, 5)
num_samples = 1000
griddataset = GridDataset(n, num_samples)
gridloader = DataLoader(griddataset, batch_size=8, shuffle=True)

num_epochs = 10

for epoch in range(num_epochs):
    for batch_idx, batch_grids in enumerate(gridloader):
        batch_actions = []
        batch_log_probs = []
        batch_rew = []
        ## rollout
        for i in  range(batch_grids.shape[0]):
            grid = batch_grids[0]
            grid_tensor = torch.tensor(grid, dtype=torch.float)
            grid_tensor = grid_tensor.view(-1, 5, 5)
            mapper = ConvNet(grid_size=n)
            dist_map = mapper(grid_tensor)
            dist_map_numpy = dist_map.detach().numpy()
            actions = []
            log_probs = []
            for i in range(n-1):
                action = random.choices(possible_actions, dist_map_numpy[:, i])[0]
                log_prob = np.log(dist_map_numpy[:, i][action])
                actions.append(action)
                log_probs.append(log_prob)
            log_probs = np.sum(log_probs)
            actions = np.array(actions)
            y = path_planner.action2point(actions)
            p = np.column_stack((x, y))
            path = path_planner.construct_sp(p)
            cost = path_planner.calculate_cost(path, target, grid)
            batch_rew.append(cost)
            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
        batch_rew = torch.tensor(batch_rew, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float32)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
        print(f'Batch {batch_idx} done')

# If you want to use DataLoader for batching, etc.
# plt.imshow(grid, cmap='gray')

## network output
grid_tensor = torch.tensor(grid, dtype=torch.float)
grid_tensor = grid_tensor.view(-1, 5, 5)
mapper = ConvNet(grid_size=5)
map = mapper(grid_tensor)
dist_map = map.detach().numpy()
print('dist_map = ', dist_map)
# plt.imshow(dist_map, cmap='gray')

## bspline

actions = []
for i in range(n-1):
    action = random.choices(possible_actions, dist_map[:, i])[0]
    actions.append(action)
actions = np.array(actions)
y = path_planner.action2point(actions)
p = np.column_stack((x, y))
path = path_planner.construct_sp(p)
cost = path_planner.calculate_cost(path, target, grid)
collision = path_planner.obstacle_check(grid)
print("Collision = ", collision)

print(f'Cost: {cost:.2f}')
# print('Actions: ', actions)
# print('Grid: ', grid)


# Plotting the path
plt.plot(path[:, 0], path[:, 1] + 2, 'r-', label='Spline Path')
plt.axis([-0.5, n + 0.5, -n/2 + 2, n/2 + 2])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True)
plt.plot(p[:, 0], p[:, 1] + 2, 'o-', label='Control Points')
plt.legend()
plt.show()
plt.pause(0.01)
plt.close()
plt.clf()  # Clear the plot for the next iteration

