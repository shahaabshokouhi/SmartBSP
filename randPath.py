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
num_samples = 1000
grids = GridDataset(n, num_samples)

# If you want to use DataLoader for batching, etc.
dataloader = DataLoader(grids, batch_size=8, shuffle=True)
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
possible_actions = np.arange(0, 5)
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

