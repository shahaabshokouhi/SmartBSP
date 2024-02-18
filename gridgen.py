import matplotlib.pyplot as plt
import numpy as np
import random
from network import ConvNet
import torch
import warnings
warnings.filterwarnings('ignore')

# Create a 5x5 grid of zeros (white cells)
grid = np.ones((5, 5))

# For each column, choose a random row to be blue
for col in range(3):
    random_row = random.randint(0, 4)
    grid[random_row, col + 2] = 0  # RGB for blue

# Plotting the grid
plt.imshow(grid, cmap='gray')
plt.xticks(range(5))
plt.yticks(range(5))
# plt.grid(which='both', color='black', linewidth=2)
plt.show()

grid = torch.tensor(grid, dtype=torch.float)
grid = grid.view(-1, 5, 5)
mapper = ConvNet(grid_size=5)
map = mapper(grid)
dist_map = map.detach().numpy()

# Plotting the result
plt.imshow(dist_map, cmap='gray')
plt.xticks(range(5))
plt.yticks(range(5))
# plt.grid(which='both', color='black', linewidth=2)
plt.show()
