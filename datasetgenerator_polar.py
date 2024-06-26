import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GridDataset(Dataset):
    def __init__(self, n, num_samples, test=False):
        self.num_samples = num_samples
        self.n = n
        self.test = test
        self.data = [self._generate_grid() for _ in range(num_samples)]
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def _generate_grid(self):
        grid = np.ones((self.n, self.n), dtype=np.float32)  # Initialize a 5x5 grid with ones for RGB white cells
        # random_array = np.random.uniform(low=-0.01, high=0.01, size=(5, 5))
        # grid += random_array

        if self.test:
            pass
            # for col in range(self.n-3):
            #     random_row = np.random.randint(0, self.n)
            #     grid[random_row, col + 3] = 0.0  # Set the chosen cell to blue (assuming [0, 0, 1] is blue in your RGB representation)
            # grid[:, 3] = 0.0
            # random_row = np.random.randint(0, self.n)

            # grid[3, 2] = 0.0
            # grid[4, 2] = 0.0
            # grid[2, 2] = 0.0
            # grid[1, 2] = 0.0



            grid[3, 3] = 0.0
            # grid[4, 3] = 0.0
            grid[2, 3] = 0.0
            grid[1, 3] = 0.0


            grid[3, 4] = 0.0
            grid[1, 4] = 0.0
            grid[2, 4] = 0.0
            # random_row = np.random.randint(0, self.n)
            # grid[random_row, 4] = 0.0
            # random_row = np.random.randint(0, 2)
            # grid[random_row, 2] = 0.0

        else:
            for col in range(self.n-2):
                random_row = np.random.randint(0, self.n)
                grid[random_row, col + 2] = 0.0  # Set the chosen cell to blue (assuming [0, 0, 1] is blue in your RGB representation)
        return grid


