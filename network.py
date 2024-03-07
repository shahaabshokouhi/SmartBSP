import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self, grid_size):
        super(ConvNet, self).__init__()
        # out_channels: num of filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=1)
        # self.conv1_5 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.softm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.conv1_5(x)
        # x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        n_cols = x.shape[-1]
        prob_image = []
        for col in range(n_cols):
            col_i = x[..., col]
            col_i = self.softm(col_i).view(-1,5,1)
            prob_image.append(col_i)
        final_image = torch.cat(prob_image, dim=-1)
        return final_image


class ConvVal(nn.Module):
    def __init__(self, grid_size):
        super(ConvVal, self).__init__()
        # out_channels: num of filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flattened_size = grid_size * grid_size
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(-1, self.flattened_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


