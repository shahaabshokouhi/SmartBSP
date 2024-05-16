import numpy as np
import matplotlib.pyplot as plt
import torch
from network import ConvNet
from bspfunctions import SmartPSB
from utils import *
from parameters import *

class Environment:
    def __init__(self, centers, size, points_per_edge):
        self.centers = centers
        self.size = size
        self.points_per_edge = points_per_edge

    def generate_square_obstacle(self, center, size, points_per_edge):
        cx, cy = center
        half_size_x = 2
        half_size_y = 10
        # Corners of the square
        corners = np.array([
            [cx - half_size_x, cy - half_size_y],  # Bottom left
            [cx + half_size_x, cy - half_size_y],  # Bottom right
            [cx + half_size_x, cy + half_size_y],  # Top right
            [cx - half_size_x, cy + half_size_y],  # Top left
        ])

        # Generate points along the edges
        edge_points = []
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            edge_points.extend(list(zip(np.linspace(start[0], end[0], points_per_edge, endpoint=False),
                                        np.linspace(start[1], end[1], points_per_edge, endpoint=False))))

        return np.array(edge_points)

    def generate_multiple_squares(self):

        all_points = []
        for i, center in enumerate(self.centers):
            current_size = self.size[i] if isinstance(self.size, list) else self.size
            square_points = self.generate_square_obstacle(center, current_size, self.points_per_edge)
            all_points.append(square_points)
        self.point_cloud = np.vstack(all_points)
        return self.point_cloud
