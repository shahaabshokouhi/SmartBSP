import numpy as np
import matplotlib.pyplot as plt
import torch
from network import ConvNet
from bspfunctions import SmartPSB
from utils import *
from parameters import *
class RangeSensorPolar:
    def __init__(self, grid_size, point_cloud, radius, theta1, theta2, num_slices_radial, num_slices_angular, rotation_angle):
        self.grid_size = grid_size
        self.point_cloud = point_cloud
        self.radius = radius
        self.theta1 = theta1
        self.theta2 = theta2
        self.num_slices_radial = num_slices_radial
        self.num_slices_angular = num_slices_angular
        self.rotation_angle = rotation_angle

    def filter_front_points(self, robot_state):
        x, y, theta = robot_state
        inertial_points = []
        body_points = []
        body_points_polar = []

        for point in self.point_cloud:
            px, py = point[0] - x, point[1] - y
            px_rot, py_rot = np.cos(-theta) * px - np.sin(-theta) * py, np.sin(-theta) * px + np.cos(-theta) * py
            r, thetap = self.cartesian_to_polar(px_rot, py_rot)

            if 0 <= r <= (self.radius) and np.radians(self.theta1 + self.rotation_angle) < thetap <= np.radians(self.theta2 + self.rotation_angle):
                inertial_points.append(point)
                body_points.append([px_rot, py_rot])
                body_points_polar.append([r, thetap])

        return np.array(inertial_points), np.array(body_points), np.array(body_points_polar)

    def cartesian_to_polar(self, x, y):

        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        # theta[theta < 0] += 2 * np.pi

        return r, theta

    def create_grid(self, robot_state):
        grid = np.ones((self.grid_size, self.grid_size))
        _, body_points, body_points_polar = self.filter_front_points(robot_state)
        cell_height_theta = np.radians((self.theta2 - self.theta1) / self.num_slices_angular)
        cell_width_r = self.radius / self.num_slices_radial
        x, y, theta = robot_state
        for point in body_points_polar:
            r, theta = point[0], point[1]
            theta = theta - np.radians(self.rotation_angle)
            # theta = np.radians(theta)

            cell_y_theta = self.grid_size - 1 - int(theta // cell_height_theta)
            if cell_y_theta == 5:
                cell_y_theta = 4
            cell_x_r = int((r - 0.5) // cell_width_r)
            if cell_x_r == 5:
                cell_x_r = 4
            grid[cell_y_theta, cell_x_r] = 0
        return torch.tensor(grid, dtype=torch.float32)

    def create_polar_grid(self, state, ax):

        x_robot, y_robot, theta_robot = state

        # Generate the outer arc of the circle slice
        theta_outer = np.linspace(np.radians(self.theta1), np.radians(self.theta2), 100)
        x_outer = self.radius * np.cos(theta_outer)
        y_outer = self.radius * np.sin(theta_outer)
        rotation_angle_rad = np.radians(self.rotation_angle)

        def rotate_points(x, y, angle_rad):
            x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
            return x_rot, y_rot

        # Rotate the outer arc
        x_outer_rot, y_outer_rot = rotate_points(x_outer, y_outer, rotation_angle_rad + theta_robot)
        x_outer_rot += x_robot
        y_outer_rot += y_robot

        # Re-initialize plot for rotated slice
        # fig, ax = plt.subplots()

        # Plot the rotated outer arc
        ax.plot(x_outer_rot, y_outer_rot, 'k')

        # Rotate and plot the straight lines
        for theta in np.linspace(np.radians(self.theta1), np.radians(self.theta2), self.num_slices_angular + 1):
            x, y = self.radius * np.cos(theta), self.radius * np.sin(theta)
            x_rot, y_rot = rotate_points(x, y, rotation_angle_rad + theta_robot)
            x_rot += x_robot
            y_rot += y_robot
            ax.plot([x_robot, x_rot], [y_robot, y_rot], 'blue')

        # Rotate and plot the concentric arcs
        for r in np.linspace(0, self.radius, self.num_slices_radial + 1):
            x_concentric = r * np.cos(theta_outer)
            y_concentric = r * np.sin(theta_outer)
            x_concentric_rot, y_concentric_rot = rotate_points(x_concentric, y_concentric, rotation_angle_rad + theta_robot)
            x_concentric_rot += x_robot
            y_concentric_rot += y_robot
            ax.plot(x_concentric_rot, y_concentric_rot, 'red')
        # grid_centers = self.calculate_grid_centers(radius, theta1, theta2, num_slices_radial, num_slices_angular,
        #                                       rotation_angle)
        # print(grid_centers)
        # ax.scatter(grid_centers[0, 0, 0], grid_centers[0, 0, 1], c='yellow', label='centers')
        # Adjust the plot for the rotated slice
        ax.set_aspect('equal')
        # plt.xlim(0, radius + 0.1)
        # plt.ylim(-radius - 0.1, radius + 0.1)
        # plt.show()
        return ax

    def distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def target_normalization(self, state, target, grid_centers):
        x, y, theta = state
        min_distance = float('inf')

        points = grid_centers

        # Rotate points
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        rotated_points = np.dot(points, rotation_matrix.T)  # Transpose to align dimensions

        # Translate points
        global_points = rotated_points + np.array([x, y])

        for idx in range(global_points.shape[0]):
            dist = self.distance(global_points[idx], target)
            if dist < min_distance:
                min_distance = dist
                normalized_target = global_points[idx]
                min_idx = idx + 1

        return min_idx, normalized_target