import numpy as np
import matplotlib.pyplot as plt
import torch
from network import ConvNet
from bspfunctions import SmartPSB
from utils import *
from parameters import *
class Robot:
    def __init__(self, initial_state, robot_path=None, wheel_radius=0.05, wheel_base=0.15,
                 kp_linear=0.3, kd_linear=0.2, ki_linear=0.1,
                 kp_angular=0.3, kd_angular=0.2, ki_angular=0.1):
        self.state = np.array(initial_state)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.path = robot_path
        self.kp_linear = kp_linear
        self.kd_linear = kd_linear
        self.ki_linear = ki_linear

        self.kp_angular = kp_angular
        self.kd_angular = kd_angular
        self.ki_angular = ki_angular

        self.prev_error_position = 0
        self.prev_error_angle = 0

        self.prev_body_to_goal = 0
        self.prev_waypoint_idx = -1
        self.dt = 0.5
        self.trajectory = np.zeros((1, 3))
        self.distance_lookahead = 0.2

    def update_state(self, left_wheel_velocity, right_wheel_velocity, dt):
        x, y, theta = self.state
        v_l, v_r = left_wheel_velocity, right_wheel_velocity
        v = self.wheel_radius * (v_r + v_l) / 2.0
        omega = self.wheel_radius * (v_r - v_l) / self.wheel_base
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = omega * dt
        self.state = np.array([x + dx, y + dy, theta + dtheta])
        return self.state

    def transform_path_to_global(self, path):
        x, y, theta = self.state
        # Create a rotation matrix
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

        # Transform each point
        path_global = np.dot(path, rotation_matrix.T) + [x, y]

        return path_global

    def get_closest_waypoint_index(self):
        distances = [np.linalg.norm(np.array(waypoint[:2]) - np.array(self.state[:2])) for waypoint in self.path]
        return np.argmin(distances)

    def get_reference_index(self):
        closest_index = self.get_closest_waypoint_index()

        for i in range(closest_index + 1, len(self.path)):
            if np.linalg.norm(np.array(self.path[i][:2]) - np.array(self.state[:2])) >= self.distance_lookahead:
                # Optionally, check if the previous index is closer to the desired lookahead distance
                # if i > 0 and (np.linalg.norm(
                #         np.array(self.path[i - 1][:2]) - np.array(self.state[:2])) - self.distance_lookahead) < \
                #         (np.linalg.norm(
                #             np.array(self.path[i][:2]) - np.array(self.state[:2])) - self.distance_lookahead):
                #     return i - 1
                return i
        return len(self.path) - 1  # Return the final state if no state is beyond the lookahead distance
    def trackPID(self, n = 10):
        trajectory = [self.state]
        reference_index = 0
        while reference_index < n:
            reference_index = self.get_reference_index()
            if reference_index == 32:
                print()
            target = self.path[reference_index, :]
            error_position = get_distance(self.state[0], self.state[1], target[0], target[1])

            body_to_goal = get_angle(self.state[0], self.state[1], target[0], target[1])
            # body_to_nose = get_angle(x[0, 0], x[1, 0], nose[0], nose[1])

            # if self.prev_waypoint_idx == waypoint_idx and 350<(abs(self.prev_body_to_goal - body_to_goal)*180/np.pi):
            # 	print("HERE")
            # 	body_to_goal = self.prev_body_to_goal
            # error_angle = body_to_goal - self.state[2]
            error_angle = get_angle_difference(body_to_goal, self.state[2])

            linear_velocity_control = self.kp_linear * error_position + self.kd_linear * (
                        error_position - self.prev_error_position)
            angular_velocity_control = self.kp_angular * error_angle + self.kd_angular * (
                        error_angle - self.prev_error_angle)

            self.prev_error_angle = error_angle
            self.prev_error_position = error_position

            self.prev_waypoint_idx = reference_index
            self.prev_body_to_goal = body_to_goal

            if linear_velocity_control > MAX_LINEAR_VELOCITY:
                linear_velocity_control = MAX_LINEAR_VELOCITY

            right_wheel_velocity, left_wheel_velocity = self.uniToDiff(linear_velocity_control, angular_velocity_control)
            state = self.update_state(left_wheel_velocity, right_wheel_velocity, self.dt)
            trajectory.append(self.state)
        self.trajectory = np.vstack((self.trajectory, trajectory))
        return  np.array(trajectory)

    def getPath(self, path):
        self.path = path

    def uniToDiff(self, v, w):
        vR = (2 * v + w * self.wheel_base) / (2 * self.wheel_radius)
        vL = (2 * v - w * self.wheel_base) / (2 * self.wheel_radius)
        return vR, vL