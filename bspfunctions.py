import numpy as np
import matplotlib.pyplot as plt

class SmartPSB():

    def __init__(self, d=1, num_y=5):
        self.d = d
        self.num_y = num_y

    def construct_sp(self, p):
        n = p.shape[0]
        order = 4

        if n < order:
            print(f"Error: Choose n >= order={order}")
            return None

        T = np.linspace(0, 3, n - order + 2)
        y = np.linspace(0, 3, 100)
        p_spl = self.deboor(T, p, y, order)
        return p_spl

    # Python equivalent of the deboor function inside construct_sp.m
    def deboor(self, T, p, t, order):
        m = p.shape[0]
        n = len(t)
        valx = np.zeros(n)
        valy = np.zeros(n)

        X = np.zeros((order, order))
        Y = np.zeros((order, order))
        # Creating extended knot vector
        Tx = np.concatenate(([T[0]] * (order - 1), T, [T[-1]] * (order - 1)))
        Ty = np.concatenate(([T[0]] * (order - 1), T, [T[-1]] * (order - 1)))

        # De Boor's Algorithm for x and y directions
        for l in range(n):
            t0 = t[l]
            kx = np.max(np.where(t0 >= Tx)[0])
            ky = np.max(np.where(t0 >= Ty)[0])

            if kx > m or ky > m:
                break

            # Spline in x-direction
            X[:, 0] = p[kx-order+1:kx+1, 0]
            for i in range(1, order):
                for j in range(i, order):
                    num = t0 - Tx[kx-order+j]
                    s = Tx[kx+j-i+1] - Tx[kx-order+j]
                    weight = 0 if num == 0 else num / s
                    X[j, i] = (1 - weight) * X[j-1, i-1] + weight * X[j, i-1]

            valx[l] = X[order-1, order-1]

            # Spline in y-direction
            Y[:, 0] = p[ky-order+1:ky+1, 1]
            for i in range(1, order):
                for j in range(i, order):
                    num = t0 - Ty[ky-order+j]
                    s = Ty[ky+j-i+1] - Ty[ky-order+j]
                    weight = 0 if num == 0 else num / s
                    Y[j, i] = (1 - weight) * Y[j-1, i-1] + weight * Y[j, i-1]

            valy[l] = Y[order-1, order-1]

        self.val = np.column_stack((valx[:-1], valy[:-1]))
        return self.val

    def calculate_cost(self, p_spl, target, grid):
        """
        Calculate the total curvature cost of a path.

        Parameters:
        - p_spl: A 2D NumPy array of path coordinates, shape (2, N), where N is the number of points.

        Returns:
        - Total curvature cost of the path.
        """
        # Transpose p_spl to match MATLAB's use of column vectors
        p_spl = p_spl.T
        curve = 0
        for ii in range(1, p_spl.shape[1] - 1):
            A = 0.5 * abs((p_spl[0, ii] - p_spl[0, ii-1]) * (p_spl[1, ii+1] - p_spl[1, ii]) -
                          (p_spl[1, ii] - p_spl[1, ii-1]) * (p_spl[0, ii+1] - p_spl[0, ii]))
            newCurve = (4 * A) / (np.linalg.norm(p_spl[:, ii-1] - p_spl[:, ii]) *
                                  np.linalg.norm(p_spl[:, ii] - p_spl[:, ii+1]) *
                                  np.linalg.norm(p_spl[:, ii+1] - p_spl[:, ii-1]))
            curve += newCurve

        final_point = p_spl[:, -1]  # Get the last column

        # normalized distance
        normalized_target = self.norm_target(target)
        distance = np.sqrt((normalized_target[0] - final_point[0])**2 + (normalized_target[1] - final_point[1])**2)
        # distance = 0
        # curve = 0
        collision = 0
        if self.obstacle_check(grid):
            collision = 1000

        cost = curve + 100 * distance + collision

        return cost

    def action2point(self, actions):
        # actions should be n-2 dimensional vector (np array)
        y_possible_points = np.linspace(-((self.num_y - 1)/2) * self.d, ((self.num_y - 1)/2) * self.d, self.num_y)
        y_possible_points = y_possible_points[::-1]
        y_points = []
        for i in range(actions.shape[0]):
            y_points.append(y_possible_points[actions[i]])
        y_points = np.array(y_points, dtype=np.float32)
        y_points = np.insert(y_points, 0, [0, 0])
        return y_points

    def action2point_polar(self, grid_centers, actions):
        control_points = []
        for idx, points in enumerate(grid_centers):
            control_points.append(points[actions[idx]])
        first_point = np.zeros((1, 2))
        second_point = np.array([[grid_centers[0,4,0], 0]])

        control_points.insert(0, second_point[0])  # Unpack the numpy array to match list format
        control_points.insert(0, first_point[0])  # Unpack the numpy array to match list format

        return np.array(control_points)

    def obstacle_check(self, grid):
        # Get the indices of the zero elements using NumPy's np.where function
        zero_indices = np.where(grid == 0)
        obs_pos = []
        collision = False
        # Print the indices of the zero elements
        for i, j in zip(zero_indices[0], zero_indices[1]):
            obs_pos.append([j + 1, np.floor(self.num_y/2) - i])
            # print(f"Obstacle detected at x = {j} and y = {2 - i}")
        for pos in self.val:
            x = pos[0]
            y = pos[1]
            for obs in obs_pos:
                if x < (obs[0] + 0.55) and x > (obs[0] - 0.55) and y < (obs[1] + 0.55) and y > (obs[1] - 0.55):
                    collision = True
        return collision

    def obstacle_from_grid(self, grid):
        # Get the indices of the zero elements using NumPy's np.where function
        zero_indices = np.where(grid == 0)
        obs_pos = []
        # collision = False
        # Print the indices of the zero elements
        for i, j in zip(zero_indices[0], zero_indices[1]):
            obs_pos.append([j + 1, np.floor(self.num_y/2) - i])
        return np.array(obs_pos)

    def distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def norm_target(self, target):
        """Find the point closest to the target location."""
        min_distance = float('inf')
        y = np.arange(-np.floor(self.num_y/2), np.floor(self.num_y/2) + 1)
        x = self.num_y * np.ones_like(y)
        points = np.column_stack((x, y))
        closest_point = None

        for point in points:
            dist = self.distance(point, target)
            if dist < min_distance:
                min_distance = dist
                normalized_target = point

        return normalized_target

