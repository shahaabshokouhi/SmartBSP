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
        y = np.linspace(0, 3, 31)
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
        distance = np.sqrt((target[0] - final_point[0])**2 + (target[1] - final_point[1])**2)
        cost = curve + distance
        collision = 0
        if self.obstacle_check(grid):
            collision = 20

        cost = curve + distance + collision

        return cost

    def action2point(self, actions):
        # actions should be n-2 dimensional vector (np array)
        y_possible_points = np.linspace(-((self.num_y - 1)/2) * self.d, ((self.num_y - 1)/2) * self.d, self.num_y)
        y_points = []
        for i in range(actions.shape[0]):
            y_points.append(y_possible_points[actions[i]])
        y_points = np.array(y_points, dtype=np.float32)
        y_points = np.insert(y_points, 0, [0, 0])
        return y_points

    def obstacle_check(self, grid):
        # Get the indices of the zero elements using NumPy's np.where function
        zero_indices = np.where(grid == 0)
        obs_pos = []
        collision = False
        # Print the indices of the zero elements
        for i, j in zip(zero_indices[0], zero_indices[1]):
            obs_pos.append([j, i - 2])
            # print(f"Obstacle detected at x = {j} and y = {i - 2}")
        for pos in self.val:
            x = pos[0]
            y = pos[1]
            for obs in obs_pos:
                if x < (obs[0] + 0.5) and x > (obs[0] - 0.5) and y < (obs[1] + 0.5) and y > (obs[1] - 0.5):
                    collision = True
        return collision





