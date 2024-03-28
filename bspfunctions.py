import numpy as np
import matplotlib.pyplot as plt

class SmartPSB():

    def __init__(self, d=1, num_y=5):
        self.d = d
        self.num_y = num_y
        self.grid_centers = []
        self.grid_centers_polar = []

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

        self.path = np.column_stack((valx[:-1], valy[:-1]))
        self.cartesian_to_polar()
        return self.path

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

    def calculate_cost_polar(self, p_spl, target, grid):
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
        # normalized_target = self.norm_target(target)
        # distance = np.sqrt((normalized_target[0] - final_point[0])**2 + (normalized_target[1] - final_point[1])**2)
        distance = 0
        # curve = 0
        collision = 0
        if self.obstacle_check_polar(grid):
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
        for idx, points in enumerate(grid_centers[2:]):
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

    def obstacle_check_polar(self, grid):
        # Get the indices of the zero elements using NumPy's np.where function
        _, obs_pos = self.obstacle_from_grid_polar(grid)
        collision = False
        for pos in self.path_polar:
            r = pos[0]
            theta = pos[1]
            for obs in obs_pos:
                if r <= (obs[0] + 0.15) and r > (obs[0] - 0.15) and theta <= (obs[1] + np.radians(5)) and theta > (obs[1] - np.radians(5)):
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

    def obstacle_from_grid_polar(self, grid):
        # Get the indices of the zero elements using NumPy's np.where function
        zero_indices = np.where(grid == 0)
        obs_pos = []
        obs_pos_polar = []
        # collision = False
        # Print the indices of the zero elements
        for i, j in zip(zero_indices[0], zero_indices[1]):
            obs_pos_polar.append(self.grid_centers_polar[j, i])
            obs_pos.append(self.grid_centers[j, i])

        return np.array(obs_pos), np.array(obs_pos_polar)

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

    def rotate_points(self, x, y, angle_deg):
        angle_rad = np.radians(angle_deg)
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        return x_rot, y_rot

    def calculate_grid_centers(self, radius, theta1, theta2, num_slices_radial, num_slices_angular, rotation_angle):
        # Function to calculate midpoints in polar coordinates
        def midpoint_polar(r1, r2, theta1, theta2):
            r_mid = (r1 + r2) / 2
            theta_mid = (theta1 + theta2) / 2
            return r_mid, theta_mid

        # Rotate a point in polar coordinates
        def rotate_point_polar(r, theta, angle_deg):
            theta_rad = np.radians(angle_deg)
            theta_rot = theta + theta_rad
            return r, theta_rot

        grid_centers = np.zeros((10, 9, 2), dtype=np.float32)
        grid_centers_polar = np.zeros((10, 9, 2), dtype=np.float32)

        # grid_centers=[]
        radial_intervals = np.linspace(0, radius, num_slices_radial + 1)
        angular_intervals = np.linspace(np.radians(theta1), np.radians(theta2), num_slices_angular + 1)

        # Calculate midpoints for each grid cell
        for i in range(num_slices_radial):
            for j in reversed(range(num_slices_angular)):
                r_mid, theta_mid = midpoint_polar(radial_intervals[i], radial_intervals[i + 1],
                                                  angular_intervals[j], angular_intervals[j + 1])
                # Apply rotation
                r_rot, theta_rot = rotate_point_polar(r_mid, theta_mid, rotation_angle)
                # Convert to Cartesian coordinates
                x_rot = r_rot * np.cos(theta_rot)
                y_rot = r_rot * np.sin(theta_rot)
                grid_centers[i, 8 - j] = [x_rot, y_rot]
                grid_centers_polar[i, 8 - j] = [r_rot, theta_rot]
        self.grid_centers = grid_centers[1:, :, :]
        self.grid_centers_polar = grid_centers_polar[1:, :, :]
        return grid_centers[1:, :, :], grid_centers_polar[1:, :, :]

    def create_polar_grid(self, radius, theta1, theta2, num_slices_radial, num_slices_angular, rotation_angle, ax):
        # Generate the outer arc of the circle slice
        theta_outer = np.linspace(np.radians(theta1), np.radians(theta2), 100)
        x_outer = radius * np.cos(theta_outer)
        y_outer = radius * np.sin(theta_outer)

        def rotate_points(x, y, angle_deg):
            angle_rad = np.radians(angle_deg)
            x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
            return x_rot, y_rot

        # Rotate the outer arc
        x_outer_rot, y_outer_rot = rotate_points(x_outer, y_outer, rotation_angle)

        # Re-initialize plot for rotated slice
        # fig, ax = plt.subplots()

        # Plot the rotated outer arc
        ax.plot(x_outer_rot, y_outer_rot, 'k')

        # Rotate and plot the straight lines
        for theta in np.linspace(np.radians(theta1), np.radians(theta2), num_slices_angular + 1):
            x, y = radius * np.cos(theta), radius * np.sin(theta)
            x_rot, y_rot = rotate_points(x, y, rotation_angle)
            ax.plot([0, x_rot], [0, y_rot], 'blue')

        # Rotate and plot the concentric arcs
        for r in np.linspace(0, radius, num_slices_radial + 1):
            x_concentric = r * np.cos(theta_outer)
            y_concentric = r * np.sin(theta_outer)
            x_concentric_rot, y_concentric_rot = rotate_points(x_concentric, y_concentric, rotation_angle)
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

    import numpy as np

    def cartesian_to_polar(self):

        x = self.path[:, 0]
        y = self.path[:, 1]

        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        # theta[theta < 0] += 2 * np.pi

        self.path_polar = np.vstack((r, theta)).T

        return self.path_polar
