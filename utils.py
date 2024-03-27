import numpy as np

def get_distance(x1, y1, x2, y2):
	return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_angle(x1, y1, x2, y2):
	# return np.arctan2(y2 - y1, x2 - x1)
	return np.arctan2(y2 - y1, x2 - x1)


def normalize_angle(angle):
    """
    Normalize the angle to be within the range [-pi, pi].
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def get_angle_difference(angle1, angle2):
    """
    Calculate the shortest path difference between two angles,
    ensuring the result is within [-pi, pi].
    """
    difference = angle1 - angle2
    return normalize_angle(difference)
