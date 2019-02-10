# cython: language_level=3
"""
With the same example path as in 'if __main__ == "name"', the A-Star needed 7 milliseconds

"""


# From http://code.activestate.com/recipes/578919-python-a-pathfinding-with-binary-heap/
import numpy as np
# from scipy.spatial.distance import cdist
import heapq
from collections import deque
import time
import math
from typing import Union, List, Set, Dict, Tuple

# from cpython cimport bool
cimport numpy as np
print(__file__)

cpdef int heuristic_manhattan((int, int) a, (int, int) b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

cpdef int heuristic_euclidean((int, int) a, (int, int) b):
    return (b[0] - a[0])**2 + (b[1] - a[1])**2

cpdef int sign(int n):
    if n > 0: return 1
    if n < 0: return -1
    return 0

# Cant remove second argument or else the compiler seems to be not working
cpdef int is_diagonal((int, int) direction, int placeholder=1):
    if abs(direction[0]) + abs(direction[1]) > 0:
        return 1
    return 0

cpdef int check_bounds((int, int) point, (int, int) array_shape):
    """ Returns 1 if is within boundse, 0 bounds were hit """
    if 0 <= point[0]  < array_shape[0] and 0 <= point[1] < array_shape[1]:
        return 1
    return 0

cpdef int check_wall((int, int) point, int[:, :] array, int wall_value):
    """ Returns 1 if is okay to move, 0 it is a wall"""
    if array[point[0], point[1]] == wall_value:
        return 0
    return 1

cpdef (int, int) calc_point((int, int) point, (int, int) offset, int subtract = 0):
    if subtract == 0:
        return (point[0] + offset[0], point[1] + offset[1])
    return (point[0] - offset[0], point[1] - offset[1])






