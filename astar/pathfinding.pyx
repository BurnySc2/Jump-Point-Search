# cython: language_level=3

# Original source from http://code.activestate.com/recipes/578919-python-a-pathfinding-with-binary-heap/

# Somehow importing cdist improved performance for no obvious reason
from scipy.spatial.distance import cdist

import time
import heapq

cimport cython
from cpython cimport bool
# from cpython cimport float
cimport numpy as np

from cython.view cimport array as cvarray

cdef int heuristic_manhattan((int, int) a, (int, int) b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

cdef int heuristic_euclidean((int, int) a, (int, int) b):
    return (b[0] - a[0])**2 + (b[1] - a[1])**2

# # http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#diagonal-distance
# cdef float h1euristic_diagonal((int, int) a, (int, int) b):
#     cdef int dx, dy
#     cdef float result
#     dx = abs(b[0] - a[0])
#     dy = abs(b[1] - a[1])
#     result = dx + dx# - 0.4142135623730951 * min(dx, dy)
#     return result

cdef int sign(int n):
    if n > 0: return 1
    if n < 0: return -1
    return 0

cdef int is_diagonal((int, int) direction):
    return abs(direction[0]) + abs(direction[1]) > 0

cdef int check_bounds((int, int) point, (int, int) array_size):
    """ Returns 1 if is within bounds, 0 if bounds were hit """
    if 0 <= point[0]  < array_size[0] and 0 <= point[1] < array_size[1]:
        return 1
    return 0

cdef int check_wall((int, int) point, int[:, :] array, int wall_value):
    """ Returns 1 if is okay to move, 0 it is a wall"""
    if array[point[0]][point[1]] == wall_value:
        return 0
    return 1

# cdef int get_numpy_value(np.ndarray[int, ndim=2] array, (int, int) index):
#     return array[index[0], index[1]]


# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
def astar((int, int) start, (int, int) goal, np.ndarray array, int diagonal=0, bool debug=True):
    """ Input needs to be
    start: (y, x)
    goal: (y, x)
    array with shape (rows=map height, columns=map width) """
    cdef int array_height, array_width
    cdef int neighbor_value, dist_to_goal
    cdef int x, y
    cdef float t0, t1, sqrt2, dist, calc_distance_to_start, total_distance, distance_to_start
    cdef (int, int) p
    cdef (int, int) neighbor, current, next_point, direction
    cdef (int, int)[8] directions
    # Defining list, set, dict only seems to make it slower

    # https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#using-memoryviews
    array_height, array_width = len(array), len(array[0])

    # Creates integer array
    # cyarr = cvarray(shape=(array_height, array_width), itemsize=sizeof(int), format="i")
    # Creates float array, faster indexing than numpy arrays somehow
    cyarr = cvarray(shape=(array_height, array_width), itemsize=sizeof(float), format="f")
    cdef float[:, :] dist_to_start_dict = cyarr

    # Const keyword makes it read-only
    cdef const int[:, :] mem_array = array
    # cdef np.ndarray[np.int32_t, ndim=2] mem_array = np.arange([10], dtype=np.int32).reshape(2, 5)

    came_from = {}
    sqrt2 = 2**0.5

    if diagonal:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        # heuristic = heuristic_diagonal
        heuristic = heuristic_euclidean
    elif not diagonal:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        heuristic = heuristic_manhattan

    # Open and closed set
    close_set = set()
    dist_to_start_dict[start[0], start[1]] = 0

    # Heap
    oheap = []
    oheap_set = set()
    heapq.heappush(oheap, (heuristic(start, goal), start))

    t0 = time.time()

    while oheap:
        current = heapq.heappop(oheap)[1]
        oheap_set.discard(current)

        if current[0] == goal[0] and current[1] == goal[1]:
            path = []
            while current[0] != start[0] or current[1] != start[1]:
                path.append(current)
                current = came_from[current]
            path.reverse()
            if debug:
                cost = sum(mem_array[p[0], p[1]] for p in path) # Cost of path
                t1 = time.time()
                print(f"Path found!\nTime: {t1-t0}\nOpen list: {len(oheap)}\nClosed list: {len(close_set)}\nDist to start: {len(path)}")

            return path, cost, close_set

        close_set.add(current)
        for direction in directions:
            if diagonal == 0 and direction[0] == 0 and direction[1] == 0:
                break

            next_point = (current[0] + direction[0], current[1] + direction[1])

            # Hit array walls or hit normal wall
            if not check_bounds(next_point, (array_height, array_width)):
            # if not (0 <= next_point[0] < array_height) or not (0 <= next_point[1] < array_width):
                continue

            # neighbor_value = get_numpy_value(array, next_point) # This is slower
            neighbor_value = mem_array[next_point[0]][next_point[1]]
            if neighbor_value == 9: # 9 is wall
                continue

            # If it was diagonal movement
            if abs(direction[0]) + abs(direction[1]) > 1:
                # dist = 2
                dist = sqrt2
            else:
                dist = 1

            calc_distance_to_start = dist_to_start_dict[current[0], current[1]] + dist # + neighbor_value
            distance_to_start = dist_to_start_dict[next_point[0], next_point[1]]
            # distance_to_start = dist_to_start_dict.get(next_point, 0)

            if next_point in close_set and calc_distance_to_start >= distance_to_start:
                continue

            if next_point not in oheap_set or calc_distance_to_start < distance_to_start:
                came_from[next_point] = current
                dist_to_start_dict[next_point[0], next_point[1]] = calc_distance_to_start
                dist_to_goal = heuristic(next_point, goal)
                total_distance = calc_distance_to_start + dist_to_goal
                heapq.heappush(oheap, (total_distance, next_point))
                oheap_set.add(next_point)

    return False, False, close_set
