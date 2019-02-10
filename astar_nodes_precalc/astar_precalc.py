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

print(__file__)

from line_calc import can_draw_line, get_line, get_line_aa


# start = (0, 0)
# end = (4, 9)
# y, x = can_draw_line(start, end)
# y1, x1 = can_draw_line(end, start)
# y2, x2, val = _line_aa(*start, *end)
# assert x == x1
# assert y == y1
# b = np.zeros((5, 10))
# b[y, x] = 1
# c = np.zeros((5, 10))
# c[y2, x2] = 1
# print(b)
# print(c)
# # y, x = np.hsplit(a, 2)
# # print(x)
# exit()

def heuristic_manhattan(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def heuristic_euclidean(a, b):
    return (b[0] - a[0])**2 + (b[1] - a[1])**2

def heuristic_diagonal(a, b):
    # http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#diagonal-distance
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return dx + dx - 0.4142135623730951 * min(dx, dy)

def sign(n):
    if n > 0: return 1
    if n < 0: return -1
    return 0



# @profile
def astar_precompute(array: np.ndarray, wall=0, debug=True):
    """ Precompute all positions in the array and calculate jump points and distances to walls
    8 Arrays, one for each direction
    Negative values indicate the distance to the wall in that direction
    Positive values indicate the distance to the (primary) jump point in that direction
    Zero if wall is directly next to position """
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    right_turn = {
        # (y, x)
        # Vertical + horizontal
        (0, 1): (1, 0), # East
        (1, 0): (0, -1), # South
        (0, -1): (-1, 0), # West
        (-1, 0): (0, 1), # North
        # Diagonal
        (1, 1): (1, -1), # Southeast
        (1, -1): (-1, -1), # Southwest
        (-1, -1): (-1, 1), # Northwest
        (-1, 1): (1, 1), # Northeast
    }
    left_turn = {value: key for key, value in right_turn.items()}

    height, width = len(array), len(array[0])
    jump_points = {}
    t0 = time.time()

    pathable_coords = np.argwhere(array != wall)

    # Find jump points
    # for y, x in pathable_coords:
    for row in pathable_coords:
        y, x = row[0], row[1]
        """ The following forms are jump points (rotate and mirror to get all combinations):
        O = pathable, X = wall, J = jump point, ? = Wall or pathable
        
        ?XO
        OOJ
        ???
        """

        for direction in directions[:4]:
            # next = (y + direction[0], x + direction[1])
            # previous = (y - direction[0], x - direction[1])
            #
            # left_direction = left_turn[direction]
            # right_direction = right_turn[direction]

            # current_left = (y + left_direction[0], x + left_direction[1])
            # next_left = (next[0] + left_direction[0], next[1] + left_direction[1])
            # current_right = (y + right_direction[0], x + right_direction[1])
            # next_right = (next[0] + right_direction[0], next[1] + right_direction[1])

            # try:
            #     if (array[next] != wall and array[previous] != wall and
            #         (
            #             array[current_left] == wall and array[next_left] != wall
            #             or array[current_right] == wall and array[next_right] != wall
            #         )):
            #         jump_points[next] = []
            # except IndexError:
            #     break

            # Same as the commented code above but in one if statement and calculations done on the spot
            try:
                next = (y + direction[0], x + direction[1])
                left_direction = left_turn[direction]
                right_direction = right_turn[direction]
                # TODO: use list comprehension to generate needed Y and X values to query them once with the numpy array, e.g. "array[yy, xx] == (0, 0, 0, 255, 0)"
                if (array[next] != wall and array[(y - direction[0], x - direction[1])] != wall and
                    (
                        array[(y + left_direction[0], x + left_direction[1])] == wall and array[(next[0] + left_direction[0], next[1] + left_direction[1])] != wall
                        or array[(y + right_direction[0], x + right_direction[1])] == wall and array[(next[0] + right_direction[0], next[1] + right_direction[1])] != wall
                    )):
                    jump_points[next] = []
            except IndexError:
                break

    t1 = time.time()
    print(f"Calculating jump points: {round(t1-t0, 5)} s")

    # Connect jump points
    for p1 in jump_points.keys():
        for p2 in jump_points.keys():
            if p1 == p2: continue

            could_draw_line = can_draw_line(*p1, *p2, array, wall)
            if could_draw_line:
                distance = heuristic_euclidean(p1, p2)
                jump_points[p1].append((p2, distance))
                jump_points[p2].append((p1, distance))

    t2 = time.time()
    print(f"Calculating jump point connections: {round(t2-t1, 5)} s")
    connection_per_point = [len(value) for value in jump_points.values()]
    avg_connection_per_point = sum(connection_per_point) / len(connection_per_point)
    print(f"Jump points: {len(jump_points)}, average amount of connections per point: {avg_connection_per_point}")

    """
    A dictionary with entries:
    {point0: [((p0_y, p0_x), dist0), ((p1_y, p0_x), dist1), ...], point1: [...]}
    """
    return jump_points


# @profile
def astar_search(start: Tuple[int, int], goal: Tuple[int, int], array: np.ndarray, jump_points: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]], wall=9, debug=True) -> List[Tuple[int, int]]:
    """ Input values:
    start: a tuple (y, x)
    goal: a tuple(y, x)
    array: a numpy array with shape (height, width)"""
    height, width = array.shape
    jump_points = jump_points.copy()

    # print(height, width)

    # @profile
    def connect_point_to_jump_points(p1):
        if p1 in jump_points: return

        jump_points[p1] = []
        for p2 in jump_points.keys():
            if heuristic_euclidean(p1, p2) > 200: continue
            could_draw_line = can_draw_line(*p1, *p2, array, 9)
            if could_draw_line:
                distance = heuristic_euclidean(p1, p2)
                jump_points[p1].append((p2, distance))
                jump_points[p2].append((p1, distance))

    def generate_path(p1, p2):
        path = []
        current = p2
        print(f"Generating path from {p1} to {p2}")
        while current != p1:
            path.append(current)
            current = came_from[current]
        path.append(p1)
        path.reverse()
        if debug:
            pass
        return path

    if start == goal:
        return [start]

    # Test if start and goal can be connected directly
    could_draw_line = can_draw_line(*start, *goal, array, 9)
    if could_draw_line:
        return [(x, y) for y, x in get_line(*start, *goal)]

    open_list = []
    open_set: Set[Tuple[int, int]] = set()
    closed_set: Set[Tuple[int, int]] = set()
    heuristic: callable = heuristic_euclidean
    dist_to_start: Dict[Tuple[int, int], float] = {}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    heapq.heappush(open_list, (heuristic(start, goal), start))
    open_set.add(start)
    dist_to_start[start] = 0

    connect_point_to_jump_points(start)
    connect_point_to_jump_points(goal)

    while open_list:
        current = heapq.heappop(open_list)[1]

        if current == goal:
            return generate_path(start, goal)

        closed_set.add(current)

        # print(dist_to_start[current], heuristic(current, goal), current, start, goal)

        for point in jump_points[current]:
            next_point, distance = point[0], point[1]
            distance_to_start = dist_to_start[current] + distance

            if next_point in closed_set and dist_to_start[next_point] >= distance_to_start:
                continue

            if next_point not in open_set or distance_to_start < dist_to_start[next_point]:
                came_from[next_point] = current
                dist_to_start[next_point] = distance_to_start

                calc_distance_to_end = heuristic(next_point, goal)
                total_distance = distance_to_start + calc_distance_to_end
                # print(total_distance, calc_distance_to_end, next_point, start, goal)
                heapq.heappush(open_list, (total_distance, next_point))
                open_set.add(next_point)

    return []



if __name__ == "__main__":
    pathing_grid = np.load("numpy_placement_better.npy")
    pathing_grid[pathing_grid == 0] = 9 # Turn all 0 values to 9 values which are walls
    pathing_grid[pathing_grid < 9] = 0 # Turn all other values to 1
    expansions = np.load("numpy_expansions.npy")
    height, width = len(pathing_grid), len(pathing_grid[0])
    print(height, width, pathing_grid.shape)

    t0 = time.time()
    jump_points = jps_precompute(pathing_grid, wall=9)
    t1 = time.time()

    spawn1 = expansions[6] # 35.5, 35.5
    spawn2 = expansions[-1] # 140.5 140.5
    spawn1 = (35.5, 35.5)
    spawn2 = (140.5, 140.5)
    # print(expansions)
    # print(spawn1)
    # print(spawn2)

    spawn1_correct = int(height - 1 - spawn1[1]+0.5), int(spawn1[0]-0.5)
    spawn2_correct = int(height - 1 - spawn2[1]+0.5), int(spawn2[0]-0.5)

    # # # Testing top left to bottom right
    # spawn1_correct = (40, 30)
    # spawn2_correct = (150, 140)
    t0 = time.time()
    result = jps_search(spawn1_correct, spawn2_correct, pathing_grid, jump_points)
    t1 = time.time()
    print(f"Path calculated!\nTime: {t1-t0}\nPath: {result}")

