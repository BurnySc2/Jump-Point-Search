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

def heuristic_manhattan(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def heuristic_euclidean(a, b):
    return math.pow(b[0] - a[0], 2) + math.pow(b[1] - a[1], 2)

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
def jps_search(start, goal, array: np.ndarray, wall_value=0, debug=True):
    """ Precompute all positions in the array and calculate jump points and distances to walls
    8 Arrays, one for each direction
    Negative values indicate the distance to the wall in that direction
    Positive values indicate the distance to the (primary) jump point in that direction
    Zero if wall is directly next to position """
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # vertical_directions = {(1, 0), (-1, 0)}
    # horizontal_directions = {(0, 1), (0, -1)}
    diagonals = {(1, 1), (1, -1), (-1, 1), (-1, -1)}
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
    half_right_turn = {
        # 45 degrees turns, required for diagonal exploration
        # From Vertical + horizontal
        (0, 1): (1, 1), # East
        (1, 0): (1, -1), # South
        (0, -1): (-1, -1), # West
        (-1, 0): (-1, 1), # North
        # From Diagonal
        (1, 1): (1, 0), # Southeast
        (1, -1): (0, -1), # Southwest
        (-1, -1): (-1, 0), # Northwest
        (-1, 1): (0, 1), # Northeast
    }
    half_left_turn = {value: key for key, value in half_right_turn.items()}
    height, width = array.shape

    array[goal] = 8

    def generate_path(p1, p2):
        if debug:
            t1 = time.time()
            print(f"Path found!\nTime: {t1-t0}\nOpen list: {len(open_list)}\nClosed list: {len(closed_set)}\nDist to start: {dist_to_start_dict[p2]}")
        path = []
        current = p2
        while current != p1:
            # print(current)
            path.append(current)
            current = previous_point_dict[current]
        path.append(p1)
        path.reverse()
        return path

    def add_point_to_explored(source, target, distance, is_diagonal=True):
        if source != target:
            previous_point_dict[target] = source
            dist_to_start_dict[target] = dist_to_start_dict[source] + distance * (sqrt2 if is_diagonal else 1)
            closed_set.add(source)
            array[source] = 1
            array[target] = 2

    def check_bounds_or_wall(point):
        """ Returns True if is within bounds and not wall value, False if wall or bounds were hit """
        if 0 <= point[0] < height and 0 <= point[1] < width and array[point] != wall_value:
            return True
        return False

    def check_jump_point(behind_point, start_point, next_point, dir, check_left=False):
        left = left_turn[dir] if check_left else right_turn[dir]
        left_neighbor_start = (start_point[0] + left[0], start_point[1] + left[1])
        # left_neighbor_behind = (behind_point[0] + left[0], behind_point[1] + left[1])
        left_neighbor_current = (next_point[0] + left[0], next_point[1] + left[1])
        if array[left_neighbor_start] == wall_value and array[left_neighbor_current] != wall_value and array[behind_point] != wall_value:# and (array[left_neighbor_behind] == wall_value or min(left_neighbor_behind) < 0):
            return left_neighbor_current, True
        else:
            return left_neighbor_current, False


    # @profile
    def explore_cardinal_direction(start_point, direction):
        """ Explore north, east, west, south only"""
        next_point = start_point
        distance_explored = 0
        while 1:
            distance_explored += 1
            current = next_point
            next_point = (next_point[0] + direction[0], next_point[1] + direction[1])
            behind_point = (current[0] - direction[0], current[1] - direction[1])

            if current == goal:
                print("hello3")
                add_point_to_explored(start_point, goal, distance_explored, False)
                return True

            # If wall was hit, stop exploring this direction
            if not check_bounds_or_wall(next_point):
                break

            # print("cardinal", next_point, direction)

            # Check if wall is on right and the next place on the right is not a wall, then that is a jump point
            try:
                right_jump_point, is_jump_point = check_jump_point(behind_point, current, next_point, direction, check_left=False)
                distance_to_start = dist_to_start_dict[start_point] + distance_explored + 1
                if is_jump_point:
                    dist_to_start_dict[right_jump_point] = distance_to_start
                    distance_to_end = heuristic(right_jump_point, goal)
                    total_distance = distance_to_start + distance_to_end
                    # total_distance = distance_to_start**2 + distance_to_end
                    # print(f"Found jump point {right_jump_point}, {right_turn[direction]}")
                    forced_neighbors[right_jump_point] = distance_to_start
                    heapq.heappush(open_list, (total_distance, right_jump_point, right_turn[direction]))

                    add_point_to_explored(start_point, current, 1, False)
                    add_point_to_explored(current, right_jump_point, 1, True)
            except IndexError: pass

            try:
                # Check if wall is on left and the next place on the left is not a wall, then that is a jump point
                left_jump_point, is_jump_point = check_jump_point(behind_point, current, next_point, direction, check_left=True)
                distance_to_start = dist_to_start_dict[start_point] + distance_explored - 1 + sqrt2
                if is_jump_point:
                    # dist_to_start_dict[left_jump_point] = distance_to_start
                    distance_to_end = heuristic(left_jump_point, goal)
                    total_distance = distance_to_start + distance_to_end
                    # total_distance = distance_to_start**2 + distance_to_end
                    # print(f"Found jump point {left_jump_point}, {left_turn[direction]}")
                    forced_neighbors[left_jump_point] = distance_to_start
                    heapq.heappush(open_list, (total_distance, left_jump_point, left_turn[direction]))

                    add_point_to_explored(start_point, current, 1, False)
                    add_point_to_explored(current, left_jump_point, 1, True)

            except IndexError: pass

    def explore_diagonal_direction(start_point, direction):
        next_point = start_point
        distance_explored = 0
        while 1:
            distance_explored += 1
            current = next_point
            next_point = (next_point[0] + direction[0], next_point[1] + direction[1])

            # print("diagonal", current, direction)

            add_point_to_explored(start_point, current, distance_explored, True)

            explore_cardinal_direction(current, half_left_turn[direction])
            explore_cardinal_direction(current, half_right_turn[direction])

            # If wall was hit, stop exploring this direction
            if not check_bounds_or_wall(next_point):
                break

    sqrt2 = 2**0.5

    # Heapq, contains all points that need to be explored (y, x, direction_tuple)
    open_list = []
    forced_neighbors = {} # {Point: distance} Point of forced neighbor, distance to start
    dist_to_start_dict = {}
    closed_set = set()
    heuristic = heuristic_euclidean
    heapq.heappush(open_list, (heuristic(start, goal), start, None))
    # open_dict[start] = 0 # Could be replaced with a numpy array? Which one is faster
    dist_to_start_dict[start] = 0
    previous_point_dict = {} # To generate the path

    t0 = time.time()

    print(start, goal)

    while open_list:
        _, current, direction = heapq.heappop(open_list)

        # TODO this is optional ?! Loops in circle if commented out or not
        # if current in closed_set and direction in diagonals:
        #     continue

        # print(len(open_list), len(closed_set), current, direction)

        # None is set for starting point
        if current == start:
            for direction in directions:
                if direction in diagonals:
                    diag_pos = current[0] + direction[0], current[1] + direction[1]
                    if not check_bounds_or_wall(diag_pos):
                        continue
                    add_point_to_explored(current, diag_pos, 1, True)
                    explore_diagonal_direction(diag_pos, direction)
                else:
                    explore_cardinal_direction(current, direction)
        else:
            """
Here, we start from a forced neighbor
OOOO
OXJO
XOOO

XOOO
OXJO
OOOO
Here, the direction is towards south, so we explore
south east, south west, west, south, east
To not explore the same positions twice, we put the diagonal explorations by offset (1, 1)
            """
            # Directions
            current_right = right_turn[direction]
            current_left = left_turn[direction]

            # Explore cardinal
            explore_cardinal_direction(current, direction)
            explore_cardinal_direction(current, current_left)
            explore_cardinal_direction(current, current_right)

            # Positions for when directions are diagonal
            current_half_right = half_right_turn[direction]
            current_half_left = half_left_turn[direction]
            current_half_right_pos = current[0] + current_half_right[0], current[1] + current_half_right[1]
            current_half_left_pos = current[0] + current_half_left[0], current[1] + current_half_left[1]

            # Explore diagonal
            if check_bounds_or_wall(current_half_right_pos):
                add_point_to_explored(current, current_half_right_pos, 1, True)
                explore_diagonal_direction(current_half_right_pos, current_half_right)

            if check_bounds_or_wall(current_half_left_pos):
                add_point_to_explored(current, current_half_left_pos, 1, True)
                explore_diagonal_direction(current_half_left_pos, current_half_left)

            if goal in dist_to_start_dict:
                return generate_path(start, goal)

        closed_set.add(current)

    # print(array)
    np.save("closed_set", list(closed_set))
    # print("written to file")






if __name__ == "__main__":
    pathing_grid = np.load("numpy_placement_better.npy")
    pathing_grid[pathing_grid == 0] = 9 # Turn all 0 values to 9 values which are walls
    pathing_grid[pathing_grid < 9] = 1 # Turn all other values to 1
    expansions = np.load("numpy_expansions.npy")
    height, width = len(pathing_grid), len(pathing_grid[0])
    print(height, width, pathing_grid.shape)

    # t0 = time.time()
    # precomputed = jps_precompute(pathing_grid, wall_value=9)
    # t1 = time.time()
    # print(f"{len(precomputed[0])} jump points, time taken: {round(t1-t0, 3)}s, jump points: {precomputed[0]}")
    # np.save("jump_points", list(precomputed[0]))
    # # np.save("no_connection", list(a[1]))


    spawn1 = expansions[6] # 35.5, 35.5
    spawn2 = expansions[-1] # 140.5 140.5
    spawn1 = (35.5, 35.5)
    spawn2 = (140.5, 140.5)
    # print(expansions)
    # print(spawn1)
    # print(spawn2)

    # Pre compute once for numba to not screw with results
    heuristic_manhattan(spawn1, spawn2)
    heuristic_euclidean(spawn1, spawn2)
    heuristic_diagonal(spawn1, spawn2)

    spawn1_correct = int(height - 1 - spawn1[1]+0.5), int(spawn1[0]-0.5)
    spawn2_correct = int(height - 1 - spawn2[1]+0.5), int(spawn2[0]-0.5)

    # # Testing top left to bottom right
    # spawn1_correct = (40, 30)
    # spawn2_correct = (150, 140)

    test = np.array([
        [0, 9, 0, 0, 0, 0],
        [0, 9, 9, 9, 0, 0],
        [0, 9, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    print(test.shape)
    p1 = (0, 0)
    p2 = (0, 2)

    # result = jps_search(p1, p2, test, wall_value=9)
    result = jps_search(spawn1_correct, spawn2_correct, pathing_grid, wall_value=9)

    print(f"Path: {result}")

    np.save("path", result)

    # TODO: Path smoothener

    exit()
