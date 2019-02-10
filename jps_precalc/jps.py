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
def jps_precompute(array: np.ndarray, wall=0, debug=True):
    """ Precompute all positions in the array and calculate jump points and distances to walls
    8 Arrays, one for each direction
    Negative values indicate the distance to the wall in that direction
    Positive values indicate the distance to the (primary) jump point in that direction
    Zero if wall is directly next to position """
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    vertical_directions = {(1, 0), (-1, 0)}
    horizontal_directions = {(0, 1), (0, -1)}
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

    jump_points = set() # Contains all points that are jump points
    data_type = object # np.int16
    # default_value = np.iinfo(data_type).min
    default_value = -10000000
    east_array = np.full_like(array, default_value, dtype=data_type)
    north_array = np.full_like(array, default_value, dtype=data_type)
    west_array = np.full_like(array, default_value, dtype=data_type)
    south_array = np.full_like(array, default_value, dtype=data_type)
    south_east_array = np.full_like(array, default_value, dtype=data_type)
    south_west_array = np.full_like(array, default_value, dtype=data_type)
    north_west_array = np.full_like(array, default_value, dtype=data_type)
    north_east_array = np.full_like(array, default_value, dtype=data_type)
    directions_greater_zero = np.full_like(array, False, dtype=data_type)
    directions_less_zero = np.full_like(array, False, dtype=data_type)

    direction_dict_array = {
        (0, 1): east_array,
        (-1, 0): north_array,
        (0, -1): west_array,
        (1, 0): south_array,
        (1, 1): south_east_array,
        (1, -1): south_west_array,
        (-1, -1): north_west_array,
        (-1, 1): north_east_array,
    }
    opposite_direction_dict_array = {
        (0, 1): west_array,
        (-1, 0): south_array,
        (0, -1): east_array,
        (1, 0): north_array,
        (1, 1): north_west_array,
        (1, -1): north_east_array,
        (-1, -1): south_east_array,
        (-1, 1): south_west_array,
    }

    # @profile
    def explore_direction(pos_y, pos_x, dir):
        if dir[0] != 0:
            pathable = np.argwhere(array[:, pos_x] != wall).tolist()
            positions_pathable: Set[Tuple[int, int]] = {(y[0], pos_x) for y in pathable}
        else:
            pathable = np.argwhere(array[pos_y, :] != wall).tolist()
            positions_pathable: Set[Tuple[int, int]] = {(pos_y, x[0]) for x in pathable}

        for current in positions_pathable:
            next = (current[0] + dir[0], current[1] + dir[1])
            if next not in positions_pathable or next in jump_points:
                continue
            previous = (current[0] - dir[0], current[1] - dir[1])
            if previous not in positions_pathable:
                continue

            # TODO: When coming from a certain direction, the jump point only then has to be visited
            # See 9:45 https://www.gdcvault.com/play/1022094/JPS-Over-100x-Faster-than%20jps+%20goal%20bounding
            # Some jump points have only arrows from certain directions

            # Check if wall is on right and the next place on the right is not a wall, then this is a jump point
            try:
                right = right_turn[dir]
                right_neighbor = (current[0] + right[0], current[1] + right[1])
                if array[right_neighbor] == wall and array[right_neighbor[0] + dir[0], right_neighbor[1] + dir[1]] != wall:
                    jump_points.add(next)
                    continue
            except IndexError: pass

            try:
                # Check if wall is on left and the next place on the left is not a wall, then this is a jump point
                left = left_turn[dir]
                left_neighbor = (current[0] + left[0], current[1] + left[1])
                if array[left_neighbor] == wall and array[left_neighbor[0] + dir[0], left_neighbor[1] + dir[1]] != wall:
                    jump_points.add(next)
            except IndexError: pass

    # Find all possible jump points
    """ Origin is on the left, all J are jump points, diagonal movement through tight path is not allowed
    OOO OOO OOO OOO
    OOJ OJO OOX OJX
    OXO XOO OXO XOO
    """
    # Explore east and west starting at bounds
    for y in range(0, height):
        explore_direction(y, 0, (0, 1))
        explore_direction(y, width - 1, (0, -1))
    # Explore north and south starting at bounds
    for x in range(0, width):
        explore_direction(0, x, (1, 0))
        explore_direction(height - 1, x, (-1, 0))


    def mark_wall_distance(start_point, direction, distance):
        for dist in range(distance):
            current = (start_point[0] + dist * direction[0], start_point[1] + dist * direction[1])
            if opposite_direction_dict_array[direction][current] == default_value:
                opposite_direction_dict_array[direction][current] = -dist

    # @profile
    def set_directional_distances(point, connect_to_distant_jump_points_and_walls=False):
        """ This function is called twice
        1) First time, it is called for each jump point, so this function wanders in each direction to connect points to it
        2) Second time, it is called for all points - this function wanders into a direction if the direction_array has no default value set. Then it checks if any point in that wandering direction has a connection to a jump point in 45° angle. If there is no such connection, the distance to the nearest wall will be used (as negative value). """
        for dir in directions:
            # Only want to find connection to other jump points with this function
            if connect_to_distant_jump_points_and_walls and direction_dict_array[dir][point] != default_value:
                continue
            distance_travelled = 0
            current = point # (y, x)
            while 1:
                distance_travelled += 1
                prev = current
                current = (prev[0] + dir[0], prev[1] + dir[1])

                # Use try except to check if bounds were hit
                try:
                    # Check if wall was hit
                    if array[current] == wall:
                        if connect_to_distant_jump_points_and_walls:
                            mark_wall_distance(prev, (-dir[0], -dir[1]), distance_travelled)
                        break
                except IndexError:
                    if connect_to_distant_jump_points_and_walls: # We dont really need this here since in SC2 we have walls instead of array bounds as walls
                        mark_wall_distance(prev, (-dir[0], -dir[1]), distance_travelled)
                    break

                if point in jump_points:
                    # Node does not have a value (default) or another more distant jump point went through this node, the second check is redundant i think
                    if opposite_direction_dict_array[dir][current] == default_value or distance_travelled < opposite_direction_dict_array[dir][current]:
                        opposite_direction_dict_array[dir][current] = distance_travelled

                if connect_to_distant_jump_points_and_walls:
                    # Starting node may not have a connection to any jump point, find a connection by travelling vertical or horizontally, the nearest node that has a diagonal connection to a jump point will be selected
                    # At this point of the iteration (is_jump_point=False) all the jump points have been found, and now this needs to connect points that have no direct connection (in one direction) to a jump point
                    # E.g. when travelling east, if a point has a connection to jump point in north-east or south-east direction, then it will use this point as connection
                    if direction_dict_array[dir][point] <= 0:
                        half_left = half_left_turn[dir] # E.g. from east to north east
                        half_right = half_right_turn[dir]
                        distance_to_jump_point = []
                        if direction_dict_array[half_left][current] > 0:
                            distance_to_jump_point.append(direction_dict_array[half_left][current])
                        if direction_dict_array[half_right][current] > 0:
                            distance_to_jump_point.append(direction_dict_array[half_right][current])
                        if distance_to_jump_point: # and (direction_dict_array[dir][point] <= 0 or distance_travelled < direction_dict_array[dir][point]):
                            # TODO: check if the if statement needs to be uncommented if it makes any difference
                            direction_dict_array[dir][point] = distance_travelled

    def set_directions(y, x):
        directions_greater_zero[y, x] = [dir for dir in directions if direction_dict_array[dir][(y, x)] > 0]
        directions_less_zero[y, x] = [dir for dir in directions if direction_dict_array[dir][(y, x)] < 0]

    # For each pathable node, set the distance to the nearest jump point (positive values) or distance to wall (0 or negative) (actually the second part will be done in the loop below)
    for jump_point in jump_points:
        set_directional_distances(jump_point)

    # Set the distances to walls for all unconnected points
    for y in range(height):
        for x in range(width):
            value = array[y, x]
    # for y, row in enumerate(array):
    #     for x, value in enumerate(row):
            if value == wall:
                continue
            set_directional_distances((y, x), connect_to_distant_jump_points_and_walls=True)
            # Set directions that can be traveled (towards jumps points)
            set_directions(y, x)

    # Testing
    if debug:
        no_connection_count = 0
        no_connection = []
        for y, row in enumerate(array):
            for x, value in enumerate(row):
                if value == wall:
                    continue
                if east_array[y, x] == default_value:
                    print(f"1Why is this still default value at {y, x}?")
                if north_array[y, x] == default_value:
                    print(f"2Why is this still default value at {y, x}?")
                if west_array[y, x] == default_value:
                    print(f"3Why is this still default value at {y, x}?")
                if south_array[y, x] == default_value:
                    print(f"4Why is this still default value at {y, x}?")
                if south_east_array[y, x] == default_value:
                    print(f"5Why is this still default value at {y, x}?")
                if south_west_array[y, x] == default_value:
                    print(f"6Why is this still default value at {y, x}?")
                if north_west_array[y, x] == default_value:
                    print(f"7Why is this still default value at {y, x}?")
                if north_east_array[y, x] == default_value:
                    print(f"8Why is this still default value at {y, x}?")

                # If an array has no jump points, it means it should have no obstacles and is of rectangular shape
                if len(jump_points) != 0:
                    # Each point/node needs at least one connection to a jump point
                    if max(east_array[y, x], north_array[y, x], west_array[y, x], south_array[y, x], south_east_array[y, x], south_west_array[y, x], north_west_array[y, x], north_east_array[y, x]) <= 0:
                        no_connection_count += 1
                        no_connection.append((y, x))
                        print(f"Value at {y, x} has no connection to a jump point, what to do? {[east_array[y, x], north_array[y, x], west_array[y, x], south_array[y, x], south_east_array[y, x], south_west_array[y, x], north_west_array[y, x], north_east_array[y, x]]}")
                        # print(f"Left point {y, x-1}: {[east_array[y, x-1], north_array[y, x-1], west_array[y, x-1], south_array[y, x-1], south_east_array[y, x-1], south_west_array[y, x-1], north_west_array[y, x-1], north_east_array[y, x-1]]}")

                    # We dont want to have default values on any point
                    if any(val[y, x] == default_value for val in [east_array, north_array, west_array, south_array, south_east_array, south_west_array, north_west_array, north_east_array]):
                        print(f"Value at {y, x} has one direction in array still set to default!")
        if no_connection_count:
            print(f"{no_connection_count} points have no connection to a jump point")

    # return [jump_points, no_connection_count, east_array, north_array, west_array, south_array, south_east_array, south_west_array, north_west_array, north_east_array]
    return [jump_points, directions_greater_zero, directions_less_zero, east_array, north_array, west_array, south_array, south_east_array, south_west_array, north_west_array, north_east_array]







# @profile
def jps_search(start: Tuple[int, int], goal: Tuple[int, int], array: np.ndarray, precomputed_arrays: List[Union[set, np.ndarray]], debug=True):
    """ Input values:
    start: a tuple (y, x)
    goal: a tuple(y, x)
    array: a numpy array with shape (height, width)"""
    jump_points: Set[Tuple[int, int]]
    jump_points, directions_greater_zero, directions_less_zero, east_array, north_array, west_array, south_array, south_east_array, south_west_array, north_west_array, north_east_array = precomputed_arrays

    height, width = array.shape

    # directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    diagonals = {(1, 1), (1, -1), (-1, 1), (-1, -1)}
    direction_dict_array = {
        (0, 1): east_array,
        (-1, 0): north_array,
        (0, -1): west_array,
        (1, 0): south_array,
        (1, 1): south_east_array,
        (1, -1): south_west_array,
        (-1, -1): north_west_array,
        (-1, 1): north_east_array,
    }
    opposite_direction_dict_array = {
        (0, 1): west_array,
        (-1, 0): south_array,
        (0, -1): east_array,
        (1, 0): north_array,
        (1, 1): north_west_array,
        (1, -1): north_east_array,
        (-1, -1): south_east_array,
        (-1, 1): south_west_array,
    }

    # Visited nodes, with values equal to distance to start
    # data_type = np.uint32
    # default_max = np.iinfo(data_type).max
    # dist_to_start_dict = np.full_like(array, default_max, dtype=data_type)
    # print(dist_to_start_dict[0, 0])

    sqrt2 = 2**0.5

    # Heapq, contains all points that need to be explored (y, x, direction_tuple)
    open_list = []
    open_dict = {}
    dist_to_start_dict = {}
    came_from = {}
    closed_set = set()
    heuristic = heuristic_euclidean
    heapq.heappush(open_list, (heuristic(start, goal), start))
    open_dict = {start: 0}
    dist_to_start_dict = {start: 0}

    print(start, goal, open_list)
    t0 = time.time()

    def generate_path(p1, p2):
        if debug:
            t1 = time.time()
            print(f"Path found!\nTime: {t1-t0} s\nOpen list: {len(open_list)}\nClosed list: {len(closed_set)}\nDist to start: {dist_to_start_dict[p2]}")
        current = p2
        path = []
        while current != p1:
            path.append(current)
            current = came_from[current]
        path.append(p1)
        path.reverse()
        return path

    while open_list:
        current = heapq.heappop(open_list)[1]

        """ Using pure python:  
        Time: 0.0005013942718505859
        Open list: 98
        Closed list: 25
        Dist to start: 179.29646455628173
        
        470 jump points, time taken: 0.8718829154968262s
        Time: 0.0005016326904296875
        Open list: 95
        Closed list: 22
        Dist to start: 136.5269119345812
        """
        for dir in directions_greater_zero[current]:
            # Towards this direction lies a jump point
            direction_value = direction_dict_array[dir][current]
            target_node = (current[0] + direction_value * dir[0], current[1] + direction_value * dir[1])
            # Check if target node is in closed_set, which means we already visited that one (prevents going in a circle)
            if target_node not in closed_set:
                distance_to_start = dist_to_start_dict[current] + (sqrt2 * direction_value if dir in diagonals else direction_value)
                # Check if target node is already in openlist, and if it is, then check if current calculated distance to start is lower than the previously calculated distance on the same node/point
                if distance_to_start < dist_to_start_dict.get(target_node, math.inf):
                    came_from[target_node] = current
                    dist_to_start_dict[target_node] = distance_to_start
                    if target_node == goal:
                        return generate_path(start, goal)
                    distance_to_goal = heuristic(target_node, goal)
                    total_distance = distance_to_start + distance_to_goal
                    heapq.heappush(open_list, (total_distance, target_node))
                    # open_dict[target_node] = distance_to_start



        for dir in directions_less_zero[current]:
            # Towards this direction lies a wall or map bound
            """ S: start, E: End, X: Wall, J: Jump Point
            OOXE
            SOXO
            OJOJ
            Here, the point in the top right direction from the start has to be considered
            """
            vert_dist = goal[0] - current[0]
            hori_dist = goal[1] - current[1]
            if dir in diagonals and dir == (sign(vert_dist), sign(hori_dist)):
                direction_value = direction_dict_array[dir][current]
                min_distance = min(abs(vert_dist), abs(hori_dist))
                if - direction_value >= min_distance:
                    target_node = (current[0] + min_distance * dir[0], current[1] + min_distance * dir[1])
                    distance_to_start = dist_to_start_dict[current] + sqrt2 * direction_value
                    came_from[target_node] = current
                    dist_to_start_dict[target_node] = distance_to_start
                    if target_node == goal:
                        return generate_path(start, goal)
                    distance_to_goal = heuristic(target_node, goal) # TODO: Goal should just lie horizontally or vertically from here!
                    total_distance = distance_to_start + distance_to_goal
                    heapq.heappush(open_list, (total_distance, target_node))

            elif dir not in diagonals and dir == (sign(vert_dist), sign(hori_dist)):
                """ If the target lies horizontally or vertically, check if the path to goal is clear
                If path is clear: path to goal is found
                Else: continue with next point in open_list                
                """
                direction_value = direction_dict_array[dir][current]
                max_distance = max(abs(vert_dist), abs(hori_dist))
                if - direction_value >= max_distance:
                    distance_to_start = dist_to_start_dict[current] + max_distance
                    came_from[goal] = current
                    dist_to_start_dict[goal] = distance_to_start
                    return generate_path(start, goal)

        closed_set.add(current)




if __name__ == "__main__":
    pathing_grid = np.load("numpy_placement_better.npy")
    pathing_grid[pathing_grid == 0] = 9 # Turn all 0 values to 9 values which are walls
    pathing_grid[pathing_grid < 9] = 1 # Turn all other values to 1
    expansions = np.load("numpy_expansions.npy")
    height, width = len(pathing_grid), len(pathing_grid[0])
    print(height, width, pathing_grid.shape)

    t0 = time.time()
    precomputed = jps_precompute(pathing_grid, wall=9)
    t1 = time.time()
    print(f"{len(precomputed[0])} jump points, time taken: {t1-t0} s")
    # print(f"{len(precomputed[0])} jump points, time taken: {round(t1-t0, 3)}s, jump points: {precomputed[0]}")
    np.save("jump_points", list(precomputed[0]))
    # np.save("no_connection", list(a[1]))


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
    spawn1_correct = (40, 30)
    spawn2_correct = (150, 140)

    result = jps_search(spawn1_correct, spawn2_correct, pathing_grid, precomputed, debug=True)
    print(f"Path: {result}")

    # TODO: Path smoothener

    exit()
