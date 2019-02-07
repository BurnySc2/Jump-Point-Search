"""
This file was just made to test if there is any speedups by cython
1) Run make_cython.bat to convert to C code and then back to python code as .pyd file
2) Run this test.py file to check if the .pyd file gave any speedups
"""

import numpy as np
import jps
print(jps.__file__)
# import jps_nc
# print(jps_nc.__file__)
import time, math

from jps import jps_precompute, jps_search
# from jps_nc import jps_search



if __name__ == "__main__":
    pathing_grid = np.load("numpy_placement_better.npy")
    pathing_grid[pathing_grid == 0] = 9 # Turn all 0 values to 9 values which are walls
    pathing_grid[pathing_grid < 9] = 1 # Turn all other values to 1
    expansions = np.load("numpy_expansions.npy")
    height, width = len(pathing_grid), len(pathing_grid[0])
    print(height, width, pathing_grid.shape)

    t0 = time.time()
    precomputed = jps_precompute(pathing_grid, wall_value=9)
    t1 = time.time()
    print(f"{len(precomputed[0])} jump points, time taken: {round(t1-t0, 3)}s, jump points: {precomputed[0]}")
    # np.save("jump_points", list(precomputed[0]))
    # # np.save("no_connection", list(a[1]))


    spawn1 = expansions[6] # 35.5, 35.5
    spawn2 = expansions[-1] # 140.5 140.5
    spawn1 = (35.5, 35.5)
    spawn2 = (140.5, 140.5)
    # print(expansions)
    # print(spawn1)
    # print(spawn2)

    # # Pre compute once for numba to not screw with results
    # heuristic_manhattan(spawn1, spawn2)
    # heuristic_euclidean(spawn1, spawn2)
    # heuristic_diagonal(spawn1, spawn2)

    spawn1_correct = int(height - 1 - spawn1[1]+0.5), int(spawn1[0]-0.5)
    spawn2_correct = int(height - 1 - spawn2[1]+0.5), int(spawn2[0]-0.5)

    # Testing top left to bottom right
    spawn1_correct = (40, 30)
    spawn2_correct = (150, 140)

    test = np.array([
        [0, 9, 0, 0, 0, 0],
        [0, 9, 9, 9, 0, 0],
        [0, 9, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    # print(test.shape)
    p1 = (0, 0)
    p2 = (0, 2)

    # result = jps_search(p1, p2, test, wall_value=9)
    t0 = time.time()
    result = jps_search(spawn1_correct, spawn2_correct, pathing_grid, precomputed, debug=False)
    t1 = time.time()

    print(f"Time: {t1-t0}, Path: {result}")

    np.save("path", result)

    # TODO: Path smoothener

    exit()