
import numpy as np

import pathfinding
import time

print(pathfinding.__file__)
from pathfinding import astar




if __name__ == "__main__":
    pathing_grid = np.load("numpy_placement_better.npy")
    pathing_grid[pathing_grid == 0] = 9 # Turn all 0 values to 9 values which are walls
    pathing_grid[pathing_grid < 9] = 1 # Turn all other values to 1
    expansions = np.load("numpy_expansions.npy")
    height, width = len(pathing_grid), len(pathing_grid[0])
    print(height, width, pathing_grid.shape)

    spawn1 = expansions[6] # 35.5, 35.5
    spawn2 = expansions[-1] # 140.5 140.5
    # print(expansions)
    # print(spawn1)

    # print(spawn2)
    spawn1 = (35.5, 35.5)
    spawn2 = (140.5, 140.5)

    spawn1_correct = int(height - 1 - spawn1[1]+0.5), int(spawn1[0]-0.5)
    spawn2_correct = int(height - 1 - spawn2[1]+0.5), int(spawn2[0]-0.5)

    # Testing top left to bottom right
    spawn1_correct = (40, 30)
    spawn2_correct = (150, 140)

    nmap = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [9,9,9,9,9,9,9,9,9,9,9,9,1,9],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,9],
        [9,1,9,9,9,9,9,9,9,9,9,9,9,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,9],
        [9,9,9,9,9,9,9,9,9,9,9,9,1,9],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [9,1,9,9,9,9,9,9,9,9,9,9,9,9],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [9,9,9,9,9,9,9,9,9,9,9,9,1,9],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1]], dtype=np.int32)

    # Testing top left to bottom right
    spawn1_correct = (40, 30)
    spawn2_correct = (150, 140)

    # amount = 100
    solution, cost, closed_set = astar(spawn1_correct, spawn2_correct, pathing_grid, diagonal=1, debug=True)
    t0 = time.time()
    # for i in range(amount):
    #     # solution, cost, closed_set = astar((0, 0), (10, 13), nmap, diagonal=True)
    #     solution, cost, closed_set = astar(spawn1_correct, spawn2_correct, pathing_grid, diagonal=True)
    solution, cost, closed_set = astar(spawn1_correct, spawn2_correct, pathing_grid, diagonal=1, debug=True)
    t1 = time.time()
    np.save("path.npy", solution) # list(closed_set))
    print(f"Time: {t1-t0}\nTotal path length: {cost}\nPath: {solution}")
    print(len(closed_set), closed_set)


    # t0 = time.time()
    # for i in range(amount):
    #     solution, cost, closed_set = astar(spawn1_correct, spawn2_correct, pathing_grid, diagonal=False)
    #     # solution, cost, closed_set = astar((0, 0), (10, 13), nmap, diagonal=False)
    # t1 = time.time()
    # print((t1-t0) / amount, cost, solution)
    # print(len(closed_set), closed_set)