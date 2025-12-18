import matplotlib.pyplot as plt
import numpy as np

def show_env(nav2_map, robot_xy, pots, mask, path=None):
    """
    Full environment visualization:
    - Inflation costmap
    - Pots (visited/unvisited)
    - Robot position
    - A* path
    """

    plt.figure(figsize=(7,7))
    plt.title("Nav2 RL Environment")

    # Show map
    plt.imshow(nav2_map.inflation_costmap, cmap="inferno")

    # Draw pots
    for i, (x, y) in enumerate(pots):
        gy, gx = nav2_map.world_to_grid(x, y)

        if mask[i] == 1.0:       # unvisited
            color = "cyan"
            marker = "o"
        else:                    # visited
            color = "gray"
            marker = "x"

        plt.scatter(gx, gy, c=color, marker=marker, s=70)

    # Draw robot
    ry, rx = nav2_map.world_to_grid(robot_xy[0], robot_xy[1])
    plt.scatter(rx, ry, c="lime", marker="s", s=90, label="Robot")

    # Draw A* path if available
    if path is not None:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        plt.plot(xs, ys, c="yellow", linewidth=2)

    plt.show()
