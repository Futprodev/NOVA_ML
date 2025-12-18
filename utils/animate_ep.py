import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

COLORS = ["yellow", "cyan", "lime", "magenta", "orange", "white"]


def animate_episode(nav2_map, robot_history, pot_positions, pot_order, path_history, reward_history, save_gif=False):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    COLORS = ["yellow", "cyan", "lime", "magenta", "orange", "white"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(nav2_map.inflation_costmap, cmap="inferno")

    # Legend showing pot order
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=3, label=f"Step {i+1}: Pot {pot_idx}")
        for i, pot_idx in enumerate(pot_order)
    ]

    ax.legend(handles=legend_elements, loc="upper right")

    frames = len(robot_history)

    def update(frame):
        ax.clear()
        ax.imshow(nav2_map.inflation_costmap, cmap="inferno")

        # Draw pots
        for i, (x, y) in enumerate(pot_positions):
            gy, gx = nav2_map.world_to_grid(x, y)
            ax.scatter(gx, gy, c="cyan", marker="o", s=80)

        # Draw A* paths so far
        for step in range(frame + 1):
            if path_history[step] is not None:
                xs = [p[1] for p in path_history[step]]
                ys = [p[0] for p in path_history[step]]
                ax.plot(xs, ys, color=COLORS[step % len(COLORS)], linewidth=3)

        # Robot position
        rx, ry = robot_history[frame]
        gy, gx = nav2_map.world_to_grid(rx, ry)
        ax.scatter(gx, gy, c="lime", marker="s", s=120)

        # Draw reward text
        ax.text(
            0.02, 0.02,
            f"Reward: {reward_history[frame]:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            color="white",
            bbox=dict(facecolor="black", alpha=0.4)
        )

        # Draw legend again
        ax.legend(handles=legend_elements, loc="upper right")

        return ax

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1200, repeat=False)

    if save_gif:
        ani.save("episode_animation.gif", writer="pillow", fps=1)
        print("GIF saved: episode_animation.gif")

    plt.show()
