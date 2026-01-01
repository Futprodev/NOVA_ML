from sys import path
import numpy as np
from heapq import heappop, heappush
from maps.map_builder import Nav2Map

class Nav2RLEnv:
    def __init__(self, map_yaml, pot_positions, noise=0.0):
        self.map = Nav2Map(map_yaml)
        self.pot_positions = pot_positions
        self.n_pots = len(self.pot_positions)
        self.noise = noise
        
        self.mask = np.ones(self.n_pots, dtype=np.float32)

        self.robot_world = np.array([
            (self.map.origin[0] + self.map.width * self.map.resolution) / 2,
            (self.map.origin[1] + self.map.height * self.map.resolution) / 2
        ], dtype=np.float32)

        self.state_dim = 2 + 2 * self.n_pots + self.n_pots
        self.action_dim = self.n_pots  # Move to pot 0..n_pots-1
        self.last_path = None

        self.step_count = 0

        self.last_action = None

    def reset(self):
        self.mask[:] = 1.0
        self.step_count = 0
        self.last_action = None
        return self._get_state()

    def step(self, action):

        self.step_count += 1

        # not moving
        if self.last_action is not None and action == self.last_action:
            reward = -1000

        # pot already collected
        if self.mask[action] == 0.0:
            reward = -1000
            done = False

            # Terminate if too many invalid steps
            if self.step_count >= self.n_pots * 2:
                reward -= 100

            return self._get_state(), reward, done

        start_xy = self.robot_world
        goal_xy  = self.pot_positions[action]

        path_cost = self._compute_path_cost(start_xy, goal_xy)
        path_cost = min(path_cost, 10)  # cap max cost
        path_cost *= (1.0 + np.random.uniform(-self.noise, self.noise))

        self.robot_world = np.array(goal_xy, dtype=np.float32)
        self.mask[action] = 0.0

        reward = 0
        reward -= path_cost * 0.1            # scaled path penalty
        reward += 300                        # new pot bonus
        reward -= 1                           # small step penalty

        remaining = np.sum(self.mask)
        reward -= remaining * 2              # penalty for unfinished job

        if remaining > 0:
            distances = [
                np.linalg.norm(self.robot_world - np.array(self.pot_positions[i]))
                for i in range(self.n_pots) if self.mask[i] == 1.0
            ]
            min_distance = min(distances)
            reward -= min_distance * 0.05      # distance to closest pot penalty

        done = (remaining == 0)

        if done:
            reward += 1000
        else:
            if self.step_count >= self.n_pots * 10:
                done = True
                reward -= 100

        self.last_action = action

        return self._get_state(), reward, done
    
    def _get_state(self):
        pots_flat = np.array(self.pot_positions).flatten().astype(np.float32)
        return np.concatenate([self.robot_world, pots_flat, self.mask]).astype(np.float32)
    
    def _compute_path_cost(self, start_xy, goal_xy):
        start_grid = self.map.world_to_grid(start_xy[0], start_xy[1])
        goal_grid  = self.map.world_to_grid(goal_xy[0], goal_xy[1])

        path, cost = self._a_star(start_grid, goal_grid)
        self.last_path = path
        return cost * self.map.resolution
    
    def _a_star(self, start, goal):
        from heapq import heappush, heappop

        # Manhattan heuristic
        h = lambda a, b: abs(b[0] - a[0]) + abs(b[1] - a[1])

        open_set = []
        heappush(open_set, (0, start))

        came_from = {start: None}
        g_score = {start: 0}

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, current = heappop(open_set)

            # Goal reached → reconstruct path
            if current == goal:
                # Reconstruct path (goal → start)
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()

                return path, g_score[current]

            # Explore neighbors
            for dx, dy in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                nb = (nx, ny)

                # Out of bounds
                if not (0 <= nx < self.map.height and 0 <= ny < self.map.width):
                    continue

                # Collision in inflated costmap
                cell_cost = self.map.inflation_costmap[nx, ny]
                if cell_cost == 255:
                    continue

                move_cost = 1 + (cell_cost / 255.0) * 3.0
                new_cost = g_score[current] + move_cost

                if nb not in g_score or new_cost < g_score[nb]:
                    g_score[nb] = new_cost
                    priority = new_cost + h(nb, goal)
                    heappush(open_set, (priority, nb))
                    came_from[nb] = current

        # unreachable goal
        return None, float("inf")
