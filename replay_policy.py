from env.nav2_rl_env import Nav2RLEnv
from utils.pot_utils import generate_random_pots, generate_grid_pots_auto
from agent.dqn_agent import DQNAgent
from utils.debug_visuals import show_env
from utils.animate_ep import animate_episode
import torch
import numpy as np

MAP_YAML = "maps/room2.yaml"
N_POTS = 6

# First create env with dummy pots 
dummy = [(0.0, 0.0)] * N_POTS
env = Nav2RLEnv(MAP_YAML, dummy, noise=0)

# Now generate real pots using the map inside env
#env.pot_positions = generate_random_pots(env.map, N_POTS)
env.pot_positions = generate_grid_pots_auto(env.map, rows=3, cols=2, spacing=1.0, min_clearance=1.0)
env.n_pots = N_POTS
env.action_dim = N_POTS
env.mask = np.ones(N_POTS, dtype=np.float32)

# Load agent
agent = DQNAgent(env.state_dim, env.action_dim)
agent.policy.load_state_dict(torch.load("./trained_model/dqn_nav2_policy.pth"))

robot_history = []
path_history = []
pot_order = []

reward_history = []

state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    pot_order.append(action)
    next_state, reward, done = env.step(action)

    robot_history.append(env.robot_world.copy())
    path_history.append(env.last_path)

    reward_history.append(reward)

    """
    Frame to visualize
    show_env(
        env.map,
        env.robot_world,
        env.pot_positions,
        env.mask,
        path=env.last_path
    )
    """

    state = next_state

animate_episode(
    env.map,
    robot_history,
    env.pot_positions,
    pot_order,
    path_history,
    reward_history,
    save_gif=True
)