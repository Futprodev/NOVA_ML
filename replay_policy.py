from env.nav2_rl_env import Nav2RLEnv
from utils.pot_utils import generate_random_pots, generate_grid_pots_auto
from agent.dqn_agent import DQNAgent
from utils.animate_ep import animate_episode
import torch
import numpy as np

MAP_YAML = "maps/room2.yaml"
N_POTS = 6

# Generate pots before env creation
tmp_env = Nav2RLEnv(MAP_YAML, [(0,0)]*N_POTS, noise=0)

pots =  generate_grid_pots_auto(
            tmp_env.map,
            rows=3,
            cols=2,
            spacing=1.5,
            min_clearance=1.5
        )

# pots = generate_random_pots(tmp_env.map, N_POTS)

# Real environment
env = Nav2RLEnv(MAP_YAML, pots, noise=0)

agent = DQNAgent(env.state_dim, env.action_dim)
agent.policy.load_state_dict(torch.load("./trained_model/dqn_nav2_policy.pth"))

robot_history = []
path_history = []
pot_order = []
reward_history = []

state = env.reset()
done = False

while not done:
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qvals = agent.policy(state_t)
    action = int(torch.argmax(qvals).item())

    pot_order.append(action)
    next_state, reward, done = env.step(action)

    robot_history.append(env.robot_world.copy())
    path_history.append(env.last_path)
    reward_history.append(reward)

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
