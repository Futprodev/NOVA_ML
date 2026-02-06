from env.nav2_rl_env import Nav2RLEnv
from agent.dqn_agent import DQNAgent
from utils.pot_utils import generate_grid_pots_auto, generate_random_pots
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

MAP_YAML = "maps/room2.yaml"
N_POTS   = 6
EPISODES = 8000

dummy_pots = [(0,0)] * N_POTS

# Initialize environment and agent
env = Nav2RLEnv(MAP_YAML, dummy_pots, noise=0.1)

# Generate random pot positions
env.pot_positions = generate_random_pots(env.map, n_pots=N_POTS)
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

state_dim = env.state_dim
action_dim = env.action_dim

agent = DQNAgent(state_dim, action_dim)

rewards_per_ep = []

save_dir = "./trained_model"
os.makedirs(save_dir, exist_ok=True)

# Training loop
for ep in range(EPISODES):
    # reset mask but keep same pot locations
    state = env.reset()

    total_reward = 0.0
    steps = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        agent.memory.push(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            agent.update_target()
            break

    rewards_per_ep.append(total_reward)
    print(f"[EP {ep}] Reward = {total_reward:.2f}  Steps = {steps}  Eps = {agent.epsilon:.3f}")

    # respawn pots every few episodes (important for generalization)
    if ep % 3 == 0 and ep > 0:
        env.pot_positions = generate_random_pots(env.map, N_POTS)
    if ep % 2 == 0 and ep > 0:
        env.pot_positions = generate_grid_pots_auto(
            env.map,
            rows=3,
            cols=2,
            spacing=1.0,
            min_clearance=1.0
        )

    if ep % 2000 == 0:
        torch.save(agent.policy.state_dict(), f"{save_dir}/checkpoint_ep{ep}.pth")
        print(f"Checkpoint saved at episode {ep}")

# Save model
save_path = f"{save_dir}/dqn_nav2_policy.pth"
torch.save(agent.policy.state_dict(), save_path)

print(f"Model saved to {save_path}")
plt.plot(rewards_per_ep)
plt.title("Training Reward Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()