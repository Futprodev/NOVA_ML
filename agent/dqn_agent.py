import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.replay_buffer import ReplayBuffer
from model.dqn import DQN

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.98
        self.epsilon = 1.0
        self.eps_min = 0.1
        self.eps_decay = 0.998
        self.batch_size = 64

        self.policy = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)
        self.target.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.target.net[-1].out_features)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        qvals = self.policy(state_t)
        return torch.argmax(qvals).item()
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q(s, a)
        q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q target = r + gamma * max Q(s', a')
        next_q = self.target(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update epsilon
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())