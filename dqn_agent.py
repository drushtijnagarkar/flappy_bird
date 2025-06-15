import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 1. Neural Network for Q-learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
             nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 2. Replay Buffer
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 3. DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.990
        self.update_steps = 0
        self.update_target_every = 100

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # explore
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()  # exploit

    def push(self, *args):
        self.memory.push(*args)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)
        max_next_q = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        expected_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        self.update_steps += 1
        if self.update_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
