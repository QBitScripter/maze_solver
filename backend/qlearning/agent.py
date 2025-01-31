import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.input_dim = env.size * env.size
        self.output_dim = 4
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.01

        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = 64

        # Neural Networks
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(range(self.output_dim))
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values
        current_q = self.policy_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.discount_factor * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, episodes=1000, max_steps=100):
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done and max_steps > 0:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                self.replay()

                state = next_state
                total_reward += reward
                max_steps -= 1

            self.update_target_network()
            self.exploration_rate = max(self.exploration_rate * self.exploration_decay, self.min_exploration_rate)
            print(f"Episode {episode}/{episodes}, Exploration Rate: {self.exploration_rate}, Total Reward: {total_reward}")
            if episode % 100 == 0:
                print(f"Episode {episode} complete")
    def get_solution_path(self):
        """
        After training, extract the solution path by following the learned policy.
        This simulates the agent's movement from start to goal using the trained model.
        """
        state = self.env.start  # Starting position
        path = [state]

        while state != self.env.goal:
            action = self.choose_action(state)  # Get the action from the policy
            next_state, _, done = self.env.step(action)  # Take the action
            path.append(next_state)  # Add the new state to the path
            state = next_state  # Update the state
            if done:  # If goal is reached, break
                break

        return path
