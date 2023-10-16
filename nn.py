import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import namedtuple, deque
import random

class QuaternionLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QuaternionLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(DuelingDQN, self).__init__()
        self.fc1 = QuaternionLinear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Switch to LayerNorm
        self.fc_value = QuaternionLinear(hidden_dim, 1)
        self.fc_advantage = QuaternionLinear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln1(torch.relu(self.fc1(x)))  # Use LayerNorm
        x = self.dropout(x)
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        return value + advantage - advantage.mean()

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def preprocess_state(state):
    if isinstance(state, tuple) and len(state) > 0:
        state = state[0]
    state_array = np.array(state).flatten()
    if len(state_array.shape) > 1:
        raise ValueError(f"Unexpected state shape: {state_array.shape}. Ensure the environment returns a simple array.")
    return np.expand_dims(state_array, 0)

def train_dqn(env_name="CartPole-v1", episodes=1000, batch_size=32, capacity=10000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=200, target_update=10, gamma=0.99, learning_rate=0.001, max_steps=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128

    model = DuelingDQN(state_dim, hidden_dim, action_dim)
    target_model = DuelingDQN(state_dim, hidden_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  # Reduce LR every 100 episodes
    criterion = nn.SmoothL1Loss()

    memory = ReplayMemory(capacity)

    for episode in range(episodes):
        state = preprocess_state(env.reset())
        total_loss = 0
        episode_steps = 0
        done = False

        while not done and episode_steps < max_steps:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * episode / epsilon_decay)
            if random.random() > epsilon:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state))
                    action = q_values.argmax().item()
            else:
                action = random.randrange(action_dim)

            result = env.step(action)
            next_state, reward, done, _ = result if len(result) == 4 else (result[0], result[1], result[2], {})

            next_state = preprocess_state(next_state)
            memory.push(state, action, next_state, reward, done)
            state = next_state

            if len(memory) <= batch_size:
                continue

            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([torch.FloatTensor(s) for s in batch.next_state if s is not None])
            state_batch = torch.cat([torch.FloatTensor(s) for s in batch.state])
            action_batch = torch.LongTensor(batch.action)
            reward_batch = torch.FloatTensor(batch.reward)

            q_values = model(state_batch).gather(1, action_batch.unsqueeze(1))

            next_q_values = torch.zeros(batch_size)
            actions_from_online_model = model(non_final_next_states).max(1)[1].detach()
            next_q_values[non_final_mask] = target_model(non_final_next_states).gather(1, actions_from_online_model.unsqueeze(-1)).squeeze(-1)
            target_q_values = (next_q_values.unsqueeze(-1) * gamma) + reward_batch.view(-1, 1)

            loss = criterion(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            episode_steps += 1

        lr_scheduler.step()

        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode: {episode}, Steps: {episode_steps}, Loss: {total_loss}")

if __name__ == "__main__":
    train_dqn()
