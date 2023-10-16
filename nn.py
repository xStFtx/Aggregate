import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import namedtuple, deque
import random
from qmath import Quaternion

class QuaternionLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QuaternionLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        # Simplifying this operation to be a standard linear operation for now.
        return torch.mm(x, self.weight) + self.bias

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(DQN, self).__init__()
        self.fc1 = QuaternionLinear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = QuaternionLinear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)
    
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
    try:
        # Check if state is a tuple (as the error suggests) and extract the numpy array
        if isinstance(state, tuple) and len(state) > 0:
            state = state[0]
        
        # Convert state to a numpy array and flatten it
        state_array = np.array(state).flatten()
        
        # Check if it's already a 1D array
        if len(state_array.shape) > 1:
            raise ValueError(f"Unexpected state shape: {state_array.shape}. Ensure the environment returns a simple array.")
        
        # Reshape to (1, state_dim)
        return np.expand_dims(state_array, 0)
    
    except ValueError as e:
        print(f"Error converting state: {state}. Type: {type(state)}")
        raise e

    except Exception as e:
        print(f"Unexpected error with state: {state}. Type: {type(state)}")
        raise e


def train_dqn(env_name="CartPole-v0", episodes=100, batch_size=32, capacity=10000, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128

    model = DQN(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    memory = ReplayMemory(capacity)
    steps_done = 0

    for episode in range(episodes):
        state = preprocess_state(env.reset())
        done = False
        total_loss = 0

        while not done:
            # Epsilon-Greedy Exploration
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1
            if random.random() > epsilon:
                with torch.no_grad():
                    # Disable batch normalization during evaluation
                    model.eval()
                    q_values = model(torch.FloatTensor(state))
                    action = q_values.argmax().item()
            else:
                action = random.randrange(action_dim)

            result = env.step(action)
            next_state, reward, done, _ = result[:4]
            next_state = preprocess_state(next_state)
            memory.push(state, action, next_state, reward, done)

            state = next_state

            if len(memory) < batch_size:
                continue

            # Sample from memory
            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([torch.FloatTensor(s) for s in batch.next_state if s is not None])
            state_batch = torch.cat([torch.FloatTensor(s) for s in batch.state])
            action_batch = torch.LongTensor(batch.action)
            reward_batch = torch.FloatTensor(batch.reward)

            # Set the model back to training mode
            model.train()

# Inside the training loop
            q_values = model(state_batch)

            # Reshape q_values to match the shape of target_q_values
            q_values = q_values.gather(1, action_batch.unsqueeze(1))

            next_q_values = torch.zeros(batch_size)
            next_q_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()
            next_q_values = next_q_values.unsqueeze(1)

            target_q_values = (next_q_values * 0.99) + reward_batch.view(-1, 1)



            loss = criterion(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Episode: {episode}, Loss: {total_loss}")

if __name__ == "__main__":
    train_dqn()

