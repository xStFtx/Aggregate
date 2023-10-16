import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Assuming you've kept the Quaternion class in main.py
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = QuaternionLinear(input_dim, hidden_dim)
        self.fc2 = QuaternionLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
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


def train_dqn(env_name="CartPole-v0", episodes=100):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128

    model = DQN(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for episode in range(episodes):
        state = preprocess_state(env.reset())

        done = False
        total_loss = 0

        while not done:
            q_values = model(torch.FloatTensor(state))
            action = np.argmax(q_values.detach().numpy())
            result = env.step(action)
            print(result)
            next_state, reward, done, _, _ = result

            next_state = preprocess_state(next_state)

            # Obtain Q-values for the next state
            next_q_values = model(torch.FloatTensor(next_state))

            # Get max Q-value for the next state 
            max_next_q_value = torch.max(next_q_values).item()

            # Calculate the target Q-value
            target_q_value = reward + (0.99 * max_next_q_value) if not done else reward
            target_q_values = q_values.clone().detach()
            target_q_values[0][action] = target_q_value

            optimizer.zero_grad()
            loss = criterion(q_values, target_q_values)
            loss.backward()
            optimizer.step()

            state = next_state
            total_loss += loss.item()

        print(f"Episode: {episode}, Loss: {total_loss}")

if __name__ == "__main__":
    train_dqn()
