# standard library imports
from typing import List
# third party imports
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_layers: List[int] = [16, 16, 16, 8]):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module("input", nn.Linear(obs_space, hidden_layers[0]))
        self.net.add_module("relu0", nn.ReLU())
        for i in range(1, len(hidden_layers)):
            self.net.add_module(f"hidden{i}", nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.net.add_module(f"relu{i}", nn.ReLU())
        self.net.add_module("output", nn.Linear(hidden_layers[-1], action_space))

    def forward(self, x):
        return self.net(x)


def main():
    env = gym.make('CartPole-v1')
    obs_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    q_network = QNetwork(obs_space_size, action_space_size)
    target_network = QNetwork(obs_space_size, action_space_size)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters())
    mse_loss = nn.MSELoss()

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    episodes = 1000
    update_target_every = 20
    max_timesteps: int = 1000

    for episode in range(episodes):
        state = env.reset()[0]
        done: bool = False
        n_timesteps: int = 0

        while not done and n_timesteps < max_timesteps:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = q_network(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, *_ = env.step(action)
            reward = reward if not done else -10

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                next_q_values = target_network(next_state_tensor)
            
            target_q_value = reward + (gamma * torch.max(next_q_values))

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            q_value = q_values[0, action]

            loss = mse_loss(q_value, target_q_value.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

            if done:
                break
            n_timesteps += 1

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        # Evaluation during training
        if episode % update_target_every == 0:
            target_network.load_state_dict(q_network.state_dict())

            eval_state = env.reset()[0]
            eval_done = False
            eval_rewards = 0
            n_timesteps = 0
            while not eval_done and n_timesteps < max_timesteps:
                with torch.no_grad():
                    eval_state_tensor = torch.FloatTensor(eval_state).unsqueeze(0)
                    eval_q_values = q_network(eval_state_tensor)
                    eval_action = torch.argmax(eval_q_values).item()
                eval_state, eval_reward, eval_done, *_ = env.step(eval_action)
                eval_rewards += eval_reward
                n_timesteps += 1
            print(f"Episode: {episode:3}, Epsilon: {epsilon:.3f}, Eval Reward: {eval_rewards:3}")

    # Evaluation after training
    eval_state = env.reset()[0]
    eval_done = False
    eval_rewards = 0
    n_timesteps = 0
    while not eval_done and n_timesteps < max_timesteps:
        with torch.no_grad():
            eval_state_tensor = torch.FloatTensor(eval_state).unsqueeze(0)
            eval_q_values = q_network(eval_state_tensor)
            eval_action = torch.argmax(eval_q_values).item()

        eval_state, eval_reward, eval_done, *_ = env.step(eval_action)
        eval_rewards += eval_reward
        n_timesteps += 1


    print(f"Final Evaluation Reward: {eval_rewards}")


if __name__ == "__main__":
    main()