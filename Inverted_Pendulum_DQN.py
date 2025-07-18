import gym
import math
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ENV_NAME = 'Pendulum-v1'  
GAMMA = 0.99              
LR = 0.003              
BATCH_SIZE = 500          
MEMORY_SIZE = 100000   
TAU = 0.01            
EPS_START = 1.0        
EPS_END = 0.01            
EPS_DECAY = 1000         
MAX_EPISODES = 500    
MODEL_PATH = "dqn_pendulum_model.pth"


DISCRETE_ACTIONS = torch.tensor(
    [-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0], device=device
)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, env):
        self.env = env
        self.n_actions = len(DISCRETE_ACTIONS)
        self.n_states = env.observation_space.shape[0]

        self.policy_net = DQN(self.n_states, self.n_actions).to(device)
        self.target_net = DQN(self.n_states, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() < eps_threshold:
            action_idx = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                action_idx = self.policy_net(state).max(1)[1].item()
        return action_idx

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # expected_state_action_values = (expected_state_action_values - expected_state_action_values.mean()) / (expected_state_action_values.std() + 1e-7)
        expected_state_action_values+=1
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.squeeze(), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)


def main():
    env = gym.make(ENV_NAME, render_mode=None)
    agent = Agent(env)
    scores = []

    for i_episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.select_action(state)
            torque = [DISCRETE_ACTIONS[action_idx].item()]
            next_state, reward, terminated, truncated, _ = env.step(np.array(torque))
            done = terminated or truncated
            total_reward += reward

            reward_tensor = torch.tensor([reward], device=device)
            next_state_tensor = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor([[action_idx]], device=device, dtype=torch.long)

            agent.memory.push(state, action_tensor, reward_tensor, next_state_tensor, done)
            state = next_state_tensor

            agent.optimize_model()

        scores.append(total_reward)
        print(f"Episode {i_episode}\tReward: {total_reward:.2f}")

    torch.save(agent.policy_net.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    env.close()


    plt.figure(figsize=(10,5))
    plt.plot(scores, label='Episode Reward')
    plt.plot(np.convolve(scores, np.ones(10)/10, mode='valid'), label='Moving Average (10)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Rewards')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("dqn_training_rewards_inv_pend.png")
    plt.close()
    print("Training reward plot saved as dqn_training_rewards_inv_pend.png")

def replay():
    env = gym.make(ENV_NAME, render_mode="human")
    n_states = env.observation_space.shape[0]
    model = DQN(n_states, len(DISCRETE_ACTIONS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    state, _ = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    total_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            action_idx = model(state).max(1)[1].item()
        torque = [DISCRETE_ACTIONS[action_idx].item()]
        next_state, reward, terminated, truncated, _ = env.step(np.array(torque))
        done = terminated or truncated
        total_reward += reward
        state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)

    print(f"Replay Reward: {total_reward:.2f}")
    env.close()

if __name__ == '__main__':
    # main()
    replay()
