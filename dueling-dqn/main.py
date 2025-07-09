import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F
from plot import plot_training_progress
import os

GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.01
REPLAY_MEMORY = 10000
BATCH = 128
LR = 3e-4
UPDATE_STEPS = 4
SAVE_INTERVAL = 10

LOAD = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = 'saved_models/dueling_dqn_model.pt'
render = 'human' if device == torch.device("cpu") else 'rgb_array'

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        # After conv layers: (96->22->10->8) for each dimension
        self.conv_output_size = 64 * 8 * 8  # 4096
        
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(512, 256)
        self.fc_adv = nn.Linear(512, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, 2)

    def forward(self, state):
        # Ensure state is in the right format: (batch_size, channels, height, width)
        if len(state.shape) == 3:  # Single image: (height, width, channels)
            state = state.permute(2, 0, 1).unsqueeze(0)  # -> (1, channels, height, width)
        elif len(state.shape) == 4 and state.shape[3] == 3:  # Batch of images: (batch, height, width, channels)
            state = state.permute(0, 3, 1, 2)  # -> (batch, channels, height, width)
        
        # Normalize pixel values to [0, 1]
        state = state.float() / 255.0
        
        # Convolutional layers
        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten
        x = x.contiguous().view(x.size(0), -1)
        
        # Fully connected layers
        y = self.relu(self.fc1(x))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

def save_param(rewards, episodes, i_ep, QNetork):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save({'dueling-dqn-param': QNetork,
                'rewards': rewards,
                'episodes': episodes,
                'i_ep': i_ep
                }, path)
    
def set_seed(env, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    env = gym.make('CarRacing-v3', domain_randomize=False, continuous=False, render_mode="rgb_array")

    set_seed(env)
    
    onlineQNetwork = QNetwork().to(device)
    targetQNetwork = QNetwork().to(device)
    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

    optimizer = torch.optim.AdamW(onlineQNetwork.parameters(), LR)

    memory_replay = Memory(REPLAY_MEMORY)

    epsilon = INITIAL_EPSILON
    learn_steps = 0
    begin_learn = False

    episode_reward = 0
    rewards = []
    initial_ep = 0

    if LOAD:
        print('Loading model from:', path)
        try:
            checkpoint = torch.load(path, map_location=device)
            rewards = checkpoint['rewards']
            initial_ep = checkpoint['i_ep'] + 1 
            onlineQNetwork.load_state_dict(checkpoint['dueling-dqn-param'])
            print(f'Loaded model at episode {initial_ep}')
        except FileNotFoundError:
            print('No saved model found - starting fresh')
            initial_ep = 0
    else:
        print('Training fresh')
        initial_ep = 0
    
    for epoch in count(initial_ep):
        done = False
        state, _ = env.reset()
        episode_reward = 0
        for time_steps in range(200):
            p = random.random()
            if p < epsilon:
                action = random.randint(0, 1)
            else:
                tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = onlineQNetwork.select_action(tensor_state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated:
                reward += 100
            if np.mean(next_state[:, :, 1]) > 185.0:
                reward -= 0.05
            
            episode_reward += reward
            memory_replay.add((state, next_state, action, reward, done))
            if memory_replay.size() > 128:
                learn_steps += 1
                if learn_steps % UPDATE_STEPS == 0:
                    print("Update at epoch: ", epoch)
                    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
                batch = memory_replay.sample(BATCH, False)
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

                batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
                batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
                batch_action = torch.FloatTensor(np.array(batch_action)).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device)

                with torch.no_grad():
                    onlineQ_next = onlineQNetwork(batch_next_state)
                    targetQ_next = targetQNetwork(batch_next_state)
                    online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
                    y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, online_max_action.long())

                loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epsilon > FINAL_EPSILON:
                    #VEDERE COME DECADE
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            if done:
                rewards.append(episode_reward)
                break
            state = next_state

        if epoch % SAVE_INTERVAL == 0:
            plot_training_progress(scores=rewards, episodes=epoch)
            save_param(rewards, list(range(len(rewards))), epoch, onlineQNetwork.state_dict())
            print('Ep {}\tMoving average score: {:.2f}\tEpsilon: {:.2}\t'.format(epoch, episode_reward, epsilon))
            
if __name__ == "__main__":
    main()