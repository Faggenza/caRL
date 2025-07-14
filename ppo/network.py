import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

torch.set_num_threads(1)


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, action_dim, img_stack=4):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.actor_head = nn.Sequential(
            nn.Linear(256, 100), nn.ReLU(),
            nn.Linear(100, action_dim),
            nn.Softmax(dim=-1)
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        probs = self.actor_head(x)
        return probs, v


class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5

    def __init__(self, action_dim, device,
                 path, transition=None, img_stack=4, gamma=0.99,
                 ppo_epoch=8, buffer_capacity=2000,
                 batch_size=256, clip_param=0.2, learning_rate=1e-4):
        self.ppo_epoch = ppo_epoch
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.clip_param = clip_param
        self.training_step = 0
        self.path = path
        self.device = device
        self.gamma = gamma
        self.net = Net(action_dim, img_stack).double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.net(state)
        dist = Categorical(probs)
        action = dist.sample()
        a_logp = dist.log_prob(action)
        return action.item(), a_logp.item()

    def select_test_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.net(state)
        action = torch.argmax(probs).item()
        return action

    def save_param(self, scores, episodes, i_ep):
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        torch.save({'ppo_net_params_discrete': self.net.state_dict(),
                    'scores': scores,
                    'episodes': episodes,
                    'i_ep': i_ep
                    }, self.path)
        
    def load_param(self):
        if os.path.exists(self.path):
            checkpoint = torch.load(self.path, map_location=self.device)
            self.net.load_state_dict(checkpoint['ppo_net_params_discrete'])
            print(f"Loaded model parameters from {self.path}")
        else:
            print(f"No model found at {self.path}, starting with a new model.")

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1
        
        torch.set_num_threads(1)
        
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.long).to(self.device).view(-1, 1)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            _, target_v = self.net(s_)
            target_v = r + self.gamma * target_v
            _, v = self.net(s)
            adv = (target_v - v).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                probs, value = self.net(s[index])
                dist = Categorical(probs)
                a_logp = dist.log_prob(a[index].squeeze()).view(-1, 1)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(value, target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class AgentGAE(Agent):
    """
    Agent for training with GAE
    """
    max_grad_norm = 0.5

    def __init__(self, action_dim, device,
                 path, transition=None, img_stack=4, gamma=0.99,
                 ppo_epoch=8, buffer_capacity=2000, gae_lambda=0.95,
                 batch_size=256, clip_param=0.2, learning_rate=1e-4):

        super(AgentGAE, self).__init__()
        self.ppo_epoch = ppo_epoch
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.clip_param = clip_param
        self.training_step = 0
        self.path = path
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.net = Net(action_dim, img_stack).double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.net(state)
        dist = Categorical(probs)
        action = dist.sample()
        a_logp = dist.log_prob(action)
        return action.item(), a_logp.item()

    def select_test_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.net(state)
        action = torch.argmax(probs).item()
        return action

    def save_param(self, scores, episodes, i_ep):
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        torch.save({'ppo_net_params_discrete': self.net.state_dict(),
                    'scores': scores,
                    'episodes': episodes,
                    'i_ep': i_ep
                    }, self.path)

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def load_param(self):
        if os.path.exists(self.path):
            checkpoint = torch.load(self.path, map_location=self.device)
            self.net.load_state_dict(checkpoint['ppo_net_params_discrete'])
            print(f"Loaded model parameters from {self.path}")
        else:
            print(f"No model found at {self.path}, starting with a new model.")

    def update(self):
        self.training_step += 1
        
        torch.set_num_threads(1)
        
        states = np.ascontiguousarray(self.buffer['s'])
        actions = np.ascontiguousarray(self.buffer['a'])
        rewards = np.ascontiguousarray(self.buffer['r'])
        next_states = np.ascontiguousarray(self.buffer['s_'])
        dones = np.ascontiguousarray(self.buffer['done'])
        old_log_probs = np.ascontiguousarray(self.buffer['a_logp'])
        
        s = torch.from_numpy(states).double().to(self.device)
        a = torch.from_numpy(actions).long().to(self.device).view(-1, 1)
        r = torch.from_numpy(rewards).double().to(self.device).view(-1, 1)
        s_ = torch.from_numpy(next_states).double().to(self.device)
        done = torch.from_numpy(dones).bool().to(self.device).view(-1, 1)
        old_a_logp = torch.from_numpy(old_log_probs).double().to(self.device).view(-1, 1)

        with torch.no_grad():
            _, values = self.net(s)
            _, next_values = self.net(s_)
            
            # Compute GAE advantages
            advantages = torch.zeros_like(r)
            returns = torch.zeros_like(r)
            
            # Compute GAE in a vectorized way to avoid threading issues
            deltas = r + self.gamma * next_values * (1.0 - done.float()) - values
            
            # Compute GAE backwards
            gae = 0
            for step in reversed(range(self.buffer_capacity)):
                if step == self.buffer_capacity - 1:
                    next_non_terminal = 1.0 - done[step].float()
                else:
                    next_non_terminal = 1.0 - done[step].float()
                
                gae = deltas[step] + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[step] = gae
                returns[step] = gae + values[step]
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        for _ in range(self.ppo_epoch):
            # Use smaller batches to reduce memory pressure
            indices = torch.randperm(self.buffer_capacity)
            for start in range(0, self.buffer_capacity, self.batch_size):
                end = min(start + self.batch_size, self.buffer_capacity)
                batch_indices = indices[start:end]
                
                probs, value = self.net(s[batch_indices])
                dist = Categorical(probs)
                a_logp = dist.log_prob(a[batch_indices].squeeze()).view(-1, 1)
                ratio = torch.exp(a_logp - old_a_logp[batch_indices])

                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages[batch_indices]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, returns[batch_indices])
                loss = action_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
