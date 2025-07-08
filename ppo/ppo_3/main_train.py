import argparse
import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import DrawLine, plot_training_progress

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args(args=[])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed(args.seed)

transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.int64), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])


class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v3', continuous=False)
        self.env.reset(seed=args.seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.action_dim = self.env.action_space.n

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        self.die = False
        img_rgb, _ = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if terminated:
                reward += 100
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            if self.av_r(reward) <= -0.1:
                done = True
            if done:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == args.img_stack
        return np.array(self.stack), total_reward, done, terminated

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        if norm:
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)
        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)
        return memory


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, action_dim):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2), nn.ReLU(),
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
    clip_param = 0.2
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self, action_dim):
        self.training_step = 0
        self.net = Net(action_dim).double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.net(state)
        dist = Categorical(probs)
        action = dist.sample()
        a_logp = dist.log_prob(action)
        return action.item(), a_logp.item()

    def save_param(self, scores, running_scores, episodes, i_ep):
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        torch.save({'ppo_net_params_discrete': self.net.state_dict(),
                    'scores': scores,
                   'running_scores': running_scores,
                    'episodes': episodes,
                    'i_ep': i_ep
                    }, 'saved_models/ppo_net_params_discrete.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def load_param(self):
        checkpoint = torch.load('saved_models/ppo_net_params_discrete.pkl', map_location=device)
        # Se il checkpoint contiene il nuovo formato con dizionario
        if isinstance(checkpoint, dict) and 'ppo_net_params_discrete' in checkpoint:
            self.net.load_state_dict(checkpoint['ppo_net_params_discrete'])
        else:
            # Formato vecchio - solo state_dict
            self.net.load_state_dict(checkpoint)

    def update(self):
        self.training_step += 1
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.long).to(device).view(-1, 1)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            _, target_v = self.net(s_)
            target_v = r + args.gamma * target_v
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


if __name__ == "__main__":
    args.vis = True
    env = Env()
    agent = Agent(env.action_dim)

    all_scores = []
    all_running_scores = []
    all_episodes = []
    initial_ep = 0

    if args.load:
        try:
            checkpoint = torch.load('saved_models/ppo_net_params_discrete.pkl', map_location=device)
            agent.load_param()

            # Carica i dati aggiuntivi se disponibili (nuovo formato)
            if isinstance(checkpoint, dict) and 'scores' in checkpoint:
                all_scores = checkpoint['scores']
                all_running_scores = checkpoint['running_scores']
                all_episodes = checkpoint['episodes']
                initial_ep = checkpoint['i_ep'] + 1  # Continua dall'episodio successivo
                running_score = all_running_scores[-1] if all_running_scores else 0
                print(f'Loaded model at episode {checkpoint["i_ep"]} with running score {running_score:.2f}')
            else:
                print('Loaded model with old format - starting fresh with episode counting')
        except FileNotFoundError:
            print('No saved model found - starting fresh')
            args.load = False
    else:
        print('Training fresh')


    training_records = []
    running_score = all_running_scores[-1] if all_running_scores else 0
    state = env.reset()

    for i_ep in range(initial_ep, 100000):
        score = 0
        state = env.reset()

        for t in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action)
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01

        all_scores.append(score)
        all_running_scores.append(running_score)
        all_episodes.append(i_ep)

        if i_ep % args.log_interval == 0:
            print(f'Ep {i_ep}\tLast score: {score:.2f}\tMoving average score: {running_score:.2f}')
            agent.save_param(all_scores, all_running_scores, all_episodes, i_ep)
        if i_ep % 5 == 0 and i_ep > 0:
            plot_training_progress(all_scores, all_running_scores, all_episodes)
        if running_score > env.reward_threshold:
            print(f"Solved! Running reward is now {running_score} and the last episode runs to {score}!")
            break