import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from utils import plot_training_progress

path = '/home/faggi/repo/caRL/fatto/ppo_256b_1400ep_GAE.pkl'

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args(args=[])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


class Env():
    """
    Test environment wrapper for CarRacing
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v3', continuous=False, render_mode='human')
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
        # La testa del critico (v) non viene usata nel test ma deve essere definita
        # per caricare correttamente lo state_dict del modello salvato.
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
        # Per il test abbiamo bisogno solo dell'output dell'attore
        probs = self.actor_head(x)
        return probs, None


class Agent():
    """
    Agent for testing
    """
    def __init__(self, action_dim):
        self.net = Net(action_dim).double().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            # Selezione dell'azione deterministica (la più probabile)
            probs, _ = self.net(state)
            action = torch.argmax(probs).item()
        return action

    def load_param(self):
        checkpoint = torch.load(path, map_location=device)
        self.net.load_state_dict(checkpoint['ppo_net_params_discrete'])

if __name__ == "__main__":
    env = Env()
    agent = Agent(env.action_dim)
    agent.load_param()
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and 'scores' in checkpoint:
        all_scores = checkpoint['scores']
        all_running_scores = checkpoint['running_scores']
        all_episodes = checkpoint['episodes']
        i_ep = checkpoint['i_ep']

        plot_training_progress(all_scores, all_episodes)


    test_rewards = []
    for i_ep in range(10):
        score = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_action(state)
            state_, reward, done, die = env.step(action)
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        test_rewards.append(score)

        print(f'Ep {i_ep}\tScore: {score:.2f}\t')

    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    max_reward = np.max(test_rewards)
    min_reward = np.min(test_rewards)

    print(f"\n=== TEST RESULTS ===")
    print(f"Episodes tested: {10}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Training episodes: {i_ep}")
    print(f"Training average (last 100 episodes): {np.mean(all_scores[-100:]):.2f}")

