import numpy as np
import gymnasium as gym

class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, env, img_stack=4, action_repeat=4):
        self.env = env
        self.img_stack = img_stack
        self.action_repeat = action_repeat
        self.reward_threshold = self.env.spec.reward_threshold
        self.action_dim = self.env.action_space.n

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        self.die = False
        img_rgb, _ = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
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
        assert len(self.stack) == self.img_stack
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
