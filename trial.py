import gymnasium as gym
import numpy as np
env = gym.make("CarRacing-v3", render_mode="human", continuous=False)
env.reset()

action_sample = env.action_space.sample()  # Sample a random action

print("Random action: " + action_sample.__str__())
env.step(action_sample)

env.render()
env.close()