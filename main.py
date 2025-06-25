import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
env.reset()  # Reset the environment to start a new episode
env.step(env.action_space.sample())  # Take a random action to initialize the environment
env.render()  # Render the environment
env.close()  # Close the environment when done