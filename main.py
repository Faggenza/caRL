import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
# Number of steps you run the agent for 
num_steps = 1500

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    action = env.action_space.sample()
    
    # apply the action
    obs, rew, done, trunc, info  = env.step(action)
    
    # Render the env
    env.render()
    
    # If the epsiode is up, then start another one
    if done:
        env.reset()

# Close the env
env.close()