import gymnasium as gym
import torch
from env import *
from ppo_agent import evaluate, create_agent
from plot import *

def test_model():
    test_reward = 0
    first_time = True
    for i in range(NUM_TEST):
        test_reward += test(first_time)
        first_time = False
    test_reward /= NUM_TEST
    print(f"Average test reward over {NUM_TEST} runs: {test_reward:.2f}")
    return test_reward

def test(first_time):
    latest_path = "saved_models/ppo_model.pt"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    if device == torch.device("cuda"):
        plot_flag = False
        render = "rgb_array"
    else:
        plot_flag = True
        render = "human"
        import matplotlib.pyplot as plt

    env = gym.make("CarRacing-v3", render_mode=render, lap_complete_percent=0.95, domain_randomize=False,
                   continuous=False)


    # Get number of actions and observations like in main()
    state, _ = env.reset()

    checkpoint = torch.load(latest_path, map_location=device)
    agent, device = create_agent(env, hidden_dimensions=HIDDEN_DIMENSIONS, dropout=0.2, device=device)
    agent.load_state_dict(checkpoint['model_state_dict'])

    train_rewards = checkpoint.get('train_rewards', [])
    test_rewards = checkpoint.get('test_rewards', [])
    policy_losses = checkpoint.get('policy_losses', [])
    value_losses = checkpoint.get('value_losses', [])

    episode_reward = evaluate(env, agent, device)

    if plot_flag and first_time:
        plot_train_rewards(train_rewards, REWARD_THRESHOLD)
        plot_test_rewards(test_rewards, REWARD_THRESHOLD)
        plot_losses(policy_losses, value_losses)

    return episode_reward