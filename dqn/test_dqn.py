import gymnasium as gym
import torch
from itertools import count
from q_network import DQN

NUM_TEST = 5

def test_model():
    test_reward = 0
    first_time = True
    for i in range(NUM_TEST):
        test_reward += test(first_time)
        first_time = False
    test_reward /= NUM_TEST
    print(f"Average test reward over {NUM_TEST} runs: {test_reward:.2f}")


def test(first_time):
    latest_path = "saved_models/dqn_model.pt"
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
        from plot import plot_test

    env = gym.make("CarRacing-v3", render_mode=render, lap_complete_percent=0.95, domain_randomize=False,
                   continuous=False)


    # Get number of actions and observations like in main()
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = state.flatten().shape[0]

    policy_net = DQN(n_observations, n_actions).to(device)
    checkpoint = torch.load(latest_path, map_location=device)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    policy_net.to(device)
    policy_net.eval()

    # Estrai le metriche dal checkpoint
    train_rewards = checkpoint.get('train_rewards', [])
    episodes = list(range(1, len(train_rewards) + 1))

    if plot_flag and first_time:
        plot_test(train_rewards, episodes)

    # Test dell'agente
    state, _ = env.reset()
    state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
    test_rewards = 0
    for t in count():
        action = policy_net(state).max(1).indices.view(1, 1)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        test_rewards += reward
        done = terminated or truncated

        if done:
            break

        state = torch.tensor(observation.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
    print(f"Test finished after {t + 1} timesteps with total reward: {test_rewards:.2f}")
    env.close()
    return test_rewards