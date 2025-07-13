import gymnasium as gym
import torch
import numpy as np
from itertools import count
from q_network import DQN

NUM_TEST = 10


def test(first_time):
    latest_path = "saved_models/dqn_model_512_65k_700ep.pt"
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
    test_rewards = []

    for episode in range(NUM_TEST):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            # Usa solo la policy greedy (no epsilon-greedy)
            tensor_state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(tensor_state).max(1).indices.view(1, 1)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            episode_reward += reward
            step_count += 1
            state = next_state

            # Limite di sicurezza per evitare loop infiniti
            if step_count > 2000:
                print(f"Episode {episode + 1} stopped after {step_count} steps")
                break

        test_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{NUM_TEST}: Score = {episode_reward:.2f}, Steps = {step_count}")

    env.close()

    # Statistiche finali
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    max_reward = np.max(test_rewards)
    min_reward = np.min(test_rewards)

    print(f"\n=== TEST RESULTS ===")
    print(f"Episodes tested: {NUM_TEST}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Training episodes: {episodes}")
    print(f"Training average (last 100 episodes): {np.mean(train_rewards[-100:]):.2f}")


    return test_rewards