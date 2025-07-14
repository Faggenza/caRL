import gymnasium as gym
import torch
import numpy as np
from itertools import count
from dqn.q_network import DQN


def test(policy_net_state_dict, train_rewards, device, env, num_episodes=5):

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = state.flatten().shape[0]

    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(policy_net_state_dict)
    policy_net.eval()

    episodes = list(range(1, len(train_rewards) + 1))
    
    # Test dell'agente
    test_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            # Use only the greedy policy (no epsilon-greedy)
            tensor_state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(tensor_state).max(1).indices.view(1, 1)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            episode_reward += reward
            step_count += 1
            state = next_state

            # Safety limit to avoid infinite loops
            if step_count > 2000:
                print(f"Episode {episode + 1} stopped after {step_count} steps")
                break

        test_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Score = {episode_reward:.2f}, Steps = {step_count}")

    env.close()

    # Final statistics
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    max_reward = np.max(test_rewards)
    min_reward = np.min(test_rewards)

    print(f"\n=== TEST RESULTS ===")
    print(f"Episodes tested: {num_episodes}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Training episodes: {episodes.__len__()}")
    print(f"Training average (last 100 episodes): {np.mean(train_rewards[-100:]):.2f}")
    print(f"=======================\n")



    return test_rewards