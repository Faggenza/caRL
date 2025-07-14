import numpy as np
import torch
from plot import plot_training_progress
from ppo.network import Agent, AgentGAE


def test_ppo(device, gae_lambda, path, env, img_stack=4, test_episodes=10):
    torch.set_num_threads(1)
    
    if gae_lambda == 0:
        agent = Agent(env.action_dim, path=path, device=device, img_stack=img_stack)
    else:
        agent = AgentGAE(env.action_dim, path=path, device=device, img_stack=img_stack)
        
    agent.load_param()
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and 'scores' in checkpoint:
        all_scores = checkpoint['scores']
        all_episodes = checkpoint['episodes']
        i_ep = checkpoint['i_ep']

        plot_training_progress(all_scores, all_episodes)
    

    test_rewards = []
    for i in range(test_episodes):
        score = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_test_action(state)
            state_, reward, done, die = env.step(action)
            #if args.render:
            #    env.render()
            score += reward
            state = state_
            if done or die:
                break

        test_rewards.append(score)

        print(f'Ep {i}\tScore: {score:.2f}\t')

    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    max_reward = np.max(test_rewards)
    min_reward = np.min(test_rewards)

    print(f"\n=== TEST RESULTS ===")
    print(f"Episodes tested: {test_episodes}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Training episodes: {i_ep}")
    print(f"Training average (last 100 episodes): {np.mean(all_scores[-100:]):.2f}")
    print(f"=======================\n")

