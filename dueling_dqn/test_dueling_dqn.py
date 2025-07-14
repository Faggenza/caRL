import numpy as np
import torch
from dueling_dqn.q_network import QNetwork

path = '/home/faggi/repo/caRL/fatto/dueling_dqn_1us_64b_900.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_dueling(dueling_dqn_param, train_rewards, device, num_episodes=10, env=None):

    agent = QNetwork().to(device)
    agent.load_state_dict(dueling_dqn_param)
    agent.eval()
    
    test_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Use only the greedy policy (no epsilon-greedy)
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.select_action(tensor_state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            # Limite di sicurezza per evitare loop infiniti
            if step_count > 2000:
                print(f"Episode {episode + 1} stopped after {step_count} steps")
                break
        
        test_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Score = {episode_reward:.2f}, Steps = {step_count}")
        
        env.close()
        
        # Statistiche finali
        avg_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        max_reward = np.max(test_rewards)
        min_reward = np.min(test_rewards)
        
        print(f"\n=== TEST RESULTS ===")
        print(f"Episodes tested: {num_episodes}")
        print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Max reward: {max_reward:.2f}")
        print(f"Min reward: {min_reward:.2f}")
        print(f"Training episodes: {episode}")
        print(f"Training average (last 100 episodes): {np.mean(train_rewards[-100:]):.2f}")
        print(f"=======================\n")

        return test_rewards
