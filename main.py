import gymnasium as gym
import numpy as np
import torch
import os
from ppo_agent import create_agent, evaluate
from torch import optim
from train import forward_pass, update_policy
from plot import plot_train_rewards, plot_test_rewards, plot_losses

def run_ppo(resume_from=None):
    MAX_EPISODES = 500
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 64
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []
    
    latest_path = "saved_models/latest_model.pt"
    
    # Create the agent
    agent = create_agent(env_train, HIDDEN_DIMENSIONS, DROPOUT)
    
    # Use a smaller learning rate for stability
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Add gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
    
    # Create directory for saved models if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Starting episode
    start_episode = 1
    
    # Load model if resuming training
    if resume_from:
        checkpoint = torch.load(latest_path)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        train_rewards = checkpoint['train_rewards']
        test_rewards = checkpoint['test_rewards']
        policy_losses = checkpoint['policy_losses']
        value_losses = checkpoint['value_losses']
        print(f"Resuming training from episode {start_episode}, loaded from {resume_from}")
    for episode in range(start_episode, MAX_EPISODES+1):
        print(f'Starting episode {episode}')
        train_reward, states, actions, actions_log_probability, advantages, returns = forward_pass(
                env_train,
                agent,
                optimizer,
                DISCOUNT_FACTOR)
        policy_loss, value_loss = update_policy(
                agent,
                states,
                actions,
                actions_log_probability,
                advantages,
                returns,
                optimizer,
                PPO_STEPS,
                EPSILON,
                ENTROPY_COEFFICIENT)
        test_reward = evaluate(env_test, agent)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
        mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))
        
        # Also save as latest model (overwrite)
        
        torch.save({
            'episode': episode,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_rewards': train_rewards,
            'test_rewards': test_rewards,
            'policy_losses': policy_losses,
            'value_losses': value_losses
        }, latest_path)
        
        if episode % PRINT_INTERVAL == 0:
            print(f'Episode: {episode:3} | \
                  Mean Train Rewards: {mean_train_rewards:3.1f} \
                  | Mean Test Rewards: {mean_test_rewards:3.1f} \
                  | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                  | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
            
        if mean_test_rewards >= REWARD_THRESHOLD:
            best_path = "saved_models/best_model.pt"
            torch.save({
                'episode': episode,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_rewards': train_rewards,
                'test_rewards': test_rewards,
                'policy_losses': policy_losses,
                'value_losses': value_losses
            }, best_path)
            print(f'Reached reward threshold in {episode} episodes')
            print(f'Best model saved to {best_path}')
            break
    plot_train_rewards(train_rewards, REWARD_THRESHOLD)
    plot_test_rewards(test_rewards, REWARD_THRESHOLD)
    plot_losses(policy_losses, value_losses)
    
def main():   
    global env_train, env_test
    env_train = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    env_test = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    
    run_ppo(True)
    env_train.close()
    env_test.close()

if __name__ == "__main__":
    main()
