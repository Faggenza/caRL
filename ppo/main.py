import gymnasium as gym
import numpy as np
import torch
import os

from env import *
from test_ppo import test_model
from ppo_agent import create_agent, evaluate
from torch import optim
from test_ppo import *
from train import forward_pass, update_policy

def run_ppo(resume_from, device, plot_flag):
    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []

    latest_path = "saved_models/ppo_model.pt"

    agent, device = create_agent(HIDDEN_DIMENSIONS, DROPOUT, device)
    
    # Use a smaller learning rate for stability
    optimizer = optim.AdamW(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Create directory for saved models if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Starting episode
    start_episode = 1
    
    # Load model if resuming training
    if resume_from:
        checkpoint = torch.load(latest_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        train_rewards = checkpoint['train_rewards']
        test_rewards = checkpoint['test_rewards']
        policy_losses = checkpoint['policy_losses']
        value_losses = checkpoint['value_losses']
        print(f"Resuming training from episode {start_episode}, loaded from {resume_from}")
    for episode in range(start_episode, MAX_EPISODES+1):
        train_reward, states, actions, actions_log_probability, advantages, returns = forward_pass(
                env_train,
                agent,
                optimizer,
                DISCOUNT_FACTOR,
                device)
        policy_loss, value_loss, _ = update_policy(
                agent,
                states,
                actions,
                actions_log_probability,
                advantages,
                returns,
                optimizer,
                PPO_STEPS,
                EPSILON,
                ENTROPY_COEFFICIENT,
                device)

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(train_reward)
        #mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        #mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
        #mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))
        
        # Also save as latest model (overwrite)
        
        torch.save({
            'episode': episode,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_rewards': train_rewards,
            'test_rewards': test_rewards,
            'policy_losses': policy_losses,
            'value_losses': value_losses,
            'device': str(device)  # Save device information
        }, latest_path)
        
        if episode % PRINT_INTERVAL == 0:
            # Evaluate the agent on the test environment every PRINT_INTERVAL episodes
            #test_reward = evaluate(env_test, agent, device)
            #test_rewards.append(test_reward)
            #mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

            print(f'Episode: {episode:3} | Train Rewards: {train_reward:3.1f} \
                  | Policy Loss: {policy_loss:2.2f} \
                  | Value Loss: {value_loss:2.2f}')

        if episode % TEST_INTERVAL == 0:
            episode_reward = test_model()
            test_rewards.append(episode_reward)
            mean_test_rewards = np.mean(test_rewards)

            if mean_test_rewards >= REWARD_THRESHOLD:
                best_path = "ppo/saved_models/best_model.pt"
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_rewards': train_rewards,
                    'test_rewards': test_rewards,
                    'policy_losses': policy_losses,
                    'value_losses': value_losses,
                    'device': str(device)
                }, best_path)
                print(f'Reached reward threshold in {episode} episodes')
                print(f'Best model saved to {best_path}')
                break

    if plot_flag:
        from plot import plot_train_rewards, plot_test_rewards, plot_losses
        plot_train_rewards(train_rewards, REWARD_THRESHOLD)
        plot_test_rewards(test_rewards, REWARD_THRESHOLD)
        plot_losses(policy_losses, value_losses)
    
def set_seed(seed=42, env=None):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def main():   
    global env_train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo dispositivo: {device}")

    if device == torch.device("cuda"):
        plot_flag = False
        MODE = 'rgb_array'
    else:
        plot_flag = True
        MODE = 'human'
    
    env_train = gym.make("CarRacing-v3", render_mode=MODE, lap_complete_percent=0.95, domain_randomize=False, continuous=False)

    set_seed(42, env_train)

    run_ppo(False, device, plot_flag)
    env_train.close()

if __name__ == "__main__":
    main()
    # test(first_time=True)
