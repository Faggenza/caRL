import argparse
import torch
import gymnasium as gym
import random
import numpy as np

torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Training and testing RL models")

    # Algorithm selection
    parser.add_argument("--algorithm", type=str, choices=["dqn", "dueling_dqn", "ppo"], 
                        default="dqn", help="RL algorithm to use (dqn, dueling_dqn, ppo)")
    # Test 
    parser.add_argument("--test", action="store_true", help="Run in testing mode (default is training mode)")
    parser.add_argument("--test_episodes", type=int, default=10, help="Number of episodes for testing")

    
    # Common hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--hidden_size", type=int, default=64, help="Size of hidden layers")
    parser.add_argument("--test_interval", type=int, default=50, help="Interval for testing during training")
    parser.add_argument("--print_interval", type=int, default=10, help="Interval for printing training progress")
    
    
    # DQN/Dueling specific hyperparameters
    parser.add_argument("--epsilon_start", type=float, default=0.9, help="Starting value of epsilon for epsilon-greedy policy")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final value of epsilon for epsilon-greedy policy")
    parser.add_argument("--epsilon_decay", type=float, default=65000, help="Decay rate of epsilon")
    parser.add_argument("--tau", type=float, default=0.005, help="Rate for soft update of target network")
    parser.add_argument("--replay_memory_size", type=int, default=10000, help="Size of replay memory")
    # Just for Dueling DQN
    parser.add_argument("--update_steps", type=int, default=4, help="Number of steps to update the network")
    
    
    # PPO specific hyperparameters
    parser.add_argument("--ppo_epoch", type=int, default=8, help="Number of PPO epochs per update")
    parser.add_argument("--buffer_capacity", type=int, default=2000, help="Capacity of the PPO buffer")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--action_repeat", type=int, default=8, help="Number of action repeats in PPO")
    parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
    # PPO GAE
    parser.add_argument("--gae_lambda", type=float, default=0.0, help="Lambda for GAE")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="saved_models/model.pt", help="Path to save/load model")
    parser.add_argument("--load", type=str, help="Load model from the specified path")
    return parser.parse_args()

def train(args, env, device):
    print("Starting training")

    match args.algorithm:
        case "dqn":
            print("DQN-specific parameters:")
            print(f"  Model Path: {args.model_path}")
            print(f"  Batch Size: {args.batch_size}")
            print(f"  Gamma (Discount Factor): {args.gamma}")
            print(f"  Epsilon: {args.epsilon_start} → {args.epsilon_end} (decay: {args.epsilon_decay})")
            print(f"  Tau (Target Network Update Rate): {args.tau}")
            print(f"  Learning Rate: {args.learning_rate}")
            print(f"  Replay Memory Size: {args.replay_memory_size}")
            print(f"  Test Interval: {args.test_interval}")
            print(f"  Test Episodes: {args.test_episodes}")
            print(f"  Print Interval: {args.print_interval}")
            print(f"  Training Epochs: {args.epochs}")
            from dqn.train_dqn import train_dqn
            train_dqn(path=args.model_path, device=device, 
                      batch_size=args.batch_size, gamma=args.gamma, 
                      epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, 
                      epsilon_decay=args.epsilon_decay, tau=args.tau, 
                      learning_rate=args.learning_rate, 
                      replay_memory_size=args.replay_memory_size, 
                      test_interval=args.test_interval,
                      test_episodes=args.test_episodes,
                      print_interval=args.print_interval,
                      epochs=args.epochs, env=env)
        case "dueling_dqn":
            print("Dueling-DQN-specific parameters:")
            print(f"  Model Path: {args.model_path}")
            print(f"  Batch Size: {args.batch_size}")
            print(f"  Gamma (Discount Factor): {args.gamma}")
            print(f"  Epsilon: {args.epsilon_start} → {args.epsilon_end} (decay: {args.epsilon_decay})")
            print(f"  Tau (Target Network Update Rate): {args.tau}")
            print(f"  Learning Rate: {args.learning_rate}")
            print(f"  Replay Memory Size: {args.replay_memory_size}")
            print(f"  Update Steps: {args.update_steps}")
            print(f"  Test Interval: {args.test_interval}")
            print(f"  Test Episodes: {args.test_episodes}")
            print(f"  Print Interval: {args.print_interval}")
            print(f"  Training Epochs: {args.epochs}")
            from dueling_dqn.train_dueling_dqn import train_dueling_dqn
            train_dueling_dqn(path=args.model_path, device=device,
                              batch_size=args.batch_size, gamma=args.gamma,
                              epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                              epsilon_decay=args.epsilon_decay, tau=args.tau,
                              learning_rate=args.learning_rate,
                              replay_memory_size=args.replay_memory_size,
                              epochs=args.epochs, update_steps=args.update_steps,
                              test_interval=args.test_interval,
                              test_episodes=args.test_episodes,
                              print_interval=args.print_interval, env=env)
        case "ppo":
            print("PPO-specific parameters:")
            print(f"  Action Repeat: {args.action_repeat}")
            print(f"  Image Stack: {args.img_stack}")
            print(f"  GAE Lambda: {args.gae_lambda}")
            print(f"  PPO Epochs: {args.ppo_epoch}")
            print(f"  Buffer Capacity: {args.buffer_capacity}")
            print(f"  Clip Parameter: {args.clip_param}")
            print(f"  Batch Size: {args.batch_size}")
            print(f"  Gamma (Discount Factor): {args.gamma}")
            print(f"  Learning Rate: {args.learning_rate}")
            print(f"  Test Interval: {args.test_interval}")
            print(f"  Test Episodes: {args.test_episodes}")
            print(f"  Print Interval: {args.print_interval}")
            print(f"  Training Epochs: {args.epochs}")
            print(f"  Model Path: {args.model_path}")
            
            if args.gae_lambda < 0 or args.gae_lambda > 1:
                raise ValueError("GAE lambda must be between 0 and 1")
            
            # Wrap the environment
            from ppo.env import Env
            wrapped_env = Env(env=env, img_stack=args.img_stack, action_repeat=args.action_repeat)

            if args.gae_lambda == 0:
                print("Using PPO without GAE")
                from ppo.train import train_ppo
                
                train_ppo(path=args.model_path, device=device, img_stack=args.img_stack,
                          epochs=args.epochs, gamma=args.gamma, test_interval=args.test_interval,
                          test_episodes=args.test_episodes, print_interval=args.print_interval,
                          env=wrapped_env, buffer_capacity=args.buffer_capacity,
                          ppo_epoch=args.ppo_epoch, batch_size=args.batch_size, clip_param=args.clip_param)
            else:
                print("Using PPO with GAE")
                from ppo.train_gae import train_ppo_gae
                train_ppo_gae(path=args.model_path, device=device, img_stack=args.img_stack,
                              epochs=args.epochs, gamma=args.gamma, test_interval=args.test_interval,
                              test_episodes=args.test_episodes, print_interval=args.print_interval,
                              env=wrapped_env, buffer_capacity=args.buffer_capacity, gae_lambda=args.gae_lambda,
                              ppo_epoch=args.ppo_epoch, batch_size=args.batch_size, clip_param=args.clip_param)
        case _:
            print(f"Unknown algorithm: {args.algorithm}")
            return
def test(args, env, device):
    if not args.load:
        print("No model specified for testing. Use --load or --model_path to specify a model.")
        return
    print("Starting testing:")
    print(f"  Test episodes: {args.test_episodes}")
    print(f"  Model path: {args.load}")
    print(f"  Algorithm: {args.algorithm}")
    match args.algorithm:
        case "dqn":
            from dqn.test_dqn import test
            from plot import plot_training_progress
            checkpoint = torch.load(args.load, map_location=torch.device(device))
            policy_net_state_dict = checkpoint['model_state_dict']
            train_rewards = checkpoint.get('train_rewards', [])
            test(device=device, policy_net_state_dict=policy_net_state_dict,
                 train_rewards=train_rewards, num_episodes=args.test_episodes, env=env)
            plot_training_progress(scores=train_rewards, episodes=list(range(1, len(train_rewards) + 1)))
        case "dueling_dqn":
            from dueling_dqn.test_dueling_dqn import test_dueling
            from plot import plot_training_progress
            checkpoint = torch.load(args.load, map_location=torch.device(device))
            dueling_dqn_param = checkpoint['model_state_dict']
            train_rewards = checkpoint.get('train_rewards', [])
            test_dueling(dueling_dqn_param=dueling_dqn_param, train_rewards=train_rewards, device=device,
                         num_episodes=args.test_episodes, env=env)
            plot_training_progress(scores=train_rewards, episodes=list(range(1, len(train_rewards) + 1)))
        case "ppo":
            from ppo.test_ppo import test_ppo
            # Wrap the environment
            from ppo.env import Env
            wrapped_env = Env(env=env, img_stack=args.img_stack, action_repeat=args.action_repeat)
            test_ppo(device=device, gae_lambda=args.gae_lambda, path=args.load, env=wrapped_env, img_stack=args.img_stack, test_episodes=args.test_episodes)
    
def set_seed(seed, device, env):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif device.type == 'mps':
        torch.backends.mps.manual_seed(seed)
        
        
        
def main():
    args = parse_args()
        
    render = 'human' if args.test else 'rgb_array'
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    
    env = gym.make("CarRacing-v3", render_mode=render, lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    
    set_seed(args.seed, device, env)
    
    if args.test:
        test(args, env, device)
    else:
        train(args, env, device)

if __name__ == "__main__":
    main()