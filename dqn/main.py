import gymnasium as gym
import math
import random
from itertools import count
import os

import numpy as np

from test_dqn import test_model, test
from plot import plot_test
from train import optimize_model
import torch
import torch.optim as optim

from memory import ReplayMemory
from q_network import DQN

def main(resume_from_checkpoint=False):
    
    latest_path = "saved_models/dqn_model.pt"

    # if GPU is to be used
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
        import matplotlib.pyplot as plt
        from plot import plot_durations
        render = "human"

    env = gym.make("CarRacing-v3", render_mode=render, lap_complete_percent=0.95, domain_randomize=False,
                       continuous=False)

    # To ensure reproducibility during training, you can fix the random seeds
    # by uncommenting the lines below. This makes the results consistent across
    # runs, which is helpful for debugging or comparing different approaches.
    #
    # That said, allowing randomness can be beneficial in practice, as it lets
    # the model explore different training trajectories.


    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 65000
    TAU = 0.005
    LR = 3e-4
    NUM_EPISODES = 1000
    TEST_INTERVAL = 50

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = state.flatten().shape[0]  # Appiattisce l'immagine e conta tutti i pixel

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0
    start_episode = 1
    episode_durations = []
    train_rewards = []
    if resume_from_checkpoint:
        try:
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            start_episode = checkpoint['episode'] + 1
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            steps_done = checkpoint.get('steps_done', 0)
            episode_durations = checkpoint.get('episode_durations', [])
            train_rewards = checkpoint.get('train_rewards', [])
            print(f"Resuming from episode {start_episode}")
        except FileNotFoundError:
            os.makedirs(os.path.dirname("saved_models"), exist_ok=True)
            print("No checkpoint found, starting fresh.")

    def select_action(state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    for i_episode in range(start_episode, NUM_EPISODES + 1):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0

        for _ in count():
            steps_done+= 1
            action = select_action(state, steps_done)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            if terminated:
                reward += 100
            if np.mean(observation[:, :, 1]) > 185.0:
                reward -= 0.05
            episode_reward += reward
            avg_reward = reward_mem(reward)
            if avg_reward <= -0.1:
                done = True
            if done:
                break

            reward = torch.tensor([reward], device=device)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(
                memory=memory,
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                device=device,
                batch_size=BATCH_SIZE,
                gamma=GAMMA
            )

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                torch.save({
                    'episode': i_episode,
                    'model_state_dict': policy_net_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_rewards': train_rewards,
                    'episode_durations': episode_durations,
                    'steps_done': steps_done,
                    'device': str(device)
                }, latest_path)
                break

        train_rewards.append(episode_reward)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

        if i_episode % 5 == 0:
            print(f'Episode {i_episode}: 'f'Train reward: {train_rewards[-1]:.2f} |'  f' Epsilon: {eps_threshold:.4f}')
        if i_episode % TEST_INTERVAL == 0:
            test_model()

    print('Complete')
    if plot_flag:
        plot_durations(show_result=True, episode_durations=episode_durations)
        plt.ioff()
        plt.show()


if __name__ == "__main__":

    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


    # Inizializza la funzione di memoria dei reward
    reward_mem = reward_memory()
    #main(resume_from_checkpoint=True)
    test(first_time=True)


