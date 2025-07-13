import torch
import torch.nn as nn
import torch.optim as optim
from dqn.memory import Transition, ReplayMemory
from dqn.q_network import DQN
import random
import math
from dqn.test_dqn import test
import numpy as np
from plot import plot_training_progress
from itertools import count
import os

def optimize_model(memory, q_net, target_net, optimizer, device, batch_size, gamma):
    if len(memory) < batch_size:
        return None, None, None
    transitions = memory.sample(batch_size)
    # Students don't copy, steal:
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = q_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(q_net.parameters(), 100)
    optimizer.step()


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# EPSILON_DECAY is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# epsilon_decay controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# learning_rate is the learning rate of the ``AdamW`` optimizer  
# replay_memory_size is the size of the replay memory buffer
# epochs is the number of training episodes
# env is the environment to train on, e.g., CarRacing-v3
# path is the path to save the model
# device is the device to run the model on (CPU or GPU)
def train_dqn(path, device, batch_size=128, gamma=0.99, epsilon_start=0.9,
              epsilon_end=0.01, epsilon_decay=65000, tau=0.005,
              learning_rate=3e-4, replay_memory_size=10000, epochs=1000,
              test_interval=50, test_episodes=5,
              print_interval=10 ,env=None):
    
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
    
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = state.flatten().shape[0]  # Appiattisce l'immagine e conta tutti i pixel

    q_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.AdamW(q_net.parameters(), lr=learning_rate, amsgrad=True)
    memory = ReplayMemory(replay_memory_size)

    steps_done = 0
    episode_durations = []
    train_rewards = []

    def select_action(state, steps_done):
        sample = random.random()
        eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
            math.exp(-1. * steps_done / epsilon_decay)

        if eps_threshold < epsilon_end:
            eps_threshold = epsilon_end

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return q_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    for i_episode in range(1, epochs + 1):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0

        for _ in count():
            steps_done+= 1
            action = select_action(state, steps_done)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # if terminated:
            #     reward += 100
            # if np.mean(observation[:, :, 1]) > 185.0:
            #     reward -= 0.05
            # episode_reward += reward
            # avg_reward = reward_mem(reward)
            # if avg_reward <= -0.1:
            #     done = True
            episode_reward += reward
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
            optimize_model(memory=memory, q_net=q_net, target_net=target_net, optimizer=optimizer, device=device,
                           batch_size=batch_size, gamma=gamma)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = q_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                torch.save({
                    'episode': i_episode,
                    'model_state_dict': policy_net_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_rewards': train_rewards,
                    'episode_durations': episode_durations,
                    'steps_done': steps_done,
                    'device': str(device)
                }, path)
                break

        train_rewards.append(episode_reward)
        eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)

        if i_episode % print_interval == 0:
            print(f'Episode {i_episode}: Train reward: {train_rewards[-1]:.2f} | Epsilon: {eps_threshold:.4f}')
            plot_training_progress(train_rewards, list(range(1, i_episode + 1)))
        if i_episode % test_interval == 0:
            test(policy_net_state_dict, train_rewards, device, num_episodes=test_episodes, env=env)

    env.close()
    print('Training finished')