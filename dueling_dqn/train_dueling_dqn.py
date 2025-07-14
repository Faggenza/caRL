import math
import torch
import numpy as np
import random
from itertools import count
import torch.nn.functional as F
from plot import plot_training_progress
import os
from dueling_dqn.q_network import QNetwork
from dueling_dqn.memory import Memory
from dueling_dqn.test_dueling_dqn import test_dueling


def save_param(rewards, episodes, i_ep, QNetork, path):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save({'dueling-dqn-param': QNetork,
                'rewards': rewards,
                'episodes': episodes,
                'i_ep': i_ep
                }, path)

def train_dueling_dqn(path, device, batch_size=64, gamma=0.99,
                      epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=65,
                      tau=0.005, learning_rate=3e-4, replay_memory_size=10000,
                      epochs=1000, update_steps=4, test_interval=50,
                      test_episodes=5, print_interval=10, env=None):

    
    agent = QNetwork().to(device)
    targetNetwork = QNetwork().to(device)
    targetNetwork.load_state_dict(agent.state_dict())

    optimizer = torch.optim.AdamW(agent.parameters(), learning_rate)

    memory_replay = Memory(replay_memory_size)

    epsilon = epsilon_start
    learn_steps = 0

    rewards = []

    for epoch in range(1, epochs + 1):
        state, _ = env.reset()
        episode_reward = 0
        for _ in count():
            p = random.random()
            if p < epsilon:
                action = random.randint(0, 1)
            else:
                tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.select_action(tensor_state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            memory_replay.add((state, next_state, action, reward, done))
            if memory_replay.size() > 128:
                learn_steps += 1

                batch = memory_replay.sample(batch_size, False)
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

                batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
                batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
                batch_action = torch.FloatTensor(np.array(batch_action)).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device)

                with torch.no_grad():
                    onlineQ_next = agent(batch_next_state)
                    targetQ_next = targetNetwork(batch_next_state)
                    online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
                    y = batch_reward + (1 - batch_done) * gamma * targetQ_next.gather(1, online_max_action.long())

                loss = F.mse_loss(agent(batch_state).gather(1, batch_action.long()), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if learn_steps % update_steps == 0:
                    target_net_state_dict = targetNetwork.state_dict()
                    policy_net_state_dict = agent.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (
                                    1 - tau)
                    targetNetwork.load_state_dict(target_net_state_dict)

                if epsilon > epsilon_end:
                    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * epoch / epsilon_decay)

            if done:
                rewards.append(episode_reward)
                break
            state = next_state
            
        if epoch % print_interval == 0:
            print(f'Episode {epoch}: Train reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f}')
            plot_training_progress(rewards, list(range(1, epoch + 1)))
            save_param(rewards, list(range(1, epoch + 1)), epoch, agent.state_dict(), path)
        
        if epoch % test_interval == 0:
            test_dueling(dueling_dqn_param=agent.state_dict(), train_rewards=rewards, device=device, num_episodes=test_episodes, env=env)
            
    env.close()
    print('Training finished')