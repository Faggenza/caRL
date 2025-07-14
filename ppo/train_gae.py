import numpy as np
import torch
from plot import plot_training_progress
from ppo.network import AgentGAE
from ppo.test_ppo import test_ppo

def train_ppo_gae(device, env, img_stack=4, gamma=0.99, gae_lambda=0.95,
              test_interval=10, test_episodes=5, print_interval=10,
              path='saved_models/model.pt', buffer_capacity=2000,
              ppo_epoch=8, batch_size=256, clip_param=0.2, epochs=1000):
    
    torch.set_num_threads(1)
    
    transition = np.dtype([('s', np.float64, (img_stack, 96, 96)), ('a', np.int64), ('a_logp', np.float64),
                ('r', np.float64), ('s_', np.float64, (img_stack, 96, 96)), ('done', np.bool_)])

    agent = AgentGAE(env.action_dim, path=path, device=device, img_stack=img_stack,
                  transition=transition, gamma=gamma, ppo_epoch=ppo_epoch, gae_lambda=gae_lambda,
                  buffer_capacity=buffer_capacity, batch_size=batch_size, clip_param=clip_param)
    all_scores = []
    all_episodes = []
    initial_ep = 0

    state = env.reset()

    for i_ep in range(initial_ep, 100000):
        score = 0
        state = env.reset()

        for _ in range(1, epochs + 1):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action)
            #if args.render:
            #    env.render()
            if agent.store((state, action, a_logp, reward, state_, done)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done or die:
                break

        all_scores.append(score)
        all_episodes.append(i_ep)

        if i_ep % print_interval == 0:
            print(f'Episode {i_ep}: Train reward: {score:.2f}')
            agent.save_param(all_scores, all_episodes, i_ep)
            plot_training_progress(all_scores, all_episodes)
        if i_ep % test_interval == 0:
            test_ppo(device, gae_lambda, path, env, img_stack=img_stack, test_episodes=test_episodes)
            pass
    env.close()
    print("Training finished")