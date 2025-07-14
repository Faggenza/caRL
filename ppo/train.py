import numpy as np
from plot import plot_training_progress
from ppo.test import test_ppo
from ppo.network import Agent


def train_ppo(device, env, img_stack=4, gamma=0.99,
              test_interval=10, test_episodes=5, print_interval=10,
              path='saved_models/model.pt', buffer_capacity=2000,
              ppo_epoch=8, batch_size=256, clip_param=0.2, epochs=1000):
    transition = np.dtype([('s', np.float64, (img_stack, 96, 96)), ('a', np.int64), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (img_stack, 96, 96))])
    
    agent = Agent(env.action_dim, path=path, device=device, img_stack=img_stack,
                  transition=transition, gamma=gamma, ppo_epoch=ppo_epoch,
                  buffer_capacity=buffer_capacity, batch_size=batch_size, clip_param=clip_param)

    all_scores = []
    all_episodes = []

    state = env.reset()

    for i_ep in range(1, epochs + 1):
        score = 0
        state = env.reset()

        for _ in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action)
            #if args.render:
            #    env.render()
            if agent.store((state, action, a_logp, reward, state_)):
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
            test_ppo(device, path, env, img_stack=img_stack, test_episodes=test_episodes)
            pass
    env.close()
    print("Training finisched")