import numpy as np

from utils import evaluate_policy, str2bool
from datetime import datetime
from ppo import PPO_discrete
import gymnasium as gym
import os, shutil
import argparse
import torch
#from plot import plot_training_results
from gymnasium.wrappers import ResizeObservation, FlattenObservation

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')

parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--T_horizon', type=int, default=50, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=5e5, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=500, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=50, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=256, help='length of sliced trajectory')
parser.add_argument('--initial_entropy_coef', type=float, default=0.9, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=65000, help='Decay rate of entropy_coef')
parser.add_argument('--min_entropy_coef', type=float, default=0.01, help='Minimum entropy coefficient to ensure some exploration')
parser.add_argument('--initial_explore_steps', type=int, default=10000, help='Number of steps for initial high exploration')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

def main():
    # Build Training Env and Evaluation Env
    EnvName = ['CarRacing-v3']
    env = gym.make(EnvName[opt.EnvIdex], continuous=False, render_mode="human" if opt.render else "rgb_array")
    opt.max_e_steps = env._max_episode_steps

    env = ResizeObservation(env, (84, 84))  # Ridimensiona a 84x84
    env = FlattenObservation(env)  # Appiattisci a vettore 1D

    eval_env = gym.make(EnvName[opt.EnvIdex], continuous=False)
    eval_env = ResizeObservation(eval_env, (84, 84))
    eval_env = FlattenObservation(eval_env)

    opt.state_dim = env.observation_space.shape[0]  # Ora sarà 7056 (84*84)
    opt.action_dim = env.action_space.n

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Env:','CarRacing-v3','  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,'   Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps)
    print('\n')

    train_rewards = []
    eval_rewards = []

    # Use tensorboard to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format('CarRacing-v3') + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('saved_models'): os.mkdir('saved_models')
    agent = PPO_discrete(**vars(opt))
    if opt.Loadmodel:
        checkpoint = agent.load(latest_path="./saved_models/ppo_model.pth")
        train_rewards = checkpoint.get('train_rewards', [])
        eval_rewards = checkpoint.get('eval_rewards', [])
        agent.total_steps_taken = checkpoint.get('total_steps_taken', 0)
        print(f'Loading model from step {agent.total_steps_taken}...')
        

    if opt.render:
        #plot_training_results(train_rewards, eval_rewards, opt.save_interval, opt.eval_interval)
        while True:
            ep_r = evaluate_policy(env, agent, turns=1)
            print(f'Env: CarRacing-v3, Episode Reward:{ep_r}')
    else:
        traj_lenth = 0
        total_steps = 0 if not opt.Loadmodel else agent.total_steps_taken

        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False
            episode_reward = 0
            '''Interact & trian'''
            while not done:
                '''Interact with Env'''
                a, logprob_a = agent.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
                done = (dw or tr)
                episode_reward += r 
                
                if done:
                    train_rewards.append(episode_reward)
                
                '''Store the current transition'''
                agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
                s = s_next

                traj_lenth += 1
                total_steps += 1

                '''Update if its time'''
                if traj_lenth % opt.T_horizon == 0:
                    agent.train()
                    traj_lenth = 0

                '''Record & log'''
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns=3)
                    eval_rewards.append(score)
                    # plot_training_results(train_rewards, eval_rewards, opt.save_interval, opt.eval_interval)
                    if opt.write: writer.add_scalar('ep_r', score, global_step=total_steps)
                    print(f'  Episode: {total_steps // opt.T_horizon}',
                          f'  Train Rewards: {train_rewards[-1]:.2f}',
                          f'  Eval Rewards: {score:.2f}',
                          f'  Total Steps: {total_steps}',
                          f'  Entropy Coef: {agent.entropy_coef:.2f}')

                '''Save model'''
                if total_steps % opt.save_interval==0:
                    print(f'Saving model at step {total_steps}...')
                    agent.save(agent.total_steps_taken / opt.T_horizon, train_rewards, eval_rewards)
                    
                                
        env.close()
        eval_env.close()
        # Plot results
        # print("Plotting training results...")
        # plot_training_results(train_rewards, eval_rewards, opt.save_interval, opt.eval_interval)

if __name__ == '__main__':
    main()