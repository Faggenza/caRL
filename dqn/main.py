import gymnasium as gym
import math
import random
from itertools import count
from train import optimize_model
import torch
import torch.optim as optim

from memory import ReplayMemory
from q_network import DQN

def main(resume_from_checkpoint=False):
    latest_path = "saved_models/dqn_model.pt"

    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False,
                         continuous=False)

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    if device == torch.device("cuda"):
        plot_flag = False
    else:
        plot_flag = True
        import matplotlib.pyplot as plt
        from plot import plot_durations

    # To ensure reproducibility during training, you can fix the random seeds
    # by uncommenting the lines below. This makes the results consistent across
    # runs, which is helpful for debugging or comparing different approaches.
    #
    # That said, allowing randomness can be beneficial in practice, as it lets
    # the model explore different training trajectories.


    # seed = 42
    # random.seed(seed)
    # torch.manual_seed(seed)
    # env.reset(seed=seed)
    # env.action_space.seed(seed)
    # env.observation_space.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

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
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4


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
    if resume_from_checkpoint:
        try:
            checkpoint = torch.load(latest_path, map_location=device)
            start_episode = checkpoint['episode'] + 1
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            steps_done = checkpoint.get('steps_done', 0)
        except FileNotFoundError:
            print("No checkpoint found, starting fresh.")

    def select_action(state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    episode_durations = []
    for i_episode in range(start_episode, num_episodes):
        print(f'Starting episode {i_episode}')
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            steps_done+= 1
            action = select_action(state, steps_done)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

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
                    'train_rewards': reward,
                    'steps_done': steps_done,
                    #'test_rewards': test_rewards,
                    #'policy_losses': policy_losses,
                    #'value_losses': value_losses,
                    'device': str(device)  # Save device information
                }, latest_path)
                episode_durations.append(t + 1)
                break

    print('Complete')
    if plot_flag:
        plot_durations(show_result=True, episode_durations=episode_durations)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main(resume_from_checkpoint=True)

