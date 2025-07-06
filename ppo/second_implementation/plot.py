import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(train_rewards, eval_rewards, save_interval, eval_interval):
    """Plot training and evaluation rewards over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training rewards (per step)
    ax1.plot(train_rewards, alpha=0.3, color='blue', label='Training Rewards')
    # Moving average for training rewards
    window = 1000
    if len(train_rewards) > window:
        moving_avg = np.convolve(train_rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(train_rewards)), moving_avg, color='red',
                 label=f'Moving Average ({window} steps)')

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards per Step')
    ax1.legend()
    ax1.grid(True)

    # Plot evaluation rewards
    eval_steps = np.arange(0, len(eval_rewards) * eval_interval, eval_interval)
    ax2.plot(eval_steps, eval_rewards, 'o-', color='green', label='Evaluation Rewards')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Evaluation Rewards')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()