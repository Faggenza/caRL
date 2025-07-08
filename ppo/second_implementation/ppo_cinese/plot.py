import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(train_rewards, eval_rewards, save_interval, eval_interval):
    """Plot training and evaluation rewards over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Grafico delle ricompense di addestramento ---
    if train_rewards:
        # Estrai i passi e le ricompense dalla lista di tuple
        train_steps, rewards = zip(*train_rewards)

        ax1.plot(train_steps, rewards, alpha=0.3, color='blue', label='Training Rewards')

        # Media mobile per le ricompense di addestramento
        window = 50  # Finestra più piccola per una media più reattiva
        if len(rewards) > window:
            # Calcola la media mobile sulle ricompense
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
            # Calcola i passi corrispondenti per la media mobile
            moving_avg_steps = train_steps[window - 1:]
            ax1.plot(moving_avg_steps, moving_avg, color='red', label=f'Moving Average ({window} episodes)')

    ax1.set_xlabel('Total Steps')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards per Episode')
    ax1.legend()
    ax1.grid(True)

    # --- Grafico delle ricompense di valutazione ---
    if eval_rewards:
        eval_steps = np.arange(eval_interval, (len(eval_rewards) + 1) * eval_interval, eval_interval)
        ax2.plot(eval_steps, eval_rewards, 'o-', color='green', label='Evaluation Rewards')

    ax2.set_xlabel('Total Steps')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Evaluation Rewards')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # plt.savefig('training_evaluation_plot.png')
    plt.show()