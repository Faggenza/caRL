import matplotlib
import torch
from matplotlib import pyplot as plt

def plot_durations(show_result=False, episode_durations=None):
    if episode_durations is None:
        episode_durations = []
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_test(train_rewards=None, episodes=None):
    # Plotta le metriche
    import matplotlib.pyplot as plt
    import numpy as np

    if train_rewards:
        fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))

        # Plot train rewards
        ax1.plot(episodes, train_rewards, 'b-', linewidth=1, alpha=0.7, label='Valori originali')

        # Media mobile di tutto il plot per i rewards
        window_size = min(len(train_rewards) // 10, 200)  # Finestra del 10% dei dati, max 50
        if window_size > 1:
            moving_avg = np.convolve(train_rewards, np.ones(window_size) / window_size, mode='valid')
            # Allinea gli episodi con la media mobile
            episodes_aligned = episodes[window_size - 1:]
            ax1.plot(episodes_aligned, moving_avg, 'r-', linewidth=2, label=f'Media mobile ({window_size} episodi)')

        ax1.set_xlabel('Episodi')
        ax1.set_ylabel('Train Reward')
        ax1.set_title('Train Rewards durante il Training')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        plt.tight_layout()
        plt.savefig('./plots/train_rewards_plot_dqn.png')
        plt.show()

        print(f"Statistiche finali:")
        print(f"Episodi totali: {len(train_rewards)}")
        print(f"Finestra media mobile rewards: {window_size} episodi")
        print(f"Train reward medio (ultimi 100): {np.mean(train_rewards[-100:]):.2f}")
    else:
        print("Nessuna metrica trovata nel checkpoint")