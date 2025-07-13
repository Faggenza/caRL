import os

def plot_training_progress(scores=None, episodes=None):
    """
    Plotta lo score e la media mobile durante il training PPO.

    Args:
        scores: Lista degli score per episodio
        episodes: Lista dei numeri di episodio
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if scores and episodes:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

        # Plot degli score originali
        ax1.plot(episodes, scores, 'b-', linewidth=1, alpha=0.7, label='Score episodio')

        # Plot della media mobile (running_score)
        # ax1.plot(episodes, running_scores, 'r-', linewidth=2, label='Media mobile (running score)')

        # Calcola una media mobile aggiuntiva per smoothing se ci sono abbastanza dati
        if len(scores) > 20:
            window_size = 200
            if window_size > 1:
                moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
                episodes_aligned = episodes[window_size - 1:]
                ax1.plot(episodes_aligned, moving_avg, 'r-', linewidth=1.5, alpha=0.8,
                         label=f'Media mobile ({window_size} episodi)')

        ax1.set_xlabel('Episodi')
        ax1.set_ylabel('Score')
        ax1.set_title('Dueling-DQN Training Progress - Score e Media Mobile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Aggiungi statistiche come testo sul grafico
        if len(scores) > 0:
            last_episodes = min(100, len(scores))
            recent_avg = np.mean(scores[-last_episodes:])
            #recent_running_avg = running_scores[-1] if running_scores else 0

            stats_text = f"Ultimi {last_episodes} episodi:\n"
            stats_text += f"Score medio: {recent_avg:.2f}\n"
            #stats_text += f"Running score: {recent_running_avg:.2f}\n"
            stats_text += f"Score massimo: {max(scores):.2f}"

            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        if not os.path.exists('plots'):
            os.makedirs('plots')
        filepath = os.path.join('plots', 'dueling_dqn_training_progress.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close(fig)

    else:
        print("Dati insufficienti per il plot")
