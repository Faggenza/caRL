import matplotlib.pyplot as plt
import os


class DrawLine:
    """
    Classe per disegnare e salvare un grafico usando matplotlib.
    """

    def __init__(self, title, xlabel, ylabel, legend):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x_data = []
        self.y_data = [[] for _ in legend]

        self.lines = [self.ax.plot([], [], label=l)[0] for l in legend]

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()
        self.ax.grid(True)

    def __call__(self, xdata, ydata):
        """
        Aggiunge nuovi dati al grafico e lo aggiorna.
        xdata: un singolo valore per l'asse x.
        ydata: una lista o tupla di valori per l'asse y, uno per ogni linea.
        """
        self.x_data.append(xdata)
        for i, y in enumerate(ydata):
            self.y_data[i].append(y)

        for i, line in enumerate(self.lines):
            line.set_data(self.x_data, self.y_data[i])

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

    def save(self, filename="training_plot.png"):
        """Salva il grafico corrente in un file PNG."""
        if not os.path.exists('plots'):
            os.makedirs('plots')
        filepath = os.path.join('plots', filename)
        self.fig.savefig(filepath)

    def close(self):
        """Chiude la finestra del grafico."""
        plt.ioff()
        plt.close(self.fig)


def plot_training_progress(scores=None, episodes=None):
    """
    Plotta lo score e la media mobile durante il training PPO.

    Args:
        scores: Lista degli score per episodio
        running_scores: Lista delle medie mobili
        episodes: Lista dei numeri di episodio
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if scores and episodes:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

        # Plot degli score originali
        ax1.plot(episodes, scores, 'b-', linewidth=1, alpha=0.7, label='Valori originali')

        # Calcola una media mobile aggiuntiva per smoothing se ci sono abbastanza dati
        if len(scores) > 20:
            window_size = 200  # Finestra del 30% dei dati, max 200
            if window_size > 1:
                moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
                episodes_aligned = episodes[window_size - 1:]
                ax1.plot(episodes_aligned, moving_avg, 'r-', linewidth=1.5, alpha=0.8,
                         label=f'Media mobile ({window_size} episodi)')

        ax1.set_xlabel('Episodi')
        ax1.set_ylabel('Train Reward')
        ax1.set_title('Train Rewards durante il Training')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        plt.tight_layout()
        if not os.path.exists('plots'):
            os.makedirs('plots')
        filepath = os.path.join('plots', 'ppo_training_progress.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    else:
        print("Dati insufficienti per il plot")
