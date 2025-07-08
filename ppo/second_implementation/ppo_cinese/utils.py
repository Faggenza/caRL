import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()
        # L'input è in scala di grigi, quindi ha 1 canale
        in_channels = 1

        self.cnn_base = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calcola la dimensione dell'output della CNN in modo dinamico
        with torch.no_grad():
            # state_dim è (H, W), es: (84, 84)
            dummy_input = torch.zeros(1, in_channels, *state_dim)
            cnn_out_dim = self.cnn_base(dummy_input).view(1, -1).size(1)

        self.fc1 = nn.Linear(cnn_out_dim, net_width)
        self.fc_pi = nn.Linear(net_width, action_dim)

    def forward(self, state):
        # Gestisce sia un singolo stato (H, W) che un batch (B, H, W)
        # Aggiunge la dimensione del canale per creare (B, C, H, W)
        if len(state.shape) == 3:  # Batch di stati (B, H, W)
            state = state.unsqueeze(1)
        elif len(state.shape) == 2:  # Singolo stato (H, W)
            state = state.unsqueeze(0).unsqueeze(0) # -> (1, 1, H, W)

        x = self.cnn_base(state)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc_pi(x)

    def pi(self, state, softmax_dim=1):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=softmax_dim)
        return probs

class Critic(nn.Module):
    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()
        # L'input è in scala di grigi, quindi ha 1 canale
        in_channels = 1

        self.cnn_base = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calcola la dimensione dell'output della CNN in modo dinamico
        with torch.no_grad():
            # state_dim è (H, W), es: (84, 84)
            dummy_input = torch.zeros(1, in_channels, *state_dim)
            cnn_out_dim = self.cnn_base(dummy_input).view(1, -1).size(1)

        self.fc1 = nn.Linear(cnn_out_dim, net_width)
        self.fc_v = nn.Linear(net_width, 1)

    def forward(self, state):
        # Gestisce sia un singolo stato (H, W) che un batch (B, H, W)
        # Aggiunge la dimensione del canale per creare (B, C, H, W)
        if len(state.shape) == 3:  # Batch di stati (B, H, W)
            state = state.unsqueeze(1)
        elif len(state.shape) == 2:  # Singolo stato (H, W)
            state = state.unsqueeze(0).unsqueeze(0) # -> (1, 1, H, W)

        x = self.cnn_base(state)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc_v(x)

def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a, logprob_a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise