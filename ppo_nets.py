import torch

'''
The actor implements the policy, and the critic predicts its estimated value. 
Both actor and critic neural networks take the same input—the state at each timestep.
'''
class BaseNetwork(torch.nn.Module):
    
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_features, hidden_dimensions)
        self.layer2 = torch.nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = torch.nn.Linear(hidden_dimensions, out_features)
        self.dropout = torch.nn.Dropout(dropout) # to face overfitting

    # TODO vedere se ci sono altre activation functions che possano andare bene oltre relu
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

'''
Le due reti Actor e Critic hanno la stessa struttura ma vengono aggiornate separatamente.
Entrambe usano lo stato come input per il forward pass
'''
class ActorCritic(torch.nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        
    def to(self, device):
        super().to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        return self
        
    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred
    