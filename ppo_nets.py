import torch

class BackboneNetwork(torch.nn.Module):
    
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_features, hidden_dimensions)
        self.layer2 = torch.nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = torch.nn.Linear(hidden_dimensions, out_features)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x
    
class ActorCritic(torch.nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred
    