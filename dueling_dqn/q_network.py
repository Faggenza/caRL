import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        # After conv layers: (96->22->10->8) for each dimension
        self.conv_output_size = 64 * 8 * 8  # 4096
        
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(512, 256)
        self.fc_adv = nn.Linear(512, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, 5)

    def forward(self, state):
        # Ensure state is in the right format: (batch_size, channels, height, width)
        if len(state.shape) == 3:  # Single image: (height, width, channels)
            state = state.permute(2, 0, 1).unsqueeze(0)  # -> (1, channels, height, width)
        elif len(state.shape) == 4 and state.shape[3] == 3:  # Batch of images: (batch, height, width, channels)
            state = state.permute(0, 3, 1, 2)  # -> (batch, channels, height, width)
        
        # Normalize pixel values to [0, 1]
        state = state.float() / 255.0
        
        # Convolutional layers
        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten
        x = x.contiguous().view(x.size(0), -1)
        
        # Fully connected layers
        y = self.relu(self.fc1(x))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

