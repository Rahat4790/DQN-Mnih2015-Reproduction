
import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(

            # Input layer -> Hidden layer 1
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            # Hidden layer 1 -> Hidden layer 2
            nn.Linear(128, 128),
            nn.ReLU(),

            # Hidden layer 2 -> Output layer
            nn.Linear(128, action_dim)
        )

    def forward(self, state):

        q_values = self.network(state)

        return q_values
