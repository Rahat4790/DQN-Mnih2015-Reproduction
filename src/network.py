
import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(24, 24)
    ):
        super(QNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        # Build configurable hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):

        q_values = self.network(state)

        return q_values
