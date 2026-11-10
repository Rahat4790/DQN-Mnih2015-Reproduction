
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.network import QNetwork
from src.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_capacity=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        target_update_interval=1000
    ):

        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Online network
        self.online_network = QNetwork(
            state_dim, action_dim
        ).to(self.device)

        # Target network
        self.target_network = QNetwork(
            state_dim, action_dim
        ).to(self.device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        # Target network does not receive gradients
        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

        # Replay memory
        self.replay_buffer = ReplayBuffer(
            buffer_capacity
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate
        )

        # Loss function
        self.loss_function = nn.MSELoss()

        self.learning_steps = 0

    def select_action(self, state, epsilon):

        # Exploration
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        # Exploitation
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.online_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def store_experience(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def learn(self):

        # Wait until enough experiences are available
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a random batch
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert data into PyTorch tensors
        states = torch.as_tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.as_tensor(
            np.asarray(actions),
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.as_tensor(
            np.asarray(rewards),
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.as_tensor(
            np.asarray(dones),
            dtype=torch.float32,
            device=self.device
        )

        # Q-values for the actions actually taken
        current_q_values = self.online_network(
            states
        ).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate Bellman targets
        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            ).max(dim=1)[0]

            target_q_values = (
                rewards
                + self.gamma * (1 - dones) * next_q_values
            )

        # Calculate TD loss
        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # Gradient descent
        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        # Update target network periodically
        self.learning_steps += 1

        if self.learning_steps % self.target_update_interval == 0:

            self.update_target_network()

        return loss.item()
