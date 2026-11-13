
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
        buffer_capacity=50000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.0005,
        target_update_interval=1000,
        hidden_dims=(24, 24),
        gradient_clip=10.0
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval

        # Gradient clipping configuration
        self.gradient_clip = gradient_clip

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Online network
        self.online_network = QNetwork(
            state_dim,
            action_dim,
            hidden_dims
        ).to(self.device)

        # Target network
        self.target_network = QNetwork(
            state_dim,
            action_dim,
            hidden_dims
        ).to(self.device)

        self.update_target_network()

        self.target_network.eval()

        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_capacity
        )

        self.learning_steps = 0

        # Existing diagnostic statistics
        self.last_q_stats = None

        # Gradient clipping diagnostics
        self.total_gradient_updates = 0
        self.clipped_gradient_updates = 0
        self.gradient_norm_sum = 0.0
        self.max_gradient_norm = 0.0

    # ========================================================
    # SELECT ACTION
    # ========================================================

    def select_action(self, state, epsilon):

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state = torch.as_tensor(
            np.asarray(state),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(state)

        return q_values.argmax(dim=1).item()

    # ========================================================
    # STORE EXPERIENCE
    # ========================================================

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

    # ========================================================
    # UPDATE TARGET NETWORK
    # ========================================================

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    # ========================================================
    # GRADIENT CLIPPING STATISTICS
    # ========================================================

    def get_gradient_clipping_stats(self):

        if self.total_gradient_updates == 0:

            return {
                "total_updates": 0,
                "clipped_updates": 0,
                "clipping_percentage": 0.0,
                "mean_gradient_norm": 0.0,
                "max_gradient_norm": 0.0
            }

        return {

            "total_updates": self.total_gradient_updates,

            "clipped_updates": self.clipped_gradient_updates,

            "clipping_percentage": (
                100.0
                * self.clipped_gradient_updates
                / self.total_gradient_updates
            ),

            "mean_gradient_norm": (
                self.gradient_norm_sum
                / self.total_gradient_updates
            ),

            "max_gradient_norm": self.max_gradient_norm
        }

    # ========================================================
    # LEARN
    # ========================================================

    def learn(self):

        # Wait until enough experiences are available
        if len(self.replay_buffer) < self.batch_size:

            self.last_q_stats = None

            return None

        # Sample random minibatch
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert to tensors
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

        # Current Q-values for selected actions
        current_q_values = self.online_network(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Bellman target
        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            ).max(dim=1)[0]

            target_q_values = (
                rewards
                + self.gamma * (1 - dones) * next_q_values
            )

        # TD error
        td_errors = target_q_values - current_q_values

        # MSE loss
        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        # Gradient norm before clipping
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=self.gradient_clip
        )

        gradient_norm_value = gradient_norm.item()

        # Update clipping diagnostics
        self.total_gradient_updates += 1

        self.gradient_norm_sum += gradient_norm_value

        self.max_gradient_norm = max(
            self.max_gradient_norm,
            gradient_norm_value
        )

        if gradient_norm_value > self.gradient_clip:

            self.clipped_gradient_updates += 1

        # Optimizer update
        self.optimizer.step()

        # Store existing diagnostic statistics
        self.last_q_stats = {

            "mean_current_q": (
                current_q_values.detach().mean().item()
            ),

            "max_abs_current_q": (
                current_q_values.detach().abs().max().item()
            ),

            "mean_target_q": (
                target_q_values.detach().mean().item()
            ),

            "mean_abs_td_error": (
                td_errors.detach().abs().mean().item()
            ),

            "max_abs_td_error": (
                td_errors.detach().abs().max().item()
            ),

            "gradient_norm": gradient_norm_value
        }

        # Update target network periodically
        self.learning_steps += 1

        if (
            self.learning_steps % self.target_update_interval == 0
        ):

            self.update_target_network()

        return loss.item()
