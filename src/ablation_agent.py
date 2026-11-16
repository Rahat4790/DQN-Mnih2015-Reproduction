
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.network import QNetwork
from src.replay_buffer import ReplayBuffer


class AblationDQNAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        ablation_mode="main",
        buffer_capacity=50000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.0005,
        target_update_interval=1000,
        hidden_dims=(24, 24),
        gradient_clip=10.0
    ):

        if ablation_mode not in ["main", "no_target", "no_replay"]:
            raise ValueError("Invalid ablation mode")

        self.ablation_mode = ablation_mode

        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.gradient_clip = gradient_clip

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Online network: always required
        self.online_network = QNetwork(
            state_dim,
            action_dim,
            hidden_dims
        ).to(self.device)

        # Target network: omitted in no_target mode
        if self.ablation_mode != "no_target":

            self.target_network = QNetwork(
                state_dim,
                action_dim,
                hidden_dims
            ).to(self.device)

            self.update_target_network()
            self.target_network.eval()

            for parameter in self.target_network.parameters():
                parameter.requires_grad = False

        else:
            self.target_network = None

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

        # Replay buffer: omitted in no_replay mode
        if self.ablation_mode != "no_replay":

            self.replay_buffer = ReplayBuffer(
                buffer_capacity
            )

        else:
            self.replay_buffer = None

        self.last_experience = None
        self.learning_steps = 0
        self.last_q_stats = None

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

    def store_experience(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        if self.ablation_mode == "no_replay":
            self.last_experience = experience

        else:
            self.replay_buffer.add(*experience)

    def update_target_network(self):

        if self.target_network is not None:

            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

    def learn(self):

        
        # Select training data
        if self.ablation_mode == "no_replay":

            if self.last_experience is None:
                return None

            state, action, reward, next_state, done = (
                self.last_experience
            )

            states = [state]
            actions = [action]
            rewards = [reward]
            next_states = [next_state]
            dones = [done]

        else:

            if len(self.replay_buffer) < self.batch_size:
                return None

            (
                states,
                actions,
                rewards,
                next_states,
                dones
            ) = self.replay_buffer.sample(self.batch_size)


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

        current_q_values = self.online_network(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():

            if self.ablation_mode == "no_target":

                next_q_values = self.online_network(
                    next_states
                ).max(dim=1)[0]

            else:

                next_q_values = self.target_network(
                    next_states
                ).max(dim=1)[0]

            target_q_values = (
                rewards
                + self.gamma * (1 - dones) * next_q_values
            )

        td_errors = target_q_values - current_q_values

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=self.gradient_clip
        )

        self.optimizer.step()

        self.learning_steps += 1

        # Update target network only when it exists
        if (
            self.ablation_mode != "no_target"
            and self.learning_steps % self.target_update_interval == 0
        ):

            self.update_target_network()

        self.last_q_stats = {

            "mean_current_q": current_q_values.detach().mean().item(),

            "mean_target_q": target_q_values.detach().mean().item(),

            "mean_abs_td_error": td_errors.detach().abs().mean().item()
        }

        return loss.item()
