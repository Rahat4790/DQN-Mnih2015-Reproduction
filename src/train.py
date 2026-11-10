import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.environment import make_env, get_env_info
from src.dqn_agent import DQNAgent


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():

    # Reproducibility
    seed = 42
    set_seed(seed)

    # Training configuration
    num_episodes = 2000

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    # Environment
    env = make_env(seed=seed)

    state_dim, action_dim = get_env_info(env)

    # Agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_capacity=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.0003,
        target_update_interval=1000
    )

    episode_rewards = []
    episode_losses = []

    # Training loop
    for episode in tqdm(range(num_episodes)):

        if episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        total_reward = 0
        losses = []

        done = False

        while not done:

            # Select action
            action = agent.select_action(
                state, epsilon
            )

            # Interact with environment
            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated

            # Store experience
            agent.store_experience(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            # Learn
            loss = agent.learn()

            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        # Decay exploration
        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        # Record performance
        episode_rewards.append(total_reward)

        average_loss = (
            np.mean(losses) if losses else 0.0
        )

        episode_losses.append(average_loss)

        # Print progress
        if (episode + 1) % 25 == 0:

            recent_reward = np.mean(
                episode_rewards[-25:]
            )

            print(
                f"Episode: {episode + 1}, "
                f"Average Reward: {recent_reward:.2f}, "
                f"Epsilon: {epsilon:.3f}, "
                f"Loss: {average_loss:.4f}"
            )

    env.close()

    # Save training results
    np.save(
        "results/raw/episode_rewards.npy",
        np.array(episode_rewards)
    )

    np.save(
        "results/raw/episode_losses.npy",
        np.array(episode_losses)
    )
    # Save trained online network
    torch.save(
        agent.online_network.state_dict(),
        "results/raw/dqn_model.pth"
    )

    print("Trained model saved successfully!")
    
    # Plot rewards and losses together

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Reward curve
    axes[0].plot(
        episode_rewards,
        label="Episode Reward"
    )

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("DQN Training Rewards")
    axes[0].grid(True)
    axes[0].legend()

    # Loss curve
    axes[1].plot(
        episode_losses,
        label="Average TD Loss"
    )

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Mean Squared TD Loss")
    axes[1].set_title("DQN Training Loss")
    axes[1].grid(True)
    axes[1].legend()

    # Overall figure title
    fig.suptitle(
        f"DQN Training on CartPole-v1 ({num_episodes} Episodes)",
        fontsize=15
    )

    plt.tight_layout()

    # Save combined figure
    plt.savefig(
        "results/figures/training_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # Final training summary
    print("\n========== TRAINING SUMMARY ==========")

    print("Total episodes:", num_episodes)

    print(
        "Final 25-episode average reward:",
        round(np.mean(episode_rewards[-25:]), 2)
    )

    print(
        "Maximum episode reward:",
        round(np.max(episode_rewards), 2)
    )

    print(
        "Average training loss:",
        round(np.mean(episode_losses), 4)
    )

    print("Final epsilon:", round(epsilon, 4))

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train()
