
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path

from src.environment import make_env, get_env_info
from src.network import QNetwork


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate():

    # Configuration
    seed = 1000
    num_episodes = 100

    set_seed(seed)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create environment
    env = make_env()

    state_dim, action_dim = get_env_info(env)

    # Initialize Q-network
    q_network = QNetwork(
        state_dim,
        action_dim
    ).to(device)

    # Load trained model
    model_path = "results/raw/dqn_model.pth"

    q_network.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    q_network.eval()

    print("\nDQN Model Evaluation")
    print("--------------------")
    print("Environment: CartPole-v1")
    print("Evaluation episodes:", num_episodes)
    print("Exploration rate (epsilon): 0")
    print("Device:", device)

    episode_rewards = []

    # Evaluation loop
    for episode in range(num_episodes):

        state, info = env.reset(seed=seed + episode)

        total_reward = 0
        done = False

        while not done:

            # Convert state to tensor
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            # Greedy action selection
            with torch.no_grad():

                q_values = q_network(state_tensor)

                action = q_values.argmax(dim=1).item()

            # Environment interaction
            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            done = terminated or truncated

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:

            print(
                f"Episode {episode + 1:3d}: "
                f"Reward = {total_reward:.0f}"
            )

    env.close()

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)

    # Save evaluation results
    Path("results/raw").mkdir(
        parents=True,
        exist_ok=True
    )

    Path("results/figures").mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        "results/raw/evaluation_rewards.npy",
        np.array(episode_rewards)
    )

    # Print summary
    print("\n========== EVALUATION SUMMARY ==========")

    print("Episodes:", num_episodes)
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Standard deviation: {std_reward:.2f}")
    print(f"Minimum reward: {min_reward:.0f}")
    print(f"Maximum reward: {max_reward:.0f}")

    print("========================================")

    # Plot evaluation rewards
    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, num_episodes + 1),
        episode_rewards,
        marker="o",
        markersize=3,
        label="Evaluation Reward"
    )

    plt.axhline(
        mean_reward,
        linestyle="--",
        label=f"Mean Reward = {mean_reward:.2f}"
    )

    plt.xlabel("Evaluation Episode")
    plt.ylabel("Total Reward")

    plt.title("DQN Evaluation on CartPole-v1")

    plt.grid(True)
    plt.legend()

    plt.savefig(
        "results/figures/evaluation_rewards.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    evaluate()
