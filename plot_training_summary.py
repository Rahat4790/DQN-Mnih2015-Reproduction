
import numpy as np
import matplotlib.pyplot as plt

# Load existing 1500-episode results
episode_rewards = np.load(
    "results/raw/episode_rewards.npy"
)

episode_losses = np.load(
    "results/raw/episode_losses.npy"
)

# Create two plots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Episode rewards
axes[0].plot(episode_rewards)
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total Reward")
axes[0].set_title("DQN Training Rewards")
axes[0].grid(True)

# Plot 2: Training losses
axes[1].plot(episode_losses)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Mean Squared TD Loss")
axes[1].set_title("DQN Training Loss")
axes[1].grid(True)

# Main title
fig.suptitle("DQN Training on CartPole-v1 (1500 Episodes)")

plt.tight_layout()

# Save combined figure
plt.savefig(
    "results/figures/training_summary_1500.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Combined reward and loss graphs saved successfully!")
print("Number of episodes:", len(episode_rewards))
print("Number of loss records:", len(episode_losses))
