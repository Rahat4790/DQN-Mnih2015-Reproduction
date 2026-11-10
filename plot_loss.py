
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load saved training losses
losses = np.load("results/raw/episode_losses.npy")

# Create output directory
Path("results/figures").mkdir(parents=True, exist_ok=True)

# Plot training losses
plt.figure(figsize=(10, 5))

plt.plot(losses, label="Average Training Loss")

plt.xlabel("Episode")
plt.ylabel("Mean Squared TD Loss")
plt.title("DQN Training Loss on CartPole-v1")

plt.grid(True)
plt.legend()

# Save graph
plt.savefig(
    "results/figures/training_losses.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Loss graph saved successfully!")
print("Number of episodes:", len(losses))
print("Average loss:", np.mean(losses))
